"""Flow model class definitions for RoboSuite trajectory policies."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from tqdm import tqdm


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for 1D feature maps.

    Applies a per-channel affine transform conditioned on a context vector.

    Args:
    in_channels: Number of channels in the feature map to be modulated. (C)
    condition_dim: Dimensionality of the conditioning vector.

    Forward Args:
    x: Tensor of shape (B, C, T), feature map to modulate.
    condition: Tensor of shape (B, condition_dim), conditioning vector.

    Forward Returns:
    Tensor of shape (B, C, T) after FiLM modulation.
    """
    def __init__(self, in_channels, condition_dim):
        super().__init__()
        self.scale_shift = nn.Sequential(
            nn.Linear(condition_dim, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels * 2)
        )

    def forward(self, x, condition):
        # x: (B, C, T), condition: (B, cond_dim)
        scale_shift = self.scale_shift(condition)           # (B, 2*C)
        scale, shift = scale_shift.chunk(2, dim=-1)         # each (B, C)
        scale = scale.unsqueeze(-1)                        # (B, C, 1)
        shift = shift.unsqueeze(-1)                        # (B, C, 1)
        return x * (1 + scale) + shift


class ConvBlock(nn.Module):
    """Conv → GroupNorm → ReLU → FiLM block for 1D sequences.

    Args:
    in_channels: Input channels to the conv layer.
    out_channels: Output channels for the conv layer.
    condition_dim: Dimensionality of the conditioning vector used by FiLM.

    Forward Args:
    x: (B, C_in, T) input tensor.
    condition: (B, condition_dim) conditioning vector.

    Forward Returns:
    (B, C_out, T) tensor after convolution, normalization, activation, and FiLM.
    """
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.film = FiLM(out_channels, condition_dim)

    def forward(self, x, condition):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.film(x, condition)
        return x


class UNet1D(nn.Module):
    """A 1D U-Net with FiLM conditioning.

    This backbone models per-timestep dynamics across joint sequences.

    Args:
    dof: Number of channels == robot DoF to model.
    condition_dim: Dimensionality of the environmental condition fed into FiLM.
    channels: Encoder channel progression; decoder mirrors these.

    Forward Args:
    x: (B, dof, T) input sequence.
    condition: (B, condition_dim) vector.

    Forward Returns:
    (B, dof, T) tensor of predicted velocities (of the vector field).
    """
    def __init__(self, dof, condition_dim, channels=[160, 320, 640, 640]):
        super().__init__()
        self.channels = channels
        # Encoder
        self.enc_blocks = nn.ModuleList()
        in_ch = dof
        for ch in channels:
            self.enc_blocks.append(ConvBlock(in_ch, ch, condition_dim))
            in_ch = ch

        # Bottleneck
        self.bottleneck = ConvBlock(channels[-1], channels[-1], condition_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        prev_ch = channels[-1]
        for skip_ch in reversed(channels):
            in_ch = prev_ch + skip_ch
            out_ch = skip_ch
            self.dec_blocks.append(ConvBlock(in_ch, out_ch, condition_dim))
            prev_ch = out_ch

        # Final layer
        self.final_conv = nn.Conv1d(channels[0], dof, kernel_size=1)

    def forward(self, x, condition):
        # x: (B, dof, seq_len), condition: (B, cond_dim)
        skips = []
        # Encoding path
        for enc in self.enc_blocks:
            x = enc(x, condition)
            skips.append(x)
            x = F.avg_pool1d(x, kernel_size=2, ceil_mode=True)

        # Bottleneck
        x = self.bottleneck(x, condition)

        # Decoding path
        for dec, skip in zip(self.dec_blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x, condition)

        # Final projection
        return self.final_conv(x)



class VectorField(nn.Module):
    """Conditional vector field v(x, t, c) over joint trajectories.

    The model predicts a velocity field for every policy channel, including
    normalized gripper pose channels. Time is encoded via a small MLP and
    concatenated with (optionally scaled) environment parameters for FiLM
    conditioning.

    Args:
        seq_len: Number of time steps (T) per trajectory.
        dof: Total joint DoF per time step (includes grippers if present).
        param_len: Dimensionality of environment parameter vector `c`.
        gripper_idx: Optional gripper channel indices, kept for compatibility.
            These channels are still modeled and included in the loss.

    Attributes:
        loss_mask (Tensor): (1, 1, dof) all-ones mask kept for compatibility.

    Forward Args:
        x: (B, T, dof) current state on the straight path between x0 and x1.
        t: (B, 1) time in [0, 1].
        env_params: (B, param_len) environment parameters (scaled by 100 internally).

    Forward Returns:
        (B, T, dof) velocity field for all policy channels.
    """

    def __init__(self, seq_len, dof, param_len, gripper_idx=None):
        super().__init__()
        self.seq_len = seq_len
        self.dof = dof
        self.param_len = param_len

        if gripper_idx is None:
            gripper_idx = []
        if isinstance(gripper_idx, list):
            gripper_idx = torch.tensor(gripper_idx, dtype=torch.long)
        self.register_buffer("gripper_idx", gripper_idx, persistent=False)

        loss_mask = torch.ones(dof, dtype=torch.float32)
        self.register_buffer("loss_mask", loss_mask.view(1, 1, dof), persistent=False)

        # Time embedding for FiLM conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32)
        )

        # additional encoding if needed
        """
        self.param_embed = nn.Sequential(
            nn.Linear(param_len, 16), nn.ReLU(),
            nn.Linear(16, 16)
        )
        """

        # Condition dimension = param_len + time embedding dim
        condition_dim = 32 + param_len

        # 1D U-Net for sequence modeling
        self.unet = UNet1D(dof=dof, condition_dim=condition_dim)

    def forward(self, x, t, env_params):
        # x: (B, seq_len, dof), t: (B, 1), env_params: (B, param_len)
        dev, dt = x.device, x.dtype
        if torch.is_tensor(t):
            t = t.to(dev)
            if torch.is_floating_point(t):
                t = t.to(dt)
            else:
                t = torch.as_tensor(t, device=dev, dtype=dt)

        if torch.is_tensor(env_params):
            env_params = env_params.to(dev)
            if torch.is_floating_point(env_params):
                env_params = env_params.to(dt)
        else:
            env_params = torch.as_tensor(env_params, device=dev, dtype=dt)

        B, seq_len, dof = x.shape
        assert seq_len == self.seq_len and dof == self.dof, "Shape mismatch"

        # Normalize environment parameters (optional preprocessing)
        env_params_norm = env_params * 100

        # Encode timestep
        t_embed = self.time_embed(t)
        # p_embed = self.param_embed(env_params_norm) # in cased of additional encoding
        p_embed = env_params_norm # without environment parameter encoding

        # Create conditioning vector
        condition = torch.cat([p_embed, t_embed], dim=-1)  # (B, cond_dim)

        v = self.unet(x.permute(0, 2, 1).contiguous(), condition)
        return v.permute(0, 2, 1).contiguous()


class VanillaFM:
    """State-conditioned Flow Matching trainer for fixed-horizon policies."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        *,
        task_name: str,
        horizon: int,
        dof: int,
        condition_dim: int,
        gripper_idx=None,
        time_sampling: str = "uniform",
        beta_a: float = 1.5,
        beta_b: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task_name = task_name
        self.horizon = horizon
        self.dof = dof
        self.condition_dim = condition_dim
        self.gripper_idx = gripper_idx
        self.time_sampling = time_sampling
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.device = device

    def sample_t(self, n: int) -> torch.Tensor:
        if self.time_sampling == "shifted":
            t = Beta(self.beta_a, self.beta_b).sample((n, 1)).to(self.device)
            return torch.ones_like(t) - t
        return torch.rand(n, device=self.device).unsqueeze(-1)

    @torch.no_grad()
    def run_flow(self, x, c, n_steps=100):
        """Transport noisy samples through this policy's vector field with Euler steps."""
        dt = 1.0 / n_steps
        t = torch.empty((x.shape[0], 1), device=self.device, dtype=x.dtype)

        for i in range(n_steps):
            t.fill_(i * dt)
            v = self.model(x, t, c)
            x.add_(v, alpha=dt)

        return x

    def train(
        self,
        target_trajectories,
        conditions,
        *,
        n_t: int,
        max_epochs: int,
        batch_size: int,
        val_period: int = 5,
        early_stopping: bool = True,
        stop_criteria: int = 3,
        val_trials: int = 25,
    ):
        N = target_trajectories.shape[0]
        best_avg_reward = 0.0
        best_success_rate = 0.0
        best_model = copy.deepcopy(self.model)
        records = {}
        stop_count = 0

        do_validation = val_period > 0 and val_trials > 0
        if do_validation:
            from Robot_simulation.models.FM_util import _generate_val_env, eval_model

        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials)

        try:
            self.model = self.model.to(self.device)
            target_trajectories = target_trajectories.to(self.device)
            conditions = conditions.to(self.device)
            for epoch in tqdm(range(1, max_epochs + 1), desc=f"{self.time_sampling.title()}FM Training", unit="epoch"):
                self.model.train()
                perm_t = torch.randperm(N, device=self.device)
                loss_sum = 0.0
                for i in range(0, N, batch_size):
                    idx = perm_t[i:min(i + batch_size, N)]
                    x1 = target_trajectories[idx]
                    x0 = torch.randn(len(idx), self.horizon, self.dof, device=self.device)
                    t = self.sample_t(len(idx) * n_t)
                    cond = conditions[idx, :]

                    x1r = x1.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
                    x0r = x0.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
                    t_col = t.view(-1, 1, 1)
                    xt = (1 - t_col) * x0r + t_col * x1r
                    cond_r = cond.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, self.condition_dim)
                    target_v = x1r - x0r

                    pred_v = self.model(xt, t, cond_r)
                    sq_err = (pred_v - target_v) ** 2
                    if hasattr(self.model, "loss_mask") and self.model.loss_mask is not None:
                        sq_err = sq_err * self.model.loss_mask
                    loss = sq_err.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_sum += float(loss.item())

                if self.scheduler is not None:
                    self.scheduler.step()

                if do_validation and epoch % val_period == 0:
                    success_rate, avg_reward = eval_model(
                        self.model,
                        VectorField,
                        self.task_name,
                        self.horizon,
                        self.dof,
                        self.condition_dim,
                        self.gripper_idx,
                        val_params,
                        env_settings_all,
                        self.device,
                        trials=val_trials,
                    )
                    records[epoch] = {"success_rate": success_rate, "avg_reward": avg_reward, "loss": loss_sum}
                    if success_rate < best_success_rate:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
                        if early_stopping:
                            if stop_count == stop_criteria:
                                tqdm.write("Early stopping triggered.")
                                break
                            stop_count += 1
                    elif (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = copy.deepcopy(self.model)
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
        except KeyboardInterrupt:
            tqdm.write("Training interrupted by user. Returning best model so far...")

        if not records:
            best_model = copy.deepcopy(self.model)
        return best_model, self.model, records


