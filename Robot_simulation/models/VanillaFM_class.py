"""Flow model class definitions for RoboSuite trajectory policies."""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from tqdm import tqdm


class EMAModel:
    """Exponential moving average of model parameters."""

    def __init__(
        self,
        model,
        *,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
        min_decay: float = 0.0,
        max_decay: float = 0.9999,
    ):
        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_decay = float(min_decay)
        self.max_decay = float(max_decay)

        self.optimization_step = 0
        self.decay = 0.0

    def get_decay(self) -> float:
        step = max(
            0,
            self.optimization_step - self.update_after_step - 1,
        )

        if step <= 0:
            return 0.0

        value = 1.0 - (
            1.0 + step / self.inv_gamma
        ) ** (-self.power)

        return max(
            self.min_decay,
            min(value, self.max_decay),
        )

    @torch.no_grad()
    def step(self, model):
        self.decay = self.get_decay()

        for ema_param, param in zip(
            self.averaged_model.parameters(),
            model.parameters(),
        ):
            ema_param.mul_(self.decay)
            ema_param.add_(
                param.detach(),
                alpha=1.0 - self.decay,
            )

        self.optimization_step += 1


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


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for a scalar continuous or discrete time variable."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb requires an even dimension, got {dim}")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,), (B, 1), scalar, or integer-like inputs.
        x = x.reshape(-1)

        half_dim = self.dim // 2
        exponent = math.log(10000.0) / max(half_dim - 1, 1)

        frequencies = torch.exp(
            -exponent
            * torch.arange(
                half_dim,
                device=x.device,
                dtype=x.dtype,
            )
        )

        phase = x[:, None] * frequencies[None, :]
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class VectorField(nn.Module):
    """Shared conditional field for FM velocity prediction and DP denoising."""

    def __init__(
        self,
        seq_len,
        dof,
        param_len,
        gripper_idx=None,
        *,
        time_embed_dim: int = 32,
        time_scale: float = 100.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dof = dof
        self.param_len = param_len
        self.time_scale = float(time_scale)

        if gripper_idx is None:
            gripper_idx = []
        if isinstance(gripper_idx, list):
            gripper_idx = torch.tensor(gripper_idx, dtype=torch.long)
        self.register_buffer("gripper_idx", gripper_idx, persistent=False)

        loss_mask = torch.ones(dof, dtype=torch.float32)
        self.register_buffer(
            "loss_mask",
            loss_mask.view(1, 1, dof),
            persistent=False,
        )

        # Shared time encoder:
        # FM receives continuous t in [0, 1].
        # DP receives normalized diffusion time (k + 1) / T_diff in (0, 1].
        # Multiplying by time_scale gives the sinusoidal encoder a useful range.
                
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        """
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        """

        self.param_embed_dim = 32
        self.param_embed = nn.Sequential(
            nn.Linear(param_len, self.param_embed_dim),
            nn.Mish(),
            nn.Linear(self.param_embed_dim, self.param_embed_dim),
        )
        """
        self.param_embed_dim = param_len
        """
        
        condition_dim = time_embed_dim + self.param_embed_dim

        self.unet = UNet1D(
            dof=dof,
            condition_dim=condition_dim,
        )

    def forward(self, x, t, env_params):
        # x:          (B, seq_len, dof)
        # t:          (B, 1), normalized to approximately [0, 1]
        # env_params: (B, param_len)

        dev, dtype = x.device, x.dtype

        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=dev, dtype=dtype)
        else:
            t = t.to(device=dev, dtype=dtype)

        if not torch.is_tensor(env_params):
            env_params = torch.as_tensor(
                env_params,
                device=dev,
                dtype=dtype,
            )
        else:
            env_params = env_params.to(device=dev, dtype=dtype)

        B, seq_len, dof = x.shape
        assert seq_len == self.seq_len and dof == self.dof, "Shape mismatch"

        # Ensure a batch-shaped scalar time.
        if t.ndim == 0:
            t = t.expand(B)
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        else:
            t = t.reshape(B)

        # Shared encoding for FM and DP.
        t_embed = self.time_embed(t * self.time_scale)
        # t_embed = self.time_embed(t.unsqueeze(-1))
        p_embed = self.param_embed(env_params)
        # p_embed = env_params

        # Do not multiply raw environment parameters by 100 here.
        # Use the same conditioning path for every method.
        condition = torch.cat([p_embed, t_embed], dim=-1)

        out = self.unet(
            x.permute(0, 2, 1).contiguous(),
            condition,
        )
        return out.permute(0, 2, 1).contiguous()
    

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
        normalization_stats: dict[str, np.ndarray] | None = None,
        use_ema: bool = False,
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
        self.normalization_stats = normalization_stats
        self.use_ema = bool(use_ema)
        self.ema = None

    def _init_ema(self):
        if self.use_ema and self.ema is None:
            self.ema = EMAModel(self.model)

    def _step_ema(self):
        if self.ema is not None:
            self.ema.step(self.model)

    def _eval_model(self):
        return self.ema.averaged_model if self.ema is not None else self.model

    def _copy_eval_model(self):
        return copy.deepcopy(self._eval_model()).eval()

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
        max_policy_steps: int = 20,
        executed_horizon: int | None = None,
        observation_horizon: int = 1,
        eval_base_seed: int = 123,
        recorded_control_freq: int | float | None = None,
        trajectory_control_freq: int | float | None = None,
    ):
        N = target_trajectories.shape[0]
        best_avg_reward = 0.0
        best_success_rate = 0.0
        best_model = copy.deepcopy(self.model)
        records = {}
        stop_count = 0
        best_validation_rollouts = None
        self.best_validation_rollouts = None

        do_validation = val_period > 0 and val_trials > 0
        if do_validation:
            from Robot_simulation.env_util import _generate_val_env, eval_model

        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials)

        try:
            self.model = self.model.to(self.device)
            self._init_ema()
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
                    self._step_ema()
                    loss_sum += float(loss.item())

                if self.scheduler is not None:
                    self.scheduler.step()

                if do_validation and epoch % val_period == 0:
                    eval_model_obj = self._eval_model()
                    eval_model_obj.eval()
                    success_rate, avg_reward, validation_rollouts = eval_model(
                        eval_model_obj,
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
                        base_seed=eval_base_seed,
                        max_policy_steps=max_policy_steps,
                        executed_horizon=executed_horizon,
                        observation_horizon=observation_horizon,
                        recorded_control_freq=recorded_control_freq,
                        trajectory_control_freq=trajectory_control_freq,
                        normalization_stats=self.normalization_stats,
                        return_rollouts=True,
                        action_representation=getattr(self, "action_representation", "joint_space"),
                    )
                    records[epoch] = {"success_rate": success_rate, "avg_reward": avg_reward, "loss": loss_sum}
                    if success_rate < best_success_rate:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
                        if early_stopping:
                            if stop_count == stop_criteria:
                                tqdm.write("Early stopping triggered.")
                                break
                            stop_count += 1
                    elif best_validation_rollouts is None or (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = self._copy_eval_model()
                        best_validation_rollouts = validation_rollouts
                        self.best_validation_rollouts = best_validation_rollouts
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
        except KeyboardInterrupt:
            tqdm.write("Training interrupted by user. Returning best model so far...")

        if not records:
            best_model = self._copy_eval_model()
        return best_model, self.model, records


