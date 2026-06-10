"""MMFP-style autoencoder + latent-space Flow Matching trainer."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from tqdm import tqdm

from Robot_simulation.models.VanillaFM_class import EMAModel, SinusoidalPosEmb


class TrajectoryAutoencoder(nn.Module):
    """MLP autoencoder for fixed-horizon trajectory windows."""

    def __init__(
        self,
        horizon: int,
        dof: int,
        latent_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.dof = int(dof)
        self.input_dim = self.horizon * self.dof
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        if hidden_dim is None:
            hidden_dim = max(128, min(2048, self.input_dim * 2))

        mid_dim = max(self.latent_dim * 2, hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, mid_dim),
            nn.Mish(),
            nn.Linear(mid_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, mid_dim),
            nn.Mish(),
            nn.Linear(mid_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x.view(z.shape[0], self.horizon, self.dof)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class LatentVectorField(nn.Module):
    """Fully connected conditional vector field for standardized latent states."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 32,
        time_scale: float = 100.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.dof = self.latent_dim
        self.condition_dim = int(condition_dim)
        self.param_len = self.condition_dim
        self.hidden_dim = int(hidden_dim)
        self.time_scale = float(time_scale)

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        self.condition_embed_dim = min(128, max(32, self.condition_dim * 2))
        self.condition_embed = nn.Sequential(
            nn.Linear(self.condition_dim, self.condition_embed_dim),
            nn.Mish(),
            nn.Linear(self.condition_embed_dim, self.condition_embed_dim),
        )

        in_dim = self.latent_dim + time_embed_dim + self.condition_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Mish(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, env_params: torch.Tensor) -> torch.Tensor:
        dev, dtype = z.device, z.dtype
        B = z.shape[0]

        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=dev, dtype=dtype)
        else:
            t = t.to(device=dev, dtype=dtype)
        if not torch.is_tensor(env_params):
            env_params = torch.as_tensor(env_params, device=dev, dtype=dtype)
        else:
            env_params = env_params.to(device=dev, dtype=dtype)

        if z.ndim != 3 or z.shape[1] != 1 or z.shape[2] != self.latent_dim:
            raise ValueError(f"Expected z shape (B, 1, {self.latent_dim}), got {tuple(z.shape)}")
        if t.ndim == 0:
            t = t.expand(B)
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        else:
            t = t.reshape(B)

        z_flat = z.squeeze(1)
        t_emb = self.time_embed(t * self.time_scale)
        c_emb = self.condition_embed(env_params)
        out = self.net(torch.cat([z_flat, t_emb, c_emb], dim=-1))
        return out.unsqueeze(1)


class LatentFlowPolicy(nn.Module):
    """Decoded policy wrapper around a latent vector-field model."""

    def __init__(
        self,
        autoencoder: TrajectoryAutoencoder,
        latent_model: nn.Module,
        *,
        horizon: int,
        dof: int,
        latent_dim: int,
        condition_dim: int,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.latent_model = latent_model
        self.horizon = int(horizon)
        self.dof = int(dof)
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.device = device

        if latent_mean is None:
            latent_mean = torch.zeros(1, self.latent_dim)
        if latent_std is None:
            latent_std = torch.ones(1, self.latent_dim)
        self.register_buffer("latent_mean", latent_mean.detach().clone().float())
        self.register_buffer("latent_std", latent_std.detach().clone().float().clamp_min(1e-4))

    def forward(self, z: torch.Tensor, t: torch.Tensor, env_params: torch.Tensor) -> torch.Tensor:
        return self.latent_model(z, t, env_params)

    @torch.no_grad()
    def run_flow(self, x, c, n_steps=100):
        """Sample in standardized latent space, then unstandardize and decode."""
        dt = 1.0 / n_steps
        z = torch.randn(
            x.shape[0],
            1,
            self.latent_dim,
            device=x.device,
            dtype=x.dtype,
        )
        t = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)

        for i in range(n_steps):
            t.fill_(i * dt)
            v = self.latent_model(z, t, c)
            z.add_(v, alpha=dt)

        latent_mean = self.latent_mean.to(device=z.device, dtype=z.dtype)
        latent_std = self.latent_std.to(device=z.device, dtype=z.dtype)
        z_raw = z.squeeze(1) * latent_std + latent_mean
        return self.autoencoder.decode(z_raw)


class LatentFM:
    """Two-stage trainer: trajectory autoencoding, then FM in standardized latent space."""

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
        normalization_stats: dict | None = None,
        use_ema: bool = False,
        compression_rate: float = 0.25,
        latent_dim: int | None = None,
        autoencoder: TrajectoryAutoencoder | None = None,
        ae_optimizer=None,
        ae_scheduler=None,
        ae_max_epochs: int | None = None,
        ae_latent_reg_weight: float = 1e-4,
        ae_smoothness_weight: float = 1e-3,
    ):
        del gripper_idx
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task_name = task_name
        self.horizon = int(horizon)
        self.dof = int(dof)
        self.condition_dim = int(condition_dim)
        self.time_sampling = time_sampling
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.device = device
        self.normalization_stats = normalization_stats
        self.use_ema = bool(use_ema)
        self.ema = None
        self.compression_rate = float(compression_rate)
        self.full_dim = self.horizon * self.dof
        self.latent_dim = int(latent_dim if latent_dim is not None else max(1, round(self.compression_rate * self.full_dim)))
        self.autoencoder = autoencoder or TrajectoryAutoencoder(self.horizon, self.dof, self.latent_dim)
        self.ae_optimizer = ae_optimizer
        self.ae_scheduler = ae_scheduler
        self.ae_max_epochs = ae_max_epochs
        self.ae_latent_reg_weight = float(ae_latent_reg_weight)
        self.ae_smoothness_weight = float(ae_smoothness_weight)
        self.best_validation_rollouts = None
        self.latent_mean = torch.zeros(1, self.latent_dim, device=self.device)
        self.latent_std = torch.ones(1, self.latent_dim, device=self.device)
        self.latent_stats_summary = {}

    def _init_ema(self):
        if self.use_ema and self.ema is None:
            self.ema = EMAModel(self.model)

    def _step_ema(self):
        if self.ema is not None:
            self.ema.step(self.model)

    def _eval_latent_model(self):
        return self.ema.averaged_model if self.ema is not None else self.model

    def _policy(self, latent_model=None):
        return LatentFlowPolicy(
            copy.deepcopy(self.autoencoder).eval(),
            copy.deepcopy(latent_model or self._eval_latent_model()).eval(),
            horizon=self.horizon,
            dof=self.dof,
            latent_dim=self.latent_dim,
            condition_dim=self.condition_dim,
            latent_mean=self.latent_mean.detach().cpu(),
            latent_std=self.latent_std.detach().cpu(),
            device=self.device,
        ).to(self.device)

    def sample_t(self, n: int) -> torch.Tensor:
        if self.time_sampling == "shifted":
            t = Beta(self.beta_a, self.beta_b).sample((n, 1)).to(self.device)
            return torch.ones_like(t) - t
        return torch.rand(n, device=self.device).unsqueeze(-1)

    def _train_autoencoder(self, target_trajectories, max_epochs: int, batch_size: int):
        self.autoencoder = self.autoencoder.to(self.device)
        if self.ae_optimizer is None:
            self.ae_optimizer = torch.optim.Adam(
                self.autoencoder.parameters(),
                lr=self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else 1e-4,
                weight_decay=self.optimizer.param_groups[0].get("weight_decay", 0.0) if self.optimizer is not None else 1e-6,
            )

        ae_epochs = self.ae_max_epochs if self.ae_max_epochs is not None else max_epochs
        N = target_trajectories.shape[0]
        metrics = {
            "reconstruction_rmse": math.nan,
            "latent_reg": math.nan,
            "smoothness_reg": math.nan,
            "total_loss": math.nan,
        }
        for epoch in tqdm(range(1, ae_epochs + 1), desc="LatentFM Autoencoder Training", unit="epoch"):
            self.autoencoder.train()
            perm = torch.randperm(N, device=self.device)
            totals = {name: 0.0 for name in metrics}
            batches = 0
            for i in range(0, N, batch_size):
                idx = perm[i:min(i + batch_size, N)]
                x = target_trajectories[idx]
                z = self.autoencoder.encode(x)
                recon = self.autoencoder.decode(z)

                recon_loss = F.mse_loss(recon, x)
                latent_reg = z.square().mean()
                perm2 = torch.randperm(z.shape[0], device=z.device)
                alpha = torch.empty(z.shape[0], 1, device=z.device).uniform_(-0.4, 1.4)
                z_aug = alpha * z + (1.0 - alpha) * z[perm2]
                decoded_aug = self.autoencoder.decode(z_aug)
                smoothness_reg = (decoded_aug[:, 1:] - decoded_aug[:, :-1]).square().mean()
                loss = (
                    recon_loss
                    + self.ae_latent_reg_weight * latent_reg
                    + self.ae_smoothness_weight * smoothness_reg
                )

                self.ae_optimizer.zero_grad()
                loss.backward()
                self.ae_optimizer.step()

                totals["reconstruction_rmse"] += float(torch.sqrt(recon_loss.detach() + 1e-8).item())
                totals["latent_reg"] += float(latent_reg.detach().item())
                totals["smoothness_reg"] += float(smoothness_reg.detach().item())
                totals["total_loss"] += float(loss.detach().item())
                batches += 1
            if self.ae_scheduler is not None:
                self.ae_scheduler.step()
            metrics = {name: value / max(1, batches) for name, value in totals.items()}
            if epoch == 1 or epoch == ae_epochs or epoch % max(1, ae_epochs // 10) == 0:
                tqdm.write(
                    "AE epoch "
                    f"{epoch}: reconstruction_rmse={metrics['reconstruction_rmse']:.6f}, "
                    f"latent_reg={metrics['latent_reg']:.6f}, "
                    f"smoothness_reg={metrics['smoothness_reg']:.6f}, "
                    f"total_loss={metrics['total_loss']:.6f}"
                )
        return metrics

    def _standardize_latents(self, target_trajectories):
        self.autoencoder.eval()
        with torch.no_grad():
            latent_raw = self.autoencoder.encode(target_trajectories)
        self.latent_mean = latent_raw.mean(dim=0, keepdim=True).detach()
        self.latent_std = latent_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4).detach()
        latent_targets = (latent_raw - self.latent_mean) / self.latent_std
        self.latent_stats_summary = {
            "latent_mean_norm": float(self.latent_mean.norm().item()),
            "latent_std_mean": float(self.latent_std.mean().item()),
            "latent_std_min": float(self.latent_std.min().item()),
            "latent_std_max": float(self.latent_std.max().item()),
        }
        print(
            "[LatentFM] "
            f"latent_mean_norm={self.latent_stats_summary['latent_mean_norm']:.6f}, "
            f"latent_std_mean={self.latent_stats_summary['latent_std_mean']:.6f}, "
            f"latent_std_min={self.latent_stats_summary['latent_std_min']:.6f}, "
            f"latent_std_max={self.latent_stats_summary['latent_std_max']:.6f}"
        )
        return latent_targets.unsqueeze(1).detach()

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
        target_trajectories = target_trajectories.to(self.device)
        conditions = conditions.to(self.device)
        ae_metrics = self._train_autoencoder(target_trajectories, max_epochs, batch_size)
        latent_targets = self._standardize_latents(target_trajectories)

        N = latent_targets.shape[0]
        best_avg_reward = -float("inf")
        best_success_rate = -float("inf")
        best_model = self._policy(self.model)
        records = {}
        stop_count = 0
        best_validation_rollouts = None
        do_validation = val_period > 0 and val_trials > 0

        if do_validation:
            from Robot_simulation.env_util import _generate_val_env, eval_model

        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials)

        try:
            self.model = self.model.to(self.device)
            self._init_ema()
            for epoch in tqdm(range(1, max_epochs + 1), desc="LatentFM Flow Training", unit="epoch"):
                self.model.train()
                perm_t = torch.randperm(N, device=self.device)
                loss_sum = 0.0
                for i in range(0, N, batch_size):
                    idx = perm_t[i:min(i + batch_size, N)]
                    z1 = latent_targets[idx]
                    z0 = torch.randn(len(idx), 1, self.latent_dim, device=self.device)
                    t = self.sample_t(len(idx) * n_t)
                    cond = conditions[idx, :]

                    z1r = z1.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, 1, self.latent_dim)
                    z0r = z0.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, 1, self.latent_dim)
                    t_col = t.view(-1, 1, 1)
                    zt = (1 - t_col) * z0r + t_col * z1r
                    cond_r = cond.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, self.condition_dim)
                    target_v = z1r - z0r

                    pred_v = self.model(zt, t, cond_r)
                    loss = F.mse_loss(pred_v, target_v)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self._step_ema()
                    loss_sum += float(loss.item())

                if self.scheduler is not None:
                    self.scheduler.step()

                if do_validation and epoch % val_period == 0:
                    eval_model_obj = self._policy()
                    eval_model_obj.eval()
                    success_rate, avg_reward, validation_rollouts = eval_model(
                        eval_model_obj,
                        None,
                        self.task_name,
                        self.horizon,
                        self.dof,
                        self.condition_dim,
                        None,
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
                    records[epoch] = {
                        "success_rate": success_rate,
                        "avg_reward": avg_reward,
                        "loss": loss_sum,
                        "ae_reconstruction_rmse": ae_metrics["reconstruction_rmse"],
                        "ae_latent_reg": ae_metrics["latent_reg"],
                        "ae_smoothness_reg": ae_metrics["smoothness_reg"],
                        "ae_total_loss": ae_metrics["total_loss"],
                        **self.latent_stats_summary,
                    }
                    improved = (
                        success_rate > best_success_rate
                        or (
                            success_rate == best_success_rate
                            and avg_reward > best_avg_reward
                        )
                    )
                    if improved:
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = self._policy()
                        best_validation_rollouts = validation_rollouts
                        self.best_validation_rollouts = best_validation_rollouts
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
                        if early_stopping:
                            if stop_count == stop_criteria:
                                tqdm.write("Early stopping triggered.")
                                break
                            stop_count += 1
        except KeyboardInterrupt:
            tqdm.write("Training interrupted by user. Returning best model so far...")

        if not records:
            best_model = self._policy()
            records[0] = {
                "success_rate": float("nan"),
                "avg_reward": float("nan"),
                "loss": None,
                "ae_reconstruction_rmse": ae_metrics["reconstruction_rmse"],
                "ae_latent_reg": ae_metrics["latent_reg"],
                "ae_smoothness_reg": ae_metrics["smoothness_reg"],
                "ae_total_loss": ae_metrics["total_loss"],
                **self.latent_stats_summary,
            }
        return best_model, self._policy(self.model), records
