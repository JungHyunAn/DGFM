"""Latent-space Flow Matching trainer."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from tqdm import tqdm

from Robot_simulation.models.VanillaFM_class import EMAModel, VectorField


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


class LatentFlowPolicy(nn.Module):
    """Decoded policy wrapper around a latent vector-field model."""

    def __init__(
        self,
        autoencoder: TrajectoryAutoencoder,
        latent_model: VectorField,
        *,
        horizon: int,
        dof: int,
        latent_dim: int,
        condition_dim: int,
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

    def forward(self, z: torch.Tensor, t: torch.Tensor, env_params: torch.Tensor) -> torch.Tensor:
        return self.latent_model(z, t, env_params)

    @torch.no_grad()
    def run_flow(self, x, c, n_steps=100):
        """Sample in latent space, then decode to trajectory space."""
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

        return self.autoencoder.decode(z.squeeze(1))


class LatentFM:
    """Two-stage trainer: trajectory autoencoding, then FM in latent space."""

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
        autoencoder: TrajectoryAutoencoder | None = None,
        ae_optimizer=None,
        ae_scheduler=None,
        ae_max_epochs: int | None = None,
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
        self.latent_dim = int(getattr(model, "dof", max(1, round(self.compression_rate * self.full_dim))))
        self.autoencoder = autoencoder or TrajectoryAutoencoder(
            self.horizon,
            self.dof,
            self.latent_dim,
        )
        self.ae_optimizer = ae_optimizer
        self.ae_scheduler = ae_scheduler
        self.ae_max_epochs = ae_max_epochs
        self.best_validation_rollouts = None

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
        last_loss = math.nan
        for epoch in tqdm(range(1, ae_epochs + 1), desc="LatentFM Autoencoder Training", unit="epoch"):
            self.autoencoder.train()
            perm = torch.randperm(N, device=self.device)
            loss_sum = 0.0
            batches = 0
            for i in range(0, N, batch_size):
                idx = perm[i:min(i + batch_size, N)]
                x = target_trajectories[idx]
                recon = self.autoencoder(x)
                loss = torch.sqrt(F.mse_loss(recon, x) + 1e-8)

                self.ae_optimizer.zero_grad()
                loss.backward()
                self.ae_optimizer.step()
                loss_sum += float(loss.item())
                batches += 1
            if self.ae_scheduler is not None:
                self.ae_scheduler.step()
            last_loss = loss_sum / max(1, batches)
            if epoch == 1 or epoch == ae_epochs or epoch % max(1, ae_epochs // 10) == 0:
                tqdm.write(f"AE epoch {epoch}: reconstruction_rmse={last_loss:.6f}")
        return last_loss

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
        eval_base_seed: int = 123,
        recorded_control_freq: int | float | None = None,
        trajectory_control_freq: int | float | None = None,
    ):
        target_trajectories = target_trajectories.to(self.device)
        conditions = conditions.to(self.device)
        ae_rmse = self._train_autoencoder(target_trajectories, max_epochs, batch_size)

        self.autoencoder.eval()
        with torch.no_grad():
            latent_targets = self.autoencoder.encode(target_trajectories).unsqueeze(1).detach()

        N = latent_targets.shape[0]
        best_avg_reward = 0.0
        best_success_rate = 0.0
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
                        VectorField,
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
                        "ae_reconstruction_rmse": ae_rmse,
                    }
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
                        best_model = self._policy()
                        best_validation_rollouts = validation_rollouts
                        self.best_validation_rollouts = best_validation_rollouts
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
        except KeyboardInterrupt:
            tqdm.write("Training interrupted by user. Returning best model so far...")

        if not records:
            best_model = self._policy()
            records[0] = {
                "success_rate": float("nan"),
                "avg_reward": float("nan"),
                "loss": None,
                "ae_reconstruction_rmse": ae_rmse,
            }
        return best_model, self._policy(self.model), records
