import copy
import math
import numpy as np
import torch
from tqdm import tqdm

from Robot_simulation.env_util import _generate_val_env, eval_model


def cosine_beta_schedule(T: int, s: float = 0.008, max_beta: float = 0.999):
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, max_beta).to(torch.float32)

class DiffusionSchedule:
    def __init__(self, T: int, device: torch.device, schedule: str = "cosine"):
        self.T = int(T)
        if schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, self.T, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas.to(device)                    # (T,)
        self.alphas = (1.0 - self.betas)                 # (T,)
        self.alphas_bar = torch.cumprod(self.alphas, 0)  # (T,)

    def _gather(self, v: torch.Tensor, k: torch.Tensor, x_shape):
        out = v.gather(0, k)  # (B,)
        return out.view(-1, *([1] * (len(x_shape) - 1)))


@torch.no_grad()
def run_diffusion(
    model,
    xT: torch.Tensor,       # (B, Tseq, dof) ~ N(0,I)
    c: torch.Tensor,        # (B, param_len)
    device: str = "cuda",
    *,
    T_diff: int = 100,
    schedule_type: str = "cosine",
    ddim_steps: int | None = None,   # if None: use all T_diff steps
    eta: float = 0.0,                # 0.0 = deterministic DDIM; >0 adds stochasticity
    pred_type: str = "x0",           # "x0" or "epsilon"
    clip_sample: bool = True,
    clip_sample_range: float = 3.0,
):
    """
    Diffusion sampler (DDIM by default).

    pred_type="x0"     : model(x_k, t, c) -> x0 directly.
                         x0 prediction is preferred when the time embedding is weak
                         (e.g. a small MLP) because xk's noise magnitude already
                         encodes the noise level, reducing reliance on t conditioning.
    pred_type="epsilon": model(x_k, t, c) -> eps  (original formulation).

    Returns:
        x0: (B, Tseq, dof)
    """
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    device = device if use_cuda else "cpu"
    x = xT.to(device)
    c = c.to(device)

    sched = DiffusionSchedule(T=T_diff, device=torch.device(device), schedule=schedule_type)

    # DDIM index set
    if ddim_steps is None or ddim_steps >= T_diff:
        ks = list(range(T_diff - 1, -1, -1))
    else:
        stride = (T_diff - 1) / float(ddim_steps - 1)
        ks = [int(round((ddim_steps - 1 - i) * stride)) for i in range(ddim_steps)]
        ks = sorted(set(ks), reverse=True)
        if ks[-1] != 0:
            ks.append(0)

    B = x.shape[0]

    for j, k_int in enumerate(ks):
        k = torch.full((B,), k_int, device=device, dtype=torch.long)
        t = ((k.float() + 1.0) / float(T_diff)).unsqueeze(-1)  # (B,1)

        abar_k = sched._gather(sched.alphas_bar, k, x.shape)
        sqrt_abar_k = torch.sqrt(abar_k)
        sqrt_omabar_k = torch.sqrt(1.0 - abar_k)

        out = model(x, t, c)  # (B,Tseq,dof)

        if pred_type == "x0":
            x0  = out
            eps = (x - sqrt_abar_k * x0) / (sqrt_omabar_k + 1e-8)
        else:  # epsilon
            eps = out
            x0  = (x - sqrt_omabar_k * eps) / (sqrt_abar_k + 1e-8)

        if clip_sample:
            x0 = torch.clamp(x0, -clip_sample_range, clip_sample_range)

        # if last step, return x0
        if j == len(ks) - 1:
            x = x0
            break

        # next (more denoised) index in our strided schedule
        k_next_int = ks[j + 1]
        k_next = torch.full((B,), k_next_int, device=device, dtype=torch.long)
        abar_next = sched._gather(sched.alphas_bar, k_next, x.shape)

        # DDIM update: x_next = sqrt(abar_next)*x0 + sqrt(1-abar_next-sigma^2)*eps + sigma*z
        if eta == 0.0:
            sigma = torch.zeros_like(abar_k)
        else:
            sigma = eta * torch.sqrt(
                (1.0 - abar_next) / (1.0 - abar_k + 1e-8) *
                (1.0 - abar_k / (abar_next + 1e-8))
            )

        c2 = torch.sqrt(torch.clamp(1.0 - abar_next - sigma**2, min=0.0))
        z = torch.randn_like(x) if eta > 0 else 0.0

        x = torch.sqrt(abar_next) * x0 + c2 * eps + sigma * z

    return x

class DiffusionPolicy:
    """State-conditioned diffusion-policy trainer matching the FM policy interface."""

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
        device: str = "cuda",
        T_diff: int = 100,
        schedule_type: str = "cosine",
        ddim_steps: int | None = None,
        eta: float = 0.0,
        pred_type: str = "x0",
        normalization_stats: dict[str, np.ndarray] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task_name = task_name
        self.horizon = horizon
        self.dof = dof
        self.condition_dim = condition_dim
        self.gripper_idx = gripper_idx
        self.device = device
        self.T_diff = T_diff
        self.schedule_type = schedule_type
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.pred_type = pred_type
        self.normalization_stats = normalization_stats

    @torch.no_grad()
    def run_diffusion(self, x, c):
        return run_diffusion(
            self.model,
            x,
            c,
            self.device,
            T_diff=self.T_diff,
            schedule_type=self.schedule_type,
            ddim_steps=self.ddim_steps,
            eta=self.eta,
            pred_type=self.pred_type,
        )

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
        """Train the diffusion baseline with the same data contract as FM trainers."""
        N = target_trajectories.shape[0]
        best_avg_reward = 0.0
        best_success_rate = 0.0
        best_model = copy.deepcopy(self.model)
        records = {}
        stop_count = 0
        best_validation_rollouts = None
        self.best_validation_rollouts = None

        do_validation = val_period > 0 and val_trials > 0
        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials)

        try:
            self.model = self.model.to(self.device)
            target_trajectories = target_trajectories.to(self.device)
            conditions = conditions.to(self.device)
            sched = DiffusionSchedule(
                T=self.T_diff,
                device=torch.device(self.device),
                schedule=self.schedule_type,
            )

            for epoch in tqdm(range(1, max_epochs + 1), desc="DiffusionPolicy Training", unit="epoch"):
                self.model.train()
                perm = torch.randperm(N, device=self.device)
                loss_sum = 0.0
                batch_count = 0

                for i in range(0, N, batch_size):
                    idx = perm[i:min(i + batch_size, N)]
                    x0_clean = target_trajectories[idx]
                    cond = conditions[idx, :]
                    bsz = x0_clean.shape[0]

                    x0r = x0_clean.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
                    cr = cond.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, self.condition_dim)
                    k = torch.randint(0, self.T_diff, (bsz * n_t,), device=self.device, dtype=torch.long)
                    eps = torch.randn_like(x0r)
                    abar_k = sched._gather(sched.alphas_bar, k, x0r.shape)
                    xk = torch.sqrt(abar_k) * x0r + torch.sqrt(1.0 - abar_k) * eps
                    t = ((k.float() + 1.0) / float(self.T_diff)).unsqueeze(-1)

                    pred = self.model(xk, t, cr)
                    target = x0r if self.pred_type == "x0" else eps
                    sq_err = (pred - target) ** 2
                    if hasattr(self.model, "loss_mask") and self.model.loss_mask is not None:
                        sq_err = sq_err * self.model.loss_mask
                    loss = sq_err.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_sum += float(loss.item())
                    batch_count += 1

                if self.scheduler is not None:
                    self.scheduler.step()

                avg_loss = loss_sum / max(1, batch_count)
                if do_validation and epoch % val_period == 0:
                    self.model.eval()
                    success_rate, avg_reward, validation_rollouts = eval_model(
                        model=self.model,
                        model_class=None,
                        task_name=self.task_name,
                        seq_len=self.horizon,
                        dof=self.dof,
                        param_len=self.condition_dim,
                        gripper_idx=self.gripper_idx,
                        val_params=val_params,
                        env_settings_all=env_settings_all,
                        device=self.device,
                        trials=val_trials,
                        base_seed=eval_base_seed,
                        max_policy_steps=max_policy_steps,
                        executed_horizon=executed_horizon,
                        T_diff=self.T_diff,
                        schedule_type=self.schedule_type,
                        ddim_steps=self.ddim_steps,
                        eta=self.eta,
                        pred_type=self.pred_type,
                        normalization_stats=self.normalization_stats,
                        recorded_control_freq=recorded_control_freq,
                        trajectory_control_freq=trajectory_control_freq,
                        sampler_type="diffusion",
                        return_rollouts=True,
                        action_representation=getattr(self, "action_representation", "joint_space"),
                    )
                    records[epoch] = {"success_rate": success_rate, "avg_reward": avg_reward, "loss": avg_loss}

                    if success_rate < best_success_rate:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={avg_loss:.3f}")
                        if early_stopping:
                            if stop_count == stop_criteria:
                                tqdm.write("Early stopping triggered.")
                                break
                            stop_count += 1
                    elif best_validation_rollouts is None or (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = copy.deepcopy(self.model)
                        best_validation_rollouts = validation_rollouts
                        self.best_validation_rollouts = best_validation_rollouts
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={avg_loss:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={avg_loss:.3f}")
        except KeyboardInterrupt:
            tqdm.write("Training interrupted by user. Returning best model so far...")

        if not records:
            best_model = copy.deepcopy(self.model)
        return best_model, self.model, records


def train_DP(
    model,
    optimizer,
    scheduler,
    task_name,
    target_trajectories,
    environment_parameters,
    seq_len,
    dof,
    param_len,
    gripper_idx,
    n_t,
    max_epochs,
    batch_size,
    device,
    val_period=5,
    early_stopping=True,
    stop_criteria=3,
    val_trials=25,
    max_policy_steps: int = 20,
    executed_horizon: int | None = None,
    eval_base_seed: int = 123,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None,
    pred_type: str = "x0",
    T_diff: int = 100,
    schedule_type: str = "cosine",
    ddim_steps: int | None = None,
    eta: float = 0.0,
):
    policy = DiffusionPolicy(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        task_name=task_name,
        horizon=seq_len,
        dof=dof,
        condition_dim=param_len,
        gripper_idx=gripper_idx,
        device=device,
        T_diff=T_diff,
        schedule_type=schedule_type,
        ddim_steps=ddim_steps,
        eta=eta,
        pred_type=pred_type,
    )
    return policy.train(
        target_trajectories=target_trajectories,
        conditions=environment_parameters,
        n_t=n_t,
        max_epochs=max_epochs,
        batch_size=batch_size,
        val_period=val_period,
        early_stopping=early_stopping,
        stop_criteria=stop_criteria,
        val_trials=val_trials,
        max_policy_steps=max_policy_steps,
        executed_horizon=executed_horizon,
        eval_base_seed=eval_base_seed,
        recorded_control_freq=recorded_control_freq,
        trajectory_control_freq=trajectory_control_freq,
    )
