import math
import numpy as np
import torch
import os
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import copy
from tqdm import tqdm
from Robot_simulation.env_util import make_env
from Robot_simulation.heuristics_util import render_trajectory, write_grid_video
from Robot_simulation.FM_util import _rollout_batch, _generate_val_env


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


def eval_model_DP(
    model,
    model_class,                 # kept for signature compatibility (not used)
    task_name: str,
    seq_len: int,
    dof: int,
    param_len: int,
    gripper_idx,                 # not used here (rollout uses qpos -> action)
    val_params,
    env_settings_all,
    device: str = "cuda",
    render_dir: str = " ",
    video_name: str = None,
    trials: int = 100,
    num_workers: int = 10,
    render_width: int = 0,
    render_num: int = 0,
    base_seed: int = 123,
    gpu_chunk_size: int | None = None,
    *,
    T_diff: int = 100,
    schedule_type: str = "cosine",
    ddim_steps: int | None = None,
    eta: float = 0.0,
    pred_type: str = "x0",
) -> Tuple[float, float]:
    """
    Same evaluation pipeline as eval_model, but trajectory generation uses diffusion.
    """

    np_rng = np.random.RandomState(base_seed)

    # ---- (2) One GPU forward (optionally chunked) to produce all q_low ----
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        device = "cpu"

    model.eval()
    q_low_all = np.empty((trials, seq_len, dof), dtype=np.float32)

    if gpu_chunk_size is None or gpu_chunk_size <= 0:
        xT = torch.from_numpy(np_rng.randn(trials, seq_len, dof).astype(np.float32)).to(device)
        c  = torch.from_numpy(val_params).to(device)

        with torch.inference_mode():
            q_low = run_diffusion(model, xT, c, device,
                                  T_diff=T_diff, schedule_type=schedule_type,
                                  ddim_steps=ddim_steps, eta=eta, pred_type=pred_type)
        q_low_all[:] = q_low.cpu().numpy()
        del q_low
        if use_cuda:
            torch.cuda.empty_cache()

    else:
        N = trials
        for s in range(0, N, gpu_chunk_size):
            e = min(N, s + gpu_chunk_size)
            bs = e - s

            xT = torch.from_numpy(np_rng.randn(bs, seq_len, dof).astype(np.float32)).to(device)
            c  = torch.from_numpy(val_params[s:e]).to(device)

            with torch.inference_mode():
                q_low = run_diffusion(model, xT, c, device,
                                      T_diff=T_diff, schedule_type=schedule_type,
                                      ddim_steps=ddim_steps, eta=eta, pred_type=pred_type)
            q_low_all[s:e] = q_low.cpu().numpy()
            del q_low
            if use_cuda:
                torch.cuda.empty_cache()

    # ---- (3) Roll out on CPU with multiple workers (unchanged) ----
    base, rem = divmod(trials, max(1, num_workers))
    splits: List[tuple[int, int]] = []
    off = 0
    for i in range(num_workers):
        n = base + (1 if i < rem else 0)
        if n > 0:
            splits.append((off, off + n))
            off += n

    total_success = 0
    total_reward  = 0.0
    success_info: List[dict] = []
    failure_info: List[dict] = []
    s_count = 0
    f_count = 0

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futs = []
        for (s, e) in splits:
            futs.append(ex.submit(
                _rollout_batch,
                task_name,
                q_low_all[s:e],
                env_settings_all[s:e],
                s == 0
            ))
        for fut in futs:
            succ, rew, info_s, info_f = fut.result()
            total_success += succ
            total_reward  += rew

            if s_count < render_num:
                take = min(render_num - s_count, len(info_s))
                success_info += info_s[:take]; s_count += take
            if f_count < (render_width * render_width - render_num):
                take = min(render_width * render_width - render_num - f_count, len(info_f))
                failure_info += info_f[:take]; f_count += take

    success_rate = total_success / trials
    mean_reward  = total_reward  / trials

    # ---- (optional) render sample grid (unchanged; reuses your helpers) ----
    episode_frames = []
    s_left = min(s_count, render_num)
    f_left = min(f_count, render_width * render_width - render_num)

    while s_left > 0:
        env_r = make_env(task_name,
                         has_offscreen_renderer=True,
                         use_camera_obs=False,
                         use_joint_control=True,
                         environment_setting=success_info[s_left - 1]["setting"],
                         training=True)
        frames = render_trajectory(env_r,
                                   task_name,
                                   success_info[s_left - 1]["traj"],
                                   success_info[s_left - 1]["traj"][0, :],
                                   camera_name="frontview",
                                   hold_init=True,
                                   set_init=False)
        episode_frames.append(frames)
        env_r.close()
        s_left -= 1

    while f_left > 0:
        env_r = make_env(task_name,
                         has_offscreen_renderer=True,
                         use_camera_obs=False,
                         use_joint_control=True,
                         environment_setting=failure_info[f_left - 1]["setting"],
                         training=True)
        frames = render_trajectory(env_r,
                                   task_name,
                                   failure_info[f_left - 1]["traj"],
                                   failure_info[f_left - 1]["traj"][0, :],
                                   camera_name="frontview",
                                   hold_init=True,
                                   set_init=False)
        episode_frames.append(frames)
        env_r.close()
        f_left -= 1

    if render_num:
        grid_path = os.path.join(render_dir, f"{task_name}_grid_{video_name}.mp4")
        write_grid_video(episode_frames, grid_path,
                         grid_shape=(render_width, render_width))

    return success_rate, mean_reward


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
    pred_type: str = "x0",
):
    """
    Train Diffusion Policy baseline. Same format as train_uniform_FM.

    pred_type="x0"     : model predicts clean trajectory directly (recommended).
                         Works better with a weak time embedding because xk's
                         noise magnitude already encodes the noise level.
    pred_type="epsilon": classic epsilon prediction.

    Returns:
        best_model, last_model, success_rate_recs
    """

    # --- diffusion hyperparams (kept internal to preserve I/O) ---
    T_diff = 100
    schedule_type = "cosine"
    # evaluation sampler choices (can be tuned for speed/accuracy)
    ddim_steps_eval = None   # set to e.g. 50 or 100 if you want faster eval
    eta_eval = 0.0           # deterministic DDIM

    N = target_trajectories.shape[0]
    best_avg_reward = 0.0
    best_success_rate = 0.0
    best_model = copy.deepcopy(model)
    success_rate_recs = {}
    stop_count = 0

    env_settings_all, val_params = _generate_val_env(task_name, val_trials)

    try:
        model = model.to(device)
        target_trajectories = target_trajectories.to(device)
        environment_parameters = environment_parameters.to(device)

        sched = DiffusionSchedule(T=T_diff, device=torch.device(device), schedule=schedule_type)

        for epoch in tqdm(range(1, max_epochs + 1), desc="DiffusionPolicy Training", unit="epoch"):
            model.train()
            perm = torch.randperm(N, device=device)
            loss_sum = 0.0

            for i in range(0, N, batch_size):
                idx = perm[i:min(i + batch_size, N)]
                x0_clean = target_trajectories[idx]          # (B, Tseq, dof)
                env_params = environment_parameters[idx, :]  # (B, P)

                B = x0_clean.shape[0]

                # replicate by n_t (same semantics as FM code)
                x0r = x0_clean.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, seq_len, dof)      # (B*n_t,T,D)
                cr  = env_params.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, param_len)           # (B*n_t,P)

                # sample diffusion step k uniformly and noise eps
                k = torch.randint(0, T_diff, (B * n_t,), device=device, dtype=torch.long)          # (B*n_t,)
                eps = torch.randn_like(x0r)                                                        # (B*n_t,T,D)

                # forward noising: x_k
                abar_k = sched._gather(sched.alphas_bar, k, x0r.shape)
                xk = torch.sqrt(abar_k) * x0r + torch.sqrt(1.0 - abar_k) * eps

                # reuse model's time conditioning; treat time as normalized step
                t = ((k.float() + 1.0) / float(T_diff)).unsqueeze(-1)  # (B*n_t,1)

                with torch.enable_grad():
                    pred = model(xk, t, cr)
                    target = x0r if pred_type == "x0" else eps
                    sq_err = (pred - target) ** 2
                    if hasattr(model, "loss_mask") and model.loss_mask is not None:
                        sq_err = sq_err * model.loss_mask

                    loss = sq_err.mean()

                optimizer.zero_grad()
                loss.backward()
                loss_sum += float(loss.item())
                optimizer.step()

            scheduler.step()

            if epoch % val_period == 0:
                success_rate, avg_reward = eval_model_DP(
                    model=model,
                    model_class=None,
                    task_name=task_name,
                    seq_len=seq_len,
                    dof=dof,
                    param_len=param_len,
                    gripper_idx=gripper_idx,
                    val_params=val_params,
                    env_settings_all=env_settings_all,
                    device=device,
                    trials=val_trials,
                    T_diff=T_diff,
                    schedule_type=schedule_type,
                    ddim_steps=ddim_steps_eval,
                    eta=eta_eval,
                    pred_type=pred_type,
                )
                success_rate_recs[epoch] = {"success_rate": success_rate, "avg_reward": avg_reward, "loss": loss_sum}

                if success_rate < best_success_rate:
                    tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
                    if early_stopping:
                        if stop_count == stop_criteria:
                            tqdm.write("Early stopping triggered.")
                            break
                        stop_count += 1
                else:
                    if (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = copy.deepcopy(model)
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")

    except KeyboardInterrupt:
        tqdm.write("Training interrupted by user. Returning best model so far...")

    return best_model, model, success_rate_recs
