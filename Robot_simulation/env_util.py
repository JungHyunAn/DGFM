"""Environment construction, restoration, and policy evaluation utilities."""

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import List, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from Robot_simulation.models.vision_encoder import encode_camera_history

from Robot_simulation.environments.heuristics_util import (
    _clip_policy_gripper_dims,
    _current_robot_q,
    _get_environment_params,
    _state_policy_success,
    _to_action_from_q,
    configure_nut_pegs,
    validate_action_representation,
    get_environment_state,
    make_env,
    render_trajectory,
    restore_environment,
    restore_mj_state,
    save_mj_state,
    write_grid_video,
    capture_camera_views,
    DEFAULT_VISION_CAMERAS,
    DEFAULT_VISION_HEIGHT,
    DEFAULT_VISION_WIDTH,
)


def get_trajectory_sample_step(
    recorded_control_freq: int | float | None,
    trajectory_control_freq: int | float | None,
) -> int:
    if recorded_control_freq is None:
        recorded_control_freq = trajectory_control_freq
    if trajectory_control_freq is None:
        trajectory_control_freq = recorded_control_freq
    if recorded_control_freq is None or trajectory_control_freq is None:
        return 1
    if recorded_control_freq <= 0:
        raise ValueError("recorded_control_freq must be positive")
    if trajectory_control_freq <= 0:
        raise ValueError("trajectory_control_freq must be positive")
    if trajectory_control_freq > recorded_control_freq:
        raise ValueError(
            "trajectory_control_freq cannot exceed recorded_control_freq when sampling recorded trajectories"
        )
    ratio = recorded_control_freq / trajectory_control_freq
    sample_step = int(round(ratio))
    if not np.isclose(ratio, sample_step):
        raise ValueError(
            "recorded_control_freq must be an integer multiple of trajectory_control_freq; "
            f"got {recorded_control_freq} and {trajectory_control_freq}"
        )
    return sample_step


def normalize_policy_data(q: np.ndarray, normalization_stats: dict[str, np.ndarray]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if "min" in normalization_stats:
        q_min = np.asarray(normalization_stats["min"], dtype=np.float32)
        q_range = np.asarray(normalization_stats["range"], dtype=np.float32)
        return np.clip((2.0 / q_range) * (q - q_min) - 1.0, -1.0, 1.0).astype(np.float32)
    mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
    std = np.asarray(normalization_stats["std"], dtype=np.float32)
    return (q - mean) / std


def denormalize_policy_data(q: np.ndarray, normalization_stats: dict[str, np.ndarray]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if "min" in normalization_stats:
        q_min = np.asarray(normalization_stats["min"], dtype=np.float32)
        q_range = np.asarray(normalization_stats["range"], dtype=np.float32)
        return ((q + 1.0) * 0.5 * q_range + q_min).astype(np.float32)
    mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
    std = np.asarray(normalization_stats["std"], dtype=np.float32)
    return q * std + mean


def _maybe_denormalize_policy_data(
    q: np.ndarray,
    normalization_stats: dict[str, np.ndarray] | None,
) -> np.ndarray:
    if normalization_stats is None:
        return np.asarray(q, dtype=np.float32)
    return denormalize_policy_data(q, normalization_stats)


def _upsample_policy_trajectory(
    q_low: np.ndarray,
    task_name: str,
    recorded_control_freq: int | float | None,
    trajectory_control_freq: int | float | None,
    action_representation: str = "joint_space",
) -> np.ndarray:
    sample_step = get_trajectory_sample_step(recorded_control_freq, trajectory_control_freq)
    action_representation = validate_action_representation(action_representation)
    q_low = np.asarray(q_low, dtype=np.float32)
    if action_representation == "joint_space":
        q_low = _clip_policy_gripper_dims(q_low, task_name)
    if sample_step == 1:
        return q_low
    if q_low.shape[0] < 2:
        return np.repeat(q_low, sample_step, axis=0)

    t_low = np.arange(q_low.shape[0], dtype=np.float32) / float(trajectory_control_freq)
    target_len = (q_low.shape[0] - 1) * sample_step + 1
    t_high = np.linspace(t_low[0], t_low[-1], target_len, dtype=np.float32)
    spline = CubicSpline(t_low, q_low, axis=0, extrapolate=False)
    q_high = spline(t_high).astype(np.float32)
    if action_representation == "joint_space":
        return _clip_policy_gripper_dims(q_high, task_name)
    return q_high


def build_state_conditioned_windows(
    trajectories: np.ndarray,
    dynamic_states: np.ndarray | None,
    static_env_params: np.ndarray,
    horizon: int,
    stride: int = 1,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None,
    observation_horizon: int = 1,
    vision_features: list[np.ndarray] | None = None,
    observation_type: str = "state",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert full episodes into state-conditioned fixed-horizon windows."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if observation_horizon <= 0:
        raise ValueError("observation_horizon must be positive")
    if observation_type not in ("state", "vision"):
        raise ValueError(f"Unsupported observation_type={observation_type!r}")
    if observation_type == "vision" and vision_features is None:
        raise ValueError("Vision windows require vision_features")
    use_oracle_environment_state = observation_type == "state"
    sample_step = get_trajectory_sample_step(recorded_control_freq, trajectory_control_freq)

    xs, cs = [], []
    n_eps = len(trajectories)
    for ep in range(n_eps):
        q = np.asarray(trajectories[ep], dtype=np.float32)
        dyn = None if not use_oracle_environment_state or dynamic_states is None else dynamic_states[ep]
        if dyn is None:
            dyn = np.zeros((len(q), 0), dtype=np.float32)
        else:
            dyn = np.asarray(dyn, dtype=np.float32)
            if dyn.shape[0] != len(q):
                raise ValueError(f"dynamic_states[{ep}] length {dyn.shape[0]} != trajectory length {len(q)}")
        env_c = (
            np.asarray(static_env_params[ep], dtype=np.float32)
            if use_oracle_environment_state
            else np.zeros((0,), dtype=np.float32)
        )
        vision = None if vision_features is None else np.asarray(vision_features[ep], dtype=np.float32)
        if vision is not None and vision.shape[0] != len(q):
            raise ValueError(f"vision_features[{ep}] length {vision.shape[0]} != trajectory length {len(q)}")
        max_start = len(q) - (horizon - 1) * sample_step
        for start in range(0, max_start, stride):
            xs.append(q[start:start + horizon * sample_step:sample_step])
            obs_parts = []
            vision_parts = []
            first_obs_idx = start - (observation_horizon - 1) * sample_step
            for obs_i in range(observation_horizon):
                obs_idx = max(0, first_obs_idx + obs_i * sample_step)
                obs_parts.append(q[obs_idx])
                if use_oracle_environment_state:
                    obs_parts.append(dyn[obs_idx])
                if vision is not None:
                    vision_parts.append(vision[obs_idx])
            if use_oracle_environment_state:
                obs_parts.append(env_c)
            obs_parts.extend(vision_parts)
            cs.append(np.concatenate(obs_parts, axis=0))

    if not xs:
        raise ValueError(f"No training windows produced; horizon={horizon} is longer than all trajectories.")

    return np.asarray(xs, dtype=np.float32), np.asarray(cs, dtype=np.float32)


def make_policy_condition(
    current_q: np.ndarray,
    current_dynamic_state: np.ndarray | None,
    static_env_params: np.ndarray,
    observation_horizon: int = 1,
    q_history: list[np.ndarray] | None = None,
    dynamic_history: list[np.ndarray] | None = None,
) -> np.ndarray:
    if observation_horizon <= 0:
        raise ValueError("observation_horizon must be positive")
    if current_dynamic_state is None:
        current_dynamic_state = np.zeros((0,), dtype=np.float32)
    q_current = np.asarray(current_q, dtype=np.float32).reshape(-1)
    dyn_current = np.asarray(current_dynamic_state, dtype=np.float32).reshape(-1)
    q_values = [np.asarray(q, dtype=np.float32).reshape(-1) for q in (q_history or [])]
    dyn_values = [np.asarray(d, dtype=np.float32).reshape(-1) for d in (dynamic_history or [])]
    if not q_values:
        q_values = [q_current]
    if not dyn_values:
        dyn_values = [dyn_current]
    q_values = q_values[-observation_horizon:]
    dyn_values = dyn_values[-observation_horizon:]
    while len(q_values) < observation_horizon:
        q_values.insert(0, q_values[0])
    while len(dyn_values) < observation_horizon:
        dyn_values.insert(0, dyn_values[0])

    condition_parts = []
    for q, dyn in zip(q_values, dyn_values):
        condition_parts.extend([q, dyn])
    condition_parts.append(np.asarray(static_env_params, dtype=np.float32).reshape(-1))
    return np.concatenate(condition_parts, axis=0)


def _condition_from_env(
    env,
    task_name: str,
    static_c: np.ndarray,
    param_len: int,
    normalization_stats: dict[str, np.ndarray] | None = None,
    action_representation: str = "joint_space",
    observation_horizon: int = 1,
    q_history: list[np.ndarray] | None = None,
    dynamic_history: list[np.ndarray] | None = None,
    include_environment_state: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    q0 = _current_robot_q(env, task_name, action_representation)
    dyn = (
        get_environment_state(env, task_name)
        if include_environment_state
        else np.zeros((0,), dtype=np.float32)
    )
    static_flat = (
        np.asarray(static_c, dtype=np.float32).reshape(-1)
        if include_environment_state
        else np.zeros((0,), dtype=np.float32)
    )
    obs_param_len = param_len - static_flat.shape[0]
    if obs_param_len < 0:
        raise ValueError(f"Model param_len {param_len} is shorter than static condition length")
    if obs_param_len % observation_horizon != 0:
        raise ValueError(
            f"Observation condition length {obs_param_len} is not divisible by "
            f"observation_horizon={observation_horizon}"
        )
    dyn_dim = obs_param_len // observation_horizon - q0.shape[0]
    if dyn_dim < 0:
        raise ValueError(f"Model param_len {param_len} is shorter than q + static condition length")
    if dyn_dim == 0:
        dyn = None
    else:
        dyn = np.asarray(dyn, dtype=np.float32).reshape(-1)
        if dyn.shape[0] < dyn_dim:
            dyn = np.pad(dyn, (0, dyn_dim - dyn.shape[0]))
        elif dyn.shape[0] > dyn_dim:
            dyn = dyn[:dyn_dim]
    if dynamic_history is not None:
        normalized_dynamic_history = []
        for hist_dyn in dynamic_history:
            hist_dyn = np.asarray(hist_dyn, dtype=np.float32).reshape(-1)
            if hist_dyn.shape[0] < dyn_dim:
                hist_dyn = np.pad(hist_dyn, (0, dyn_dim - hist_dyn.shape[0]))
            elif hist_dyn.shape[0] > dyn_dim:
                hist_dyn = hist_dyn[:dyn_dim]
            normalized_dynamic_history.append(hist_dyn)
        dynamic_history = normalized_dynamic_history
    cond = make_policy_condition(
        q0,
        dyn,
        static_flat,
        observation_horizon=observation_horizon,
        q_history=q_history,
        dynamic_history=dynamic_history,
    )
    if cond.shape[0] != param_len:
        raise ValueError(f"Condition length {cond.shape[0]} != model param_len {param_len}")
    return cond.astype(np.float32), q0


def _state_policy_env_worker(
    conn,
    idx: int,
    task_name: str,
    setting: dict,
    static_c: np.ndarray,
    seq_len: int,
    param_len: int,
    max_policy_steps: int,
    executed_horizon: int,
    observation_horizon: int,
    recorded_control_freq: int | float | None,
    trajectory_control_freq: int | float | None,
    normalization_stats: dict[str, np.ndarray] | None = None,
    action_representation: str = "joint_space",
    observation_type: str = "state",
    camera_names=DEFAULT_VISION_CAMERAS,
    image_height: int = DEFAULT_VISION_HEIGHT,
    image_width: int = DEFAULT_VISION_WIDTH,
):
    env = None
    executed = []
    try:
        env = make_env(
            task_name,
            use_joint_control=(action_representation == "joint_space"),
            environment_setting=setting,
            training=True,
            action_representation=action_representation,
            has_offscreen_renderer=(observation_type == "vision"),
            camera_names=camera_names,
            camera_heights=[image_height] * len(camera_names),
            camera_widths=[image_width] * len(camera_names),
        )
        steps = 0
        q0 = _current_robot_q(env, task_name, action_representation)
        include_environment_state = observation_type == "state"
        condition_static_c = static_c if include_environment_state else np.zeros((0,), dtype=np.float32)
        dyn0 = (
            get_environment_state(env, task_name)
            if include_environment_state
            else np.zeros((0,), dtype=np.float32)
        )
        q_history = [q0]
        dynamic_history = [dyn0]
        image_history = []
        if observation_type == "vision":
            image_history.append(capture_camera_views(env, camera_names, image_width, image_height))
        cond, _ = _condition_from_env(
            env,
            task_name,
            condition_static_c,
            param_len,
            normalization_stats,
            action_representation,
            observation_horizon,
            q_history,
            dynamic_history,
            include_environment_state=include_environment_state,
        )
        padded_images = None
        if image_history:
            padded_images = [image_history[0]] * (observation_horizon - len(image_history)) + image_history
        conn.send({"type": "cond", "idx": idx, "cond": cond, "images": padded_images})

        while True:
            msg = conn.recv()
            if msg.get("type") == "close":
                break
            if msg.get("type") != "act":
                raise ValueError(f"Unknown worker message: {msg}")

            planned_q = msg["q_low"][:executed_horizon]
            q_low = _upsample_policy_trajectory(
                planned_q,
                task_name,
                recorded_control_freq,
                trajectory_control_freq,
                action_representation,
            )

            done = False
            for q in q_low:
                _, _, done, _ = env.step(_to_action_from_q(q, task_name, action_representation, env))
                executed.append(q.copy())
                if done or env._check_success():
                    break

            steps += 1
            success = _state_policy_success(env, task_name)
            if success or done or steps >= max_policy_steps:
                conn.send({
                    "type": "result",
                    "idx": idx,
                    "success": bool(success),
                    "reward": float(env.reward()),
                    "traj": np.asarray(executed, dtype=np.float32),
                    "setting": setting,
                })
                break

            q_history.append(_current_robot_q(env, task_name, action_representation))
            dynamic_history.append(
                get_environment_state(env, task_name)
                if include_environment_state
                else np.zeros((0,), dtype=np.float32)
            )
            q_history = q_history[-observation_horizon:]
            dynamic_history = dynamic_history[-observation_horizon:]
            if observation_type == "vision":
                image_history.append(capture_camera_views(env, camera_names, image_width, image_height))
                image_history = image_history[-observation_horizon:]
            cond, _ = _condition_from_env(
                env,
                task_name,
                condition_static_c,
                param_len,
                normalization_stats,
                action_representation,
                observation_horizon,
                q_history,
                dynamic_history,
                include_environment_state=include_environment_state,
            )
            padded_images = None
            if image_history:
                padded_images = [image_history[0]] * (observation_horizon - len(image_history)) + image_history
            conn.send({"type": "cond", "idx": idx, "cond": cond, "images": padded_images})
    except Exception as exc:
        conn.send({"type": "error", "idx": idx, "error": repr(exc)})
    finally:
        if env is not None:
            env.close()
        conn.close()


def _generate_val_env(task_name, val_trials):
    env_params_list: List[np.ndarray] = []
    env_settings_all: List[dict] = []

    if task_name == "nut":
        for _ in range(val_trials):
            env = make_env(task_name, training=True)
            env.reset()
            configure_nut_pegs(env, delta_x=-0.05, delta_z=0.1)

            env_settings_all.append({
                "qpos": env.sim.data.qpos.copy(),
                "qvel": env.sim.data.qvel.copy(),
                "body_pos": env.sim.model.body_pos.copy(),
                "body_quat": env.sim.model.body_quat.copy(),
                "act": env.sim.data.act.copy(),
                "ctrl": env.sim.data.ctrl.copy(),
                "mocap_pos": env.sim.data.mocap_pos.copy(),
                "mocap_quat": env.sim.data.mocap_quat.copy(),
            })
            env_params_list.append(_get_environment_params(env, task_name))
            env.close()
    else:
        for _ in range(val_trials):
            env = make_env(task_name, use_joint_control=True, training=True)
            env.reset()

            env_settings_all.append({
                "qpos": env.sim.data.qpos.copy(),
                "qvel": env.sim.data.qvel.copy(),
                "body_pos": env.sim.model.body_pos.copy(),
                "body_quat": env.sim.model.body_quat.copy(),
            })
            env_params_list.append(_get_environment_params(env, task_name))
            env.close()

    val_params = np.asarray(env_params_list, dtype=np.float32)
    return env_settings_all, val_params


def _rollout_batch(
    task_name: str,
    q_low_batch: np.ndarray,
    env_settings: List[dict],
    print_true: bool,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None,
    action_representation: str = "joint_space",
) -> Tuple[int, float, list, list]:
    """Worker: restore envs from settings, upsample to controller rate, roll out CPU-only."""
    successes, reward_sum = 0, 0.0
    success_info, fail_info = [], []

    m = len(env_settings)
    assert m == len(q_low_batch), "settings and q_low_batch length mismatch"

    for i in range(m):
        setting = env_settings[i]
        env = make_env(
            task_name,
            use_joint_control=(action_representation == "joint_space"),
            environment_setting=setting,
            training=True,
            action_representation=action_representation,
        )

        q_low = _upsample_policy_trajectory(
            q_low_batch[i],
            task_name,
            recorded_control_freq,
            trajectory_control_freq,
            action_representation,
        )

        for q in q_low:
            env.step(_to_action_from_q(q, task_name, action_representation, env))

        if env._check_success():
            if task_name == "two_arm":
                z0 = env._handle0_xpos[2]
                z1 = env._handle1_xpos[2]

                if abs(z1 - z0) < 0.05:
                    successes += 1
                    success_info.append({"traj": q_low, "setting": setting})
                else:
                    fail_info.append({"traj": q_low, "setting": setting})
            else:
                successes += 1
                success_info.append({"traj": q_low, "setting": setting})
        else:
            fail_info.append({"traj": q_low, "setting": setting})

        reward_sum += env.reward()
        env.close()

    return successes, reward_sum, success_info, fail_info


def _make_flow_runner(
    model,
    task_name: str,
    seq_len: int,
    dof: int,
    condition_dim: int,
    device: str,
):
    from Robot_simulation.models.VanillaFM_class import VanillaFM

    if hasattr(model, "run_flow"):
        return model

    return VanillaFM(
        model,
        optimizer=None,
        scheduler=None,
        task_name=task_name,
        horizon=seq_len,
        dof=dof,
        condition_dim=condition_dim,
        device=device,
    )


def _run_flow_batched(
    model,
    task_name: str,
    cond_batch: np.ndarray,
    seq_len: int,
    dof: int,
    device: str,
    rng: np.random.RandomState,
    flow_steps: int,
    gpu_chunk_size: int | None = None,
    normalization_stats: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        device = "cpu"

    n = cond_batch.shape[0]
    out = np.empty((n, seq_len, dof), dtype=np.float32)
    chunk = n if gpu_chunk_size is None or gpu_chunk_size <= 0 else gpu_chunk_size

    if hasattr(model, "sample"):
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            with torch.inference_mode():
                q_low = model.sample(cond_batch[s:e].astype(np.float32))
            if isinstance(q_low, torch.Tensor):
                q_low = q_low.detach().cpu().numpy()
            q_low = _maybe_denormalize_policy_data(q_low, normalization_stats)
            out[s:e] = _clip_policy_gripper_dims(np.asarray(q_low, dtype=np.float32), task_name)
        return out

    flow = _make_flow_runner(model, task_name, seq_len, dof, cond_batch.shape[1], device)

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        bs = e - s
        x0 = torch.from_numpy(rng.randn(bs, seq_len, dof).astype(np.float32)).to(device)
        c = torch.from_numpy(cond_batch[s:e].astype(np.float32)).to(device)
        if use_cuda:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                q_low = flow.run_flow(x0, c, n_steps=flow_steps)
            q_np = _maybe_denormalize_policy_data(q_low.float().cpu().numpy(), normalization_stats)
            out[s:e] = _clip_policy_gripper_dims(q_np, task_name)
            del q_low
            torch.cuda.empty_cache()
        else:
            with torch.inference_mode():
                q_low = flow.run_flow(x0, c, n_steps=flow_steps)
            q_np = _maybe_denormalize_policy_data(q_low.cpu().numpy(), normalization_stats)
            out[s:e] = _clip_policy_gripper_dims(q_np, task_name)
    return out


@torch.no_grad()
def _run_diffusion_batched(
    model,
    task_name: str,
    cond_batch: np.ndarray,
    seq_len: int,
    dof: int,
    device: str,
    rng: np.random.RandomState,
    gpu_chunk_size: int | None = None,
    normalization_stats: dict[str, np.ndarray] | None = None,
    action_representation: str = "joint_space",
    *,
    T_diff: int = 100,
    schedule_type: str = "cosine",
    ddim_steps: int | None = None,
    eta: float = 0.0,
    pred_type: str = "x0",
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
) -> np.ndarray:
    from Robot_simulation.models.DP_class import run_diffusion

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        device = "cpu"

    n = cond_batch.shape[0]
    out = np.empty((n, seq_len, dof), dtype=np.float32)
    chunk = n if gpu_chunk_size is None or gpu_chunk_size <= 0 else gpu_chunk_size

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        bs = e - s
        xT = torch.from_numpy(rng.randn(bs, seq_len, dof).astype(np.float32)).to(device)
        c = torch.from_numpy(cond_batch[s:e].astype(np.float32)).to(device)
        q_low = run_diffusion(
            model,
            xT,
            c,
            device,
            T_diff=T_diff,
            schedule_type=schedule_type,
            ddim_steps=ddim_steps,
            eta=eta,
            pred_type=pred_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        q_np = _maybe_denormalize_policy_data(q_low.float().cpu().numpy(), normalization_stats)
        out[s:e] = _clip_policy_gripper_dims(q_np, task_name)
        del q_low
        if use_cuda:
            torch.cuda.empty_cache()
    return out


def _run_policy_batched(
    model,
    sampler_type: str,
    task_name: str,
    cond_batch: np.ndarray,
    seq_len: int,
    dof: int,
    device: str,
    rng: np.random.RandomState,
    flow_steps: int,
    gpu_chunk_size: int | None,
    normalization_stats: dict[str, np.ndarray] | None,
    action_representation: str = "joint_space",
    *,
    T_diff: int,
    schedule_type: str,
    ddim_steps: int | None,
    eta: float,
    pred_type: str,
    clip_sample: bool,
    clip_sample_range: float,
) -> np.ndarray:
    if sampler_type == "flow":
        return _run_flow_batched(
            model,
            task_name,
            cond_batch,
            seq_len,
            dof,
            device,
            rng,
            flow_steps,
            gpu_chunk_size,
            normalization_stats=normalization_stats,
        )
    if sampler_type == "diffusion":
        return _run_diffusion_batched(
            model,
            task_name,
            cond_batch,
            seq_len,
            dof,
            device,
            rng,
            gpu_chunk_size,
            normalization_stats=normalization_stats,
            action_representation=action_representation,
            T_diff=T_diff,
            schedule_type=schedule_type,
            ddim_steps=ddim_steps,
            eta=eta,
            pred_type=pred_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
    raise ValueError(f"Unknown sampler_type: {sampler_type}")


def _rollout_state_policy_synchronized(
    model,
    sampler_type: str,
    task_name: str,
    seq_len: int,
    dof: int,
    param_len: int,
    static_params: np.ndarray,
    env_settings: List[dict],
    device: str,
    trials: int,
    num_workers: int,
    base_seed: int,
    max_policy_steps: int,
    executed_horizon: int,
    observation_horizon: int,
    flow_steps: int,
    gpu_chunk_size: int | None = None,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None,
    normalization_stats: dict[str, np.ndarray] | None = None,
    action_representation: str = "joint_space",
    *,
    T_diff: int = 100,
    schedule_type: str = "cosine",
    ddim_steps: int | None = None,
    eta: float = 0.0,
    pred_type: str = "x0",
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
) -> Tuple[float, float, list, list]:
    """Synchronize state-conditioned environments and batch inference in the parent."""
    if executed_horizon <= 0:
        raise ValueError(f"executed_horizon must be positive, got {executed_horizon}")
    if executed_horizon > seq_len:
        raise ValueError(f"executed_horizon={executed_horizon} exceeds planned seq_len={seq_len}")
    if observation_horizon <= 0:
        raise ValueError(f"observation_horizon must be positive, got {observation_horizon}")

    observation_type = getattr(model, "observation_type", "state")
    state_param_len = int(getattr(model, "state_condition_dim", param_len))
    vision_encoder = getattr(model, "vision_encoder", None)
    camera_names = tuple(getattr(model, "camera_names", DEFAULT_VISION_CAMERAS))
    image_height = int(getattr(model, "vision_image_height", DEFAULT_VISION_HEIGHT))
    image_width = int(getattr(model, "vision_image_width", DEFAULT_VISION_WIDTH))
    if observation_type == "vision" and vision_encoder is None:
        raise ValueError("Vision-conditioned rollout is missing its vision_encoder")

    rng = np.random.RandomState(base_seed)
    total_success = 0
    total_reward = 0.0
    success_info: List[dict] = []
    failure_info: List[dict] = []
    ctx = get_context("spawn")

    for batch_start in range(0, trials, max(1, num_workers)):
        batch_end = min(trials, batch_start + max(1, num_workers))
        conns = {}
        procs = {}
        active = set()

        for idx in range(batch_start, batch_end):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_state_policy_env_worker,
                args=(
                    child_conn,
                    idx,
                    task_name,
                    env_settings[idx],
                    static_params[idx],
                    seq_len,
                    state_param_len,
                    max_policy_steps,
                    executed_horizon,
                    observation_horizon,
                    recorded_control_freq,
                    trajectory_control_freq,
                    normalization_stats,
                    action_representation,
                    observation_type,
                    camera_names,
                    image_height,
                    image_width,
                ),
            )
            proc.start()
            child_conn.close()
            conns[idx] = parent_conn
            procs[idx] = proc
            active.add(idx)

        pending_conditions = {}
        while active:
            for idx in list(active):
                if idx in pending_conditions:
                    continue
                msg = conns[idx].recv()
                mtype = msg.get("type")
                if mtype == "cond":
                    pending_conditions[idx] = msg
                elif mtype == "result":
                    active.remove(idx)
                    if msg["success"]:
                        total_success += 1
                        success_info.append({"traj": msg["traj"], "setting": msg["setting"]})
                    else:
                        failure_info.append({"traj": msg["traj"], "setting": msg["setting"]})
                    total_reward += msg["reward"]
                elif mtype == "error":
                    raise RuntimeError(f"State rollout worker {idx} failed: {msg['error']}")
                else:
                    raise RuntimeError(f"Unexpected state rollout message from {idx}: {msg}")

            ready = [idx for idx in sorted(active) if idx in pending_conditions]
            if not ready:
                continue

            ready_messages = [pending_conditions.pop(idx) for idx in ready]
            cond_batch = np.stack([message["cond"] for message in ready_messages], axis=0)
            if observation_type == "vision":
                vision_batch = np.stack([
                    encode_camera_history(message["images"], vision_encoder, device)
                    for message in ready_messages
                ], axis=0)
                cond_batch = np.concatenate([cond_batch, vision_batch], axis=1)
            if cond_batch.shape[1] != param_len:
                raise ValueError(f"Live condition length {cond_batch.shape[1]} != model param_len {param_len}")
            q_low_batch = _run_policy_batched(
                model,
                sampler_type,
                task_name,
                cond_batch,
                seq_len,
                dof,
                device,
                rng,
                flow_steps,
                gpu_chunk_size,
                normalization_stats,
                action_representation,
                T_diff=T_diff,
                schedule_type=schedule_type,
                ddim_steps=ddim_steps,
                eta=eta,
                pred_type=pred_type,
                clip_sample=clip_sample,
                clip_sample_range=clip_sample_range,
            )
            for local_i, idx in enumerate(ready):
                conns[idx].send({"type": "act", "q_low": q_low_batch[local_i]})

        for conn in conns.values():
            conn.close()
        for idx, proc in procs.items():
            proc.join()
            if proc.exitcode not in (0, None):
                raise RuntimeError(f"State rollout worker {idx} exited with code {proc.exitcode}")

    return total_success / trials, total_reward / trials, success_info, failure_info


def _render_rollout_grid(
    task_name: str,
    render_dir: str,
    video_name: str | None,
    render_width: int,
    render_num: int,
    success_info: list,
    failure_info: list,
    action_representation: str = "joint_space",
):
    episode_frames = []
    total_slots = render_width * render_width
    if total_slots <= 0:
        return

    success_slots = min(render_num, total_slots)
    failure_slots = max(0, total_slots - success_slots)

    def take_without_repeat(infos: list, count: int) -> list:
        if count <= 0 or not infos:
            return []
        return infos[:count]

    success_grid = take_without_repeat(success_info, success_slots)
    remaining_success_slots = success_slots - len(success_grid)
    failure_overflow = take_without_repeat(failure_info, remaining_success_slots)
    success_grid += failure_overflow

    used_failures = len(failure_overflow)
    failure_grid = take_without_repeat(failure_info[used_failures:], failure_slots)
    remaining_failure_slots = failure_slots - len(failure_grid)
    success_grid += take_without_repeat(success_info[len(success_grid):], remaining_failure_slots)

    for info in success_grid + failure_grid:
        env_r = make_env(
            task_name,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            use_joint_control=(action_representation == "joint_space"),
            environment_setting=info["setting"],
            training=True,
            action_representation=action_representation,
        )
        frames = render_trajectory(
            env_r,
            task_name,
            info["traj"],
            info["traj"][0, :],
            camera_name="frontview",
            hold_init=False,
            set_init=False,
            action_representation=action_representation,
        )
        episode_frames.append(frames)
        env_r.close()

    if render_num and episode_frames:
        grid_path = os.path.join(render_dir, f"{task_name}_grid_{video_name}.mp4")
        write_grid_video(episode_frames, grid_path, grid_shape=(render_width, render_width))


def eval_model(
    model,
    model_class,
    task_name: str,
    seq_len: int,
    dof: int,
    param_len: int,
    gripper_idx,
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
    q_low=None,
    base_mixture=False,
    max_policy_steps: int = 20,
    executed_horizon: int | None = None,
    observation_horizon: int = 1,
    flow_steps: int = 100,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None,
    normalization_stats: dict[str, np.ndarray] | None = None,
    sampler_type: str = "flow",
    *,
    T_diff: int = 100,
    schedule_type: str = "cosine",
    ddim_steps: int | None = None,
    eta: float = 0.0,
    pred_type: str = "x0",
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
    return_rollouts: bool = False,
    action_representation: str = "joint_space",
) -> Tuple[float, float]:
    """Evaluate a flow or diffusion policy on restored RoboSuite environments."""
    del model_class, gripper_idx

    action_representation = validate_action_representation(action_representation)
    np_rng = np.random.RandomState(base_seed)
    if executed_horizon is None:
        executed_horizon = seq_len

    val_params = np.asarray(val_params, dtype=np.float32)
    state_conditioned = val_params.shape[1] != param_len
    if state_conditioned:
        if model is None:
            raise ValueError("State-conditioned evaluation requires a model.")
        model.eval()
        success_rate, mean_reward, success_all, failure_all = _rollout_state_policy_synchronized(
            model=model,
            sampler_type=sampler_type,
            task_name=task_name,
            seq_len=seq_len,
            dof=dof,
            param_len=param_len,
            static_params=val_params,
            env_settings=env_settings_all,
            device=device,
            trials=trials,
            num_workers=num_workers,
            base_seed=base_seed,
            max_policy_steps=max_policy_steps,
            executed_horizon=executed_horizon,
            observation_horizon=observation_horizon,
            flow_steps=flow_steps,
            gpu_chunk_size=gpu_chunk_size,
            recorded_control_freq=recorded_control_freq,
            trajectory_control_freq=trajectory_control_freq,
            normalization_stats=normalization_stats,
            action_representation=action_representation,
            T_diff=T_diff,
            schedule_type=schedule_type,
            ddim_steps=ddim_steps,
            eta=eta,
            pred_type=pred_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        _render_rollout_grid(
            task_name,
            render_dir,
            video_name,
            render_width,
            render_num,
            success_all,
            failure_all,
            action_representation=action_representation,
        )
        if return_rollouts:
            return success_rate, mean_reward, {"success": success_all, "failure": failure_all}
        return success_rate, mean_reward

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        device = "cpu"

    q_low_all = np.empty((trials, seq_len, dof), dtype=np.float32)
    flow = None
    if sampler_type == "flow" and model is not None:
        flow = _make_flow_runner(model, task_name, seq_len, dof, param_len, device)

    if q_low is None and model is not None:
        model.eval()
        if sampler_type == "flow":
            if gpu_chunk_size is None or gpu_chunk_size <= 0:
                x0 = torch.from_numpy(np_rng.randn(trials, seq_len, dof).astype(np.float32)).to(device)
                c = torch.from_numpy(val_params).to(device)
                if use_cuda:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        q_low = flow.run_flow(x0, c)
                    q_low_all[:] = _maybe_denormalize_policy_data(q_low.float().cpu().numpy(), normalization_stats)
                    del q_low
                    torch.cuda.empty_cache()
                else:
                    with torch.inference_mode():
                        q_low = flow.run_flow(x0, c)
                    q_low_all[:] = _maybe_denormalize_policy_data(q_low.cpu().numpy(), normalization_stats)
            else:
                for s in range(0, trials, gpu_chunk_size):
                    e = min(trials, s + gpu_chunk_size)
                    bs = e - s
                    x0 = torch.from_numpy(np_rng.randn(bs, seq_len, dof).astype(np.float32)).to(device)
                    c = torch.from_numpy(val_params[s:e]).to(device)
                    if use_cuda:
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                            q_low = flow.run_flow(x0, c)
                        q_low_all[s:e] = _maybe_denormalize_policy_data(q_low.float().cpu().numpy(), normalization_stats)
                        del q_low
                        torch.cuda.empty_cache()
                    else:
                        with torch.inference_mode():
                            q_low = flow.run_flow(x0, c)
                        q_low_all[s:e] = _maybe_denormalize_policy_data(q_low.cpu().numpy(), normalization_stats)
        elif sampler_type == "diffusion":
            q_low_all[:] = _run_diffusion_batched(
                model,
                task_name,
                val_params,
                seq_len,
                dof,
                device,
                np_rng,
                gpu_chunk_size,
                normalization_stats=normalization_stats,
                T_diff=T_diff,
                schedule_type=schedule_type,
                ddim_steps=ddim_steps,
                eta=eta,
                pred_type=pred_type,
                clip_sample=clip_sample,
                clip_sample_range=clip_sample_range,
            )
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
    elif q_low is not None:
        if isinstance(q_low, torch.Tensor):
            q_low_all = q_low.detach().cpu().numpy()
        else:
            q_low_all = np.asarray(q_low, dtype=np.float32)

        if base_mixture:
            if sampler_type != "flow":
                raise ValueError("base_mixture refinement is only supported with sampler_type='flow'")
            model.eval()
            x0 = torch.from_numpy(q_low_all).to(device)
            c = torch.from_numpy(val_params).to(device)

            if use_cuda:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    q_low = flow.run_flow(x0, c)
                q_low_all[:] = _maybe_denormalize_policy_data(q_low.float().cpu().numpy(), normalization_stats)
                del q_low
                torch.cuda.empty_cache()
            else:
                with torch.inference_mode():
                    q_low = flow.run_flow(x0, c)
                q_low_all[:] = _maybe_denormalize_policy_data(q_low.cpu().numpy(), normalization_stats)
        else:
            q_low_all = _maybe_denormalize_policy_data(q_low_all, normalization_stats)

    base, rem = divmod(trials, max(1, num_workers))
    splits: List[tuple[int, int]] = []
    off = 0
    for i in range(num_workers):
        n = base + (1 if i < rem else 0)
        if n > 0:
            splits.append((off, off + n))
            off += n

    total_success = 0
    total_reward = 0.0
    success_info: List[dict] = []
    failure_info: List[dict] = []
    s_count = 0
    f_count = 0
    rollout_collect_limit = trials if return_rollouts else (render_width * render_width if render_width > 0 else 0)

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futs = []
        for (s, e) in splits:
            futs.append(ex.submit(
                _rollout_batch,
                task_name,
                q_low_all[s:e],
                env_settings_all[s:e],
                s == 0,
                recorded_control_freq,
                trajectory_control_freq,
                action_representation,
            ))
        for fut in futs:
            succ, rew, info_s, info_f = fut.result()
            total_success += succ
            total_reward += rew

            if s_count < rollout_collect_limit:
                take = min(rollout_collect_limit - s_count, len(info_s))
                success_info += info_s[:take]
                s_count += take
            if f_count < rollout_collect_limit:
                take = min(rollout_collect_limit - f_count, len(info_f))
                failure_info += info_f[:take]
                f_count += take

    success_rate = total_success / trials
    mean_reward = total_reward / trials

    _render_rollout_grid(
        task_name,
        render_dir,
        video_name,
        render_width,
        render_num,
        success_info,
        failure_info,
        action_representation=action_representation,
    )

    if return_rollouts:
        return success_rate, mean_reward, {"success": success_info, "failure": failure_info}
    return success_rate, mean_reward


__all__ = [
    "build_state_conditioned_windows",
    "denormalize_policy_data",
    "eval_model",
    "get_trajectory_sample_step",
    "make_env",
    "make_policy_condition",
    "normalize_policy_data",
    "restore_environment",
    "restore_mj_state",
    "save_mj_state",
    "_generate_val_env",
    "_maybe_denormalize_policy_data",
    "_rollout_batch",
    "_state_policy_env_worker",
]
