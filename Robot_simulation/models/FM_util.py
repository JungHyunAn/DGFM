"""
Flow Matching (FM) training & evaluation functions on RoboSuite
===============================================================

This module implements the training / inference building blocks for learning
trajectory generators with Flow Matching on RoboSuite tasks. It includes:

What this module provides
-------------------------
- **VectorField**: a conditional velocity field (1D U-Net backbone with FiLM)
  mapping (x_t, t, c) → v_t.
- **Flow classes**: `VanillaFM`, `UniformFM`, and `ShiftedFM` train fixed-horizon
  state-conditioned policies.
- **Evaluation**: 'eval_model' runs many parallel rollouts in RoboSuite,
  reports success rate / average reward, and (optionally) renders a grid video.
- **Utilities**: evaluation helpers for integrating and rolling out learned policies.

Key design choices & assumptions
--------------------------------
1) **Shapes & conventions**
   - Trajectories are '(B, T, D)' = (batch, seq_len, dof).
   - Environment parameters are '(B, P)'.
   - Times 't ∈ [0, 1]'.
   - Gripper pose channels are learned like arm channels. For gripper tasks,
     datasets use compact normalized gripper poses: 0 closed, 1 open.

2) **Trainers**
   - **UniformFM**: sample t ~ Uniform[0,1], regress v_t (standard FM).
   - **ShiftedFM**: shift / warp t to emphasize early (or late) time steps via Beta distribution.
   - **DGFM** lives in `Robot_simulation.models.DGFM_class`.
   - Schedules: cosine with warmup; Adam optimizer; early stopping supported.

3) **Evaluation loop**
   - Before training: build env, extract env params c
   - For each trial: restore env, sample a trajectory by integrating the vector field, 
     upsample / smooth externally, replay, then compute success / reward. 
     Multiprocessing is used for speed.
   - Rendering (optional) uses the external utilities from
     'Robot_simulation.environments.heuristics_util' (not defined here).

Note
----
This module focuses on learning and evaluation. Environment construction,
state restoration, trajectory smoothing, and rendering are provided by
'Robot_simulation.environments.heuristics_util'.
For nut assembly task, grasping the nut is ensured by lifting the nut in '_align_handle_to_nut'
"""

import numpy as np
import os
import torch

from scipy.interpolate import CubicSpline
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List
import logging
logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)  
robosuite_logger.propagate = False        
for h in list(robosuite_logger.handlers): 
    robosuite_logger.removeHandler(h)
from robosuite.utils.transform_utils import mat2quat, quat_multiply, quat_inverse

from Robot_simulation.env_util import make_env
from Robot_simulation.models.VanillaFM_class import VanillaFM
from Robot_simulation.environments.heuristics_util import (
    _clip_policy_gripper_dims,
    _current_robot_q,
    _get_environment_params,
    _state_policy_success,
    _to_action_from_q,
    get_dynamic_state,
    render_trajectory,
    step_towards,
    write_grid_video,
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


def _upsample_policy_trajectory(
    q_low: np.ndarray,
    task_name: str,
    recorded_control_freq: int | float | None,
    trajectory_control_freq: int | float | None,
) -> np.ndarray:
    sample_step = get_trajectory_sample_step(recorded_control_freq, trajectory_control_freq)
    q_low = _clip_policy_gripper_dims(np.asarray(q_low, dtype=np.float32), task_name)
    if sample_step == 1:
        return q_low
    if q_low.shape[0] < 2:
        return np.repeat(q_low, sample_step, axis=0)

    t_low = np.arange(q_low.shape[0], dtype=np.float32) / float(trajectory_control_freq)
    target_len = (q_low.shape[0] - 1) * sample_step + 1
    t_high = np.linspace(t_low[0], t_low[-1], target_len, dtype=np.float32)
    spline = CubicSpline(t_low, q_low, axis=0, extrapolate=False)
    q_high = spline(t_high).astype(np.float32)
    return _clip_policy_gripper_dims(q_high, task_name)


def build_state_conditioned_windows(
    trajectories: np.ndarray,
    dynamic_states: np.ndarray | None,
    static_env_params: np.ndarray,
    horizon: int,
    stride: int = 1,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert full episodes into diffusion-policy-style FM training windows.

    Each training item maps
    ``[current_joint_angles, current_dynamic_state, static_environment_params]``
    to the next fixed-horizon joint trajectory.

    If ``trajectory_control_freq`` is lower than ``recorded_control_freq``, each
    episode is downsampled before windows are built. For example, 20 Hz recorded
    data with a 5 Hz trajectory rate keeps one sample every four recorded steps.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    sample_step = get_trajectory_sample_step(recorded_control_freq, trajectory_control_freq)

    xs, cs = [], []
    n_eps = len(trajectories)
    for ep in range(n_eps):
        q = np.asarray(trajectories[ep], dtype=np.float32)
        dyn = None if dynamic_states is None else dynamic_states[ep]
        if dyn is None:
            dyn = np.zeros((len(q), 0), dtype=np.float32)
        else:
            dyn = np.asarray(dyn, dtype=np.float32)
            if dyn.shape[0] != len(q):
                raise ValueError(f"dynamic_states[{ep}] length {dyn.shape[0]} != trajectory length {len(q)}")
        if sample_step > 1:
            q = q[::sample_step]
            dyn = dyn[::sample_step]
        env_c = np.asarray(static_env_params[ep], dtype=np.float32)
        for start in range(0, len(q) - horizon + 1, stride):
            xs.append(q[start:start + horizon])
            cs.append(np.concatenate([q[start], dyn[start], env_c], axis=0))

    if not xs:
        raise ValueError(f"No training windows produced; horizon={horizon} is longer than all trajectories.")

    return np.asarray(xs, dtype=np.float32), np.asarray(cs, dtype=np.float32)


def make_policy_condition(
    current_q: np.ndarray,
    current_dynamic_state: np.ndarray | None,
    static_env_params: np.ndarray,
) -> np.ndarray:
    if current_dynamic_state is None:
        current_dynamic_state = np.zeros((0,), dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(current_q, dtype=np.float32).reshape(-1),
            np.asarray(current_dynamic_state, dtype=np.float32).reshape(-1),
            np.asarray(static_env_params, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )


def _condition_from_env(env, task_name: str, static_c: np.ndarray, param_len: int) -> tuple[np.ndarray, np.ndarray]:
    q0 = _current_robot_q(env, task_name)
    dyn = get_dynamic_state(env, task_name)
    static_flat = np.asarray(static_c, dtype=np.float32).reshape(-1)
    dyn_dim = param_len - q0.shape[0] - static_flat.shape[0]
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
    cond = make_policy_condition(q0, dyn, static_flat)
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
    recorded_control_freq: int | float | None,
    trajectory_control_freq: int | float | None,
):
    env = None
    executed = []
    try:
        env = make_env(
            task_name,
            use_joint_control=True,
            environment_setting=setting,
            training=True,
        )
        steps = 0
        cond, _ = _condition_from_env(env, task_name, static_c, param_len)
        conn.send({"type": "cond", "idx": idx, "cond": cond})

        while True:
            msg = conn.recv()
            if msg.get("type") == "close":
                break
            if msg.get("type") != "act":
                raise ValueError(f"Unknown worker message: {msg}")

            q_low = _upsample_policy_trajectory(
                msg["q_low"],
                task_name,
                recorded_control_freq,
                trajectory_control_freq,
            )

            done = False
            for q in q_low:
                _, _, done, _ = env.step(_to_action_from_q(q, task_name))
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

            cond, _ = _condition_from_env(env, task_name, static_c, param_len)
            conn.send({"type": "cond", "idx": idx, "cond": cond})
    except Exception as exc:
        conn.send({"type": "error", "idx": idx, "error": repr(exc)})
    finally:
        if env is not None:
            env.close()
        conn.close()


def _align_handle_to_nut(env, 
                         delta_x: float = 0.05, delta_z: float = 0.1, 
                         use_vision: bool = False, camera_name: str = "frontview"):
    def record_q(): # helper for recording, empty
        return

    # randomize peg x and y coordinates & shift upward with the table (delta_z)
    """
    delta_x_range = [-0.015, 0.015]
    delta_y_range = [-0.015, 0.015]
    delta_x = np.random.uniform(delta_x_range[0], delta_x_range[1])
    delta_y = np.random.uniform(delta_y_range[0], delta_y_range[1])
    """

    # fixed peg position for now
    delta_x = -delta_x
    delta_y = 0
    
    env.sim.model.body_pos[env.peg1_body_id][0] += delta_x
    env.sim.data.body_xpos[env.peg1_body_id][0] += delta_x
    env.sim.model.body_pos[env.peg1_body_id][1] += delta_y
    env.sim.data.body_xpos[env.peg1_body_id][1] += delta_y
    env.sim.model.body_pos[env.peg1_body_id][2] += delta_z
    env.sim.data.body_xpos[env.peg1_body_id][2] += delta_z

    env.sim.model.body_pos[env.peg2_body_id][2] = 0
    env.sim.data.body_xpos[env.peg2_body_id][2] = 0
    
    robot = env.robots[0]
    # arm+gripper
    arm_joints  = robot.robot_model.joints
    grip_joints = robot.gripper["right"].joints
    all_joints  = arm_joints + grip_joints
    joint_idx   = [env.sim.model.get_joint_qpos_addr(n) for n in all_joints]
    # eef & square peg & nut handle
    eef_id      = list(robot.eef_site_id.values())[0]
    peg_id      = env.peg1_body_id
    nut_handle_id = env.object_site_ids[0]
    adim        = env.action_dim

    init_qpos = env.sim.data.qpos[joint_idx].copy()

    nut_pos = env.sim.data.site_xpos[nut_handle_id].copy()
    nut_pos[2] = getattr(env, "table_offset", np.zeros(3))[2] # since the pegs drop from midair

    peg_pos = env.sim.data.body_xpos[peg_id].copy()

    R0       = env.sim.data.site_xmat[nut_handle_id].reshape(3,3)
    quat0    = mat2quat(R0)
    for axis, ang in [(R0[:,0], np.pi), (R0[:, 2], -np.pi/2)]:
        axis = axis / np.linalg.norm(axis)
        q_rot = np.concatenate([axis * np.sin(ang/2), [np.cos(ang/2)]]).astype(np.float32)
        quat0 = quat_multiply(q_rot, quat0)
    # calculate minimum shift in orientation
    q_cur = mat2quat(env.sim.data.site_xmat[eef_id].reshape(3,3))
    q_rel = quat_multiply(quat_inverse(q_cur), quat0)
    angle = 2 * np.arccos(np.clip(q_rel[3], -1.0, 1.0))
    if angle > np.pi/2 and angle < np.pi*3/2:
        axis, ang = (R0[:, 2], np.pi)
        axis = axis / np.linalg.norm(axis)
        q_rot = np.concatenate([axis * np.sin(ang/2), [np.cos(ang/2)]]).astype(np.float32)
        quat0 = quat_multiply(q_rot, quat0)

    # ---------- PHASE1‑1: 100‑step approach to pre-grasp pose ----------
    pre_grasp = nut_pos + np.array([0.0, 0.0, 0.06], dtype=np.float32) # 6cm above the nut
    step_towards(env, eef_id, adim, record_q,
                 target_pos=pre_grasp,
                 target_quat=quat0,
                 steps=100,
                 gripper_val=-1)
    # ---------- PHASE1‑2: 20‑step careful approach to nut handle ----------
    grasp_height = nut_pos + np.array([0.0, 0.0, 0.015], dtype=np.float32) # 15mm above handle
    step_towards(env, eef_id, adim, record_q,
                 target_pos=grasp_height,
                 target_quat=quat0,
                 steps=20,
                 gripper_val=-1)
    # ---------- PHASE2: 10‑step grasping handle ----------
    for _ in range(10):
        a = np.zeros(adim); a[6] = 1.0
        obs, _, _, _ = env.step(a)

    # ---------- record environment setting (to record after nuts drop) ----------
    # environment_setting = save_mj_state(env) # save full mujoco settings
    environment_setting = {
        "qpos":     env.sim.data.qpos.copy(),
        "qvel":     env.sim.data.qvel.copy(),
        "body_pos": env.sim.model.body_pos.copy(),
        "body_quat":env.sim.model.body_quat.copy(),
        "act": env.sim.data.act.copy(),
        "ctrl": env.sim.data.ctrl.copy(),
        "mocap_pos": env.sim.data.mocap_pos.copy(),
        "mocap_quat": env.sim.data.mocap_quat.copy()
    }
    yaw = np.arctan2(R0[1,0], R0[0,0])
    # environment_parameters = (nut_pos[0], nut_pos[1], yaw, peg_pos[0], peg_pos[1]) # for peg variation
    environment_parameters = (nut_pos[0], nut_pos[1], yaw)
    if (use_vision):
        vision = env.sim.render(640, 480, camera_name=camera_name)
    else:
        vision = None

    # ---------- PHASE3: 20‑step lift to check grasp ----------
    check_grasp = False
    step_towards(env, eef_id, adim, record_q,
                 target_pos=pre_grasp,
                 target_quat=quat0,
                 steps=20,
                 gripper_val=1)
    if (env.sim.data.site_xpos[nut_handle_id][2] > nut_pos[2] + 0.02):
        check_grasp = True

    return environment_setting, environment_parameters, check_grasp, vision
def _generate_val_env(task_name, val_trials):
    env_params_list: List[np.ndarray] = []
    env_settings_all: List[dict] = []

    if (task_name == "nut"):
        for _ in range(val_trials):
            for i in range(100):
                env = make_env(task_name)
                env.reset()
                env_setting, env_param, check_grasp = _align_handle_to_nut(env)
                env.close()  

                if check_grasp:
                    env_settings_all.append(env_setting)
                    env_params_list.append(env_param)
                    break
                if i == 99:
                    print("Nut environment failed grasping!")
    else:
        for _ in range(val_trials):
            env = make_env(task_name, use_joint_control=True)
            env.reset()

            env_settings_all.append({
                "qpos":      env.sim.data.qpos.copy(),
                "qvel":      env.sim.data.qvel.copy(),
                "body_pos":  env.sim.model.body_pos.copy(),
                "body_quat": env.sim.model.body_quat.copy(),
            })
            env_params_list.append(_get_environment_params(env, task_name))
            
            env.close()
    val_params = np.asarray(env_params_list, dtype=np.float32)         # (N, Dc)

    return env_settings_all, val_params


def _rollout_batch(
    task_name: str,
    q_low_batch: np.ndarray,          # (m, T, D) on CPU
    env_settings: List[dict],         # length m
    print_true: bool,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None) -> Tuple[int, float, list, list]:
    """Worker: restore envs from settings, upsample to controller rate, roll out CPU-only."""
    successes, reward_sum = 0, 0.0
    success_info, fail_info = [], []

    m = len(env_settings)
    assert m == len(q_low_batch), "settings and q_low_batch length mismatch"

    for i in range(m):
        setting = env_settings[i]

        env = make_env(task_name,
                       use_joint_control=True,
                       environment_setting=setting,
                       training=True)

        q_low = _upsample_policy_trajectory(
            q_low_batch[i],
            task_name,
            recorded_control_freq,
            trajectory_control_freq,
        )
        
        for q in q_low:
            env.step(_to_action_from_q(q, task_name))

        if env._check_success():
            if task_name == "two_arm": # check grasp for two_arm
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
) -> VanillaFM:
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
            out[s:e] = _clip_policy_gripper_dims(q_low.float().cpu().numpy(), task_name)
            del q_low
            torch.cuda.empty_cache()
        else:
            with torch.inference_mode():
                q_low = flow.run_flow(x0, c, n_steps=flow_steps)
            out[s:e] = _clip_policy_gripper_dims(q_low.cpu().numpy(), task_name)
    return out


def _rollout_state_policy_synchronized(
    model,
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
    flow_steps: int,
    gpu_chunk_size: int | None = None,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None,
) -> Tuple[float, float, list, list]:
    """Synchronize state-conditioned environments and batch flow inference in the parent."""
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
                    param_len,
                    max_policy_steps,
                    recorded_control_freq,
                    trajectory_control_freq,
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
                    pending_conditions[idx] = msg["cond"]
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

            cond_batch = np.stack([pending_conditions.pop(idx) for idx in ready], axis=0)
            q_low_batch = _run_flow_batched(
                model,
                task_name,
                cond_batch,
                seq_len,
                dof,
                device,
                rng,
                flow_steps,
                gpu_chunk_size,
            )
            for local_i, idx in enumerate(ready):
                conns[idx].send({"type": "act", "q_low": q_low_batch[local_i]})

        for idx, conn in conns.items():
            conn.close()
        for idx, proc in procs.items():
            proc.join()
            if proc.exitcode not in (0, None):
                raise RuntimeError(f"State rollout worker {idx} exited with code {proc.exitcode}")

    return total_success / trials, total_reward / trials, success_info, failure_info


def eval_model(
    model,
    model_class,                 # kept for signature compatibility (not used here)
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
    q_low = None,
    base_mixture = False,
    max_policy_steps: int = 20,
    flow_steps: int = 100,
    recorded_control_freq: int | float | None = None,
    trajectory_control_freq: int | float | None = None) -> Tuple[float, float]:
    """ Evaluates a flow model on given environments.

    Multiprocessing is used by calling _rollout_batch to evaluate multiple environments in parallel.

    Args:
        model: Neural vector field model to evaluate.
        task_name: Name of task, within {"door", "two arm", "nut", "wipe (currently unimplemented)"}
        trials: number of evaluation environments
        val_params: (trials, ) environment parameters of the evaluation environments
        env_settings_all: (trials, ) list of dicts to restore evaluation environments
        render_num: number of successful trajectories to render
        render_width: length of width/height for the rendered grid video (render_width^2 - render_num failed trajectories rendered)
        gpu_chunk_size: In case of multiple neural inference

    Pipeline:
        1) Environments are formed outside the function
        2) Environment parameters are passed to run the flow in a chunk on GPU (if available)
        3) Workers restore envs, upsample, and roll out on CPU
        4) For rendering, trajectory is replayed and saved to "render_dir/{task_name}_grid_{video_name}.mp4"

    Returns:
        Success rate and mean reward

    Evaluation pipeline:
      1) Parent makes envs sequentially, saves settings/params, then closes.
      2) Parent runs ONE GPU forward (optionally chunked) to get all q_low.
      3) Workers restore envs, upsample, and roll out on CPU (no GPU in workers).
    """
    np_rng = np.random.RandomState(base_seed)


    state_conditioned = val_params.shape[1] != param_len
    if state_conditioned:
        if model is None:
            raise ValueError("State-conditioned evaluation requires a model.")
        model.eval()
        success_rate, mean_reward, success_all, failure_all = _rollout_state_policy_synchronized(
            model=model,
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
            flow_steps=flow_steps,
            gpu_chunk_size=gpu_chunk_size,
            recorded_control_freq=recorded_control_freq,
            trajectory_control_freq=trajectory_control_freq,
        )
        success_info = success_all[:render_num]
        fail_slots = max(0, render_width * render_width - render_num)
        failure_info = failure_all[:fail_slots]
        s_count = len(success_info)
        f_count = len(failure_info)

        episode_frames = []
        s_left = min(s_count, render_num)
        f_left = min(f_count, render_width * render_width - render_num)
        while s_left > 0:
            env_r = make_env(task_name, has_offscreen_renderer=True, use_camera_obs=False,
                             use_joint_control=True, environment_setting=success_info[s_left - 1]["setting"],
                             training=True)
            frames = render_trajectory(env_r, task_name, success_info[s_left - 1]["traj"],
                                       success_info[s_left - 1]["traj"][0, :],
                                       camera_name="frontview", hold_init=False, set_init=False)
            episode_frames.append(frames)
            env_r.close()
            s_left -= 1
        while f_left > 0:
            env_r = make_env(task_name, has_offscreen_renderer=True, use_camera_obs=False,
                             use_joint_control=True, environment_setting=failure_info[f_left - 1]["setting"],
                             training=True)
            frames = render_trajectory(env_r, task_name, failure_info[f_left - 1]["traj"],
                                       failure_info[f_left - 1]["traj"][0, :],
                                       camera_name="frontview", hold_init=False, set_init=False)
            episode_frames.append(frames)
            env_r.close()
            f_left -= 1
        if render_num and episode_frames:
            grid_path = os.path.join(render_dir, f"{task_name}_grid_{video_name}.mp4")
            write_grid_video(episode_frames, grid_path, grid_shape=(render_width, render_width))
        return success_rate, mean_reward

    # ---- (1) One GPU forward (optionally chunked) to produce all q_low ----
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        device = "cpu"

    # pre-allocate to collect all q_low on CPU
    q_low_all = np.empty((trials, seq_len, dof), dtype=np.float32)
    flow = None
    if model is not None:
        flow = _make_flow_runner(model, task_name, seq_len, dof, param_len, device)

    if q_low is None and model is not None:
        model.eval()
        if gpu_chunk_size is None or gpu_chunk_size <= 0:
            # single shot (ensure it fits!)
            x0 = torch.from_numpy(np_rng.randn(trials, seq_len, dof).astype(np.float32)).to(device)
            c  = torch.from_numpy(val_params).to(device)
            if use_cuda:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    q_low = flow.run_flow(x0, c)          # (N, T, D) on CUDA
                q_low_all[:] = q_low.float().cpu().numpy()
                del q_low; torch.cuda.empty_cache()
            else:
                with torch.inference_mode():
                    q_low = flow.run_flow(x0, c)          # CPU path
                q_low_all[:] = q_low.cpu().numpy()
        else:
            # chunked single pass to cap VRAM
            N = trials
            for s in range(0, N, gpu_chunk_size):
                e = min(N, s + gpu_chunk_size)
                bs = e - s
                x0 = torch.from_numpy(np_rng.randn(bs, seq_len, dof).astype(np.float32)).to(device)
                c  = torch.from_numpy(val_params[s:e]).to(device)
                if use_cuda:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        q_low = flow.run_flow(x0, c)
                    q_low_all[s:e] = q_low.float().cpu().numpy()
                    del q_low; torch.cuda.empty_cache()
                else:
                    with torch.inference_mode():
                        q_low = flow.run_flow(x0, c)
                    q_low_all[s:e] = q_low.cpu().numpy()
    elif q_low is not None:
        q_low_all = q_low.cpu().numpy()

        if base_mixture:
            model.eval()
            x0 = torch.from_numpy(q_low_all).to(device)
            c  = torch.from_numpy(val_params).to(device)

            if use_cuda:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    q_low = flow.run_flow(x0, c)          # (N, T, D) on CUDA
                q_low_all[:] = q_low.float().cpu().numpy()
                del q_low; torch.cuda.empty_cache()
            else:
                with torch.inference_mode():
                    q_low = flow.run_flow(x0, c)          # CPU path
                q_low_all[:] = q_low.cpu().numpy()


    # ---- (2) Roll out on CPU with multiple workers ----
    # even split of trials across workers
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
                q_low_all[s:e],                 # numpy slice (copies view to child)
                env_settings_all[s:e],
                s == 0,
                recorded_control_freq,
                trajectory_control_freq
            ))
        for fut in futs:
            succ, rew, info_s, info_f = fut.result()
            total_success += succ
            total_reward  += rew

            if s_count < render_num:
                take = min(render_num - s_count, len(info_s))
                success_info += info_s[:take]; s_count += take
            if f_count < (render_width*render_width - render_num):
                take = min(render_width*render_width - render_num - f_count, len(info_f))
                failure_info += info_f[:take]; f_count += take

    success_rate = total_success / trials
    mean_reward  = total_reward  / trials

    # ---- (optional) render sample grid ----
    episode_frames = []
    s_left = min(s_count, render_num)
    f_left = min(f_count, render_width*render_width - render_num)

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
                                   hold_init=False,
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
                                   hold_init=False,
                                   set_init=False)
        episode_frames.append(frames)
        env_r.close()
        f_left -= 1

    if render_num:
        grid_path = os.path.join(render_dir, f"{task_name}_grid_{video_name}.mp4")
        write_grid_video(episode_frames, grid_path,
                         grid_shape=(render_width, render_width))

    return success_rate, mean_reward
