import os
import imageio
import numpy as np
import logging
logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)  
robosuite_logger.propagate = False        
for h in list(robosuite_logger.handlers): 
    robosuite_logger.removeHandler(h)

from typing import List, Dict, Tuple

from robosuite.utils.transform_utils import mat2quat
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

from Robot_simulation.environments.heuristics_util import get_dynamic_state, step_towards


def generate_wipe_trajectory(
    env,
    render: bool = False,
    video_folder: str = "Robot_simulation/videos",
    verbose: bool = False,
) -> Tuple[np.ndarray, bool, List[np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """
    Heuristic trajectory generator for Wipe, return the **joint-angle trajectory**.

    Trajectory Phases:
      • PHASE1‑1: 40‑step approach to hover above center
      • PHASE1‑2: 20‑step decending to wiping height
      • PHASE2: start wiping until the dirt is gone

    Args:
        env: robosuite Door environment (already constructed).
        render: If True, collect frames and save an MP4 to `video_folder`.
        video_folder: Output directory for the rendered video.
        verbose: Print extra info (e.g., video save path).

    Returns:
        q_traj : np.ndarray
            (T, dof_arm) joint positions sampled every env.step().
        success : bool
            Result of `env._check_success()` at the end of the rollout.
        frontview_frames : list[np.ndarray]
            Captured RGB frames (empty if render=False).
        init_qpos : np.ndarray
            Initial robot joint configuration (subset indexed by `joint_idx`).
        environment_setting : dict[str, np.ndarray]
            Full simulator snapshot for deterministic replay:
            {
              "qpos":      env.sim.data.qpos.copy(),
              "qvel":      env.sim.data.qvel.copy(),
              "body_pos":  env.sim.model.body_pos.copy(),
              "body_quat": env.sim.model.body_quat.copy(),
            }
        environment_parameters : tuple
            Empty for wipe; current dirt center/radius is recorded dynamically.
        dynamic_traj : np.ndarray
            (T, 3) values of (center_x, center_y, max_radius) sampled every env.step().
    """

    # ---------- storage ----------
    q_traj: List[np.ndarray] = []
    dynamic_traj: List[np.ndarray] = []
    frames: List[np.ndarray] = []

    # ---------- reset & joint indices ----------
    env.reset()
    robot = env.robots[0]
    # arm only; wipe does not need a gripper state channel
    arm_names  = robot.robot_model.joints
    joint_idx  = [env.sim.model.get_joint_qpos_addr(n) for n in arm_names]
    # end effector
    eef_id  = list(robot.eef_site_id.values())[0]
    adim = env.action_dim

    # ---------- record initial states ----------
    init_qpos = env.sim.data.qpos[joint_idx].copy()
    env_setting = {
        "qpos":      env.sim.data.qpos.copy(),
        "qvel":      env.sim.data.qvel.copy(),
        "body_pos":  env.sim.model.body_pos.copy(),
        "body_quat": env.sim.model.body_quat.copy(),
    }

    # ---------- helper for recording ----------
    def record_q():
        q_traj.append(env.sim.data.qpos[joint_idx].copy())
        dynamic_traj.append(get_dynamic_state(env, "wipe"))

    # ---------- helper for step then check & record ----------
    def episode_done():
        return bool(getattr(env, "done", False) or getattr(env, "_done", False))

    def step_and_check(pos, quat, steps=1):
        if episode_done():
            return True
        try:
            step_towards(env, eef_id, adim, record_q,
                            target_pos=pos, target_quat=quat, steps=steps,
                            render=render, frames=frames)
        except ValueError as exc:
            if "terminated episode" in str(exc):
                return True
            raise
        return env._check_success() or episode_done()

    # ---------- begin recording ----------
    record_q()

    # ---------- EEF orientation to hold ----------
    R_eef0  = env.sim.data.site_xmat[eef_id].reshape(3, 3)
    q_eef0  = mat2quat(R_eef0)

    # ---------- choose initial dirt center ----------
    max_radius, center, _ = env._get_wipe_information()
    environment_parameters = ()
    
    # ---------- contact&approach height ----------
    table_z = getattr(env, "table_offset", np.array([0, 0, 0]))[2]
    wipe_z  = table_z - 0.01  # light contact
    center[2] = wipe_z

    hover_z = wipe_z + 0.10

    # ---------- parameters to tune for policy -----------
    tool_width      = 0.03
    radius_margin   = 0.30
    min_wipe_radius = tool_width
    max_wipe_steps  = 360
    jitter_xy       = 0.003
    rng             = np.random.default_rng()

    cx, cy, _ = center
    wipe_z = center[2]           # already set to contact height

    # ---------- PHASE1‑1: 40‑step approach to hover above center ----------
    hover = np.array([cx, cy, hover_z])
    step_and_check(hover, q_eef0, steps=40)
    # ---------- PHASE1‑2: 20‑step decending to wiping height ----------
    start_down = np.array([cx, cy, wipe_z])
    step_and_check(start_down, q_eef0, steps=20)

    # ---------- PHASE2: wipe near the current dirt center, then recenter ----------
    sweep_angle = rng.uniform(0.0, 2.0 * np.pi)
    for wipe_step in range(max_wipe_steps):
        if env._check_success() or episode_done():
            break

        max_radius, center, _ = env._get_wipe_information()
        center = np.asarray(center, dtype=np.float32).copy()
        cx, cy = float(center[0]), float(center[1])
        local_radius = max(min_wipe_radius, float(max_radius) * (1.0 + radius_margin))

        if wipe_step % 6 == 0:
            sweep_angle = rng.uniform(0.0, 2.0 * np.pi)
        direction = np.array([np.cos(sweep_angle), np.sin(sweep_angle)], dtype=np.float32)
        tangent = np.array([-direction[1], direction[0]], dtype=np.float32)

        phase = wipe_step % 4
        if phase == 0:
            offset = np.zeros(2, dtype=np.float32)
        elif phase == 1:
            offset = direction * local_radius
        elif phase == 2:
            offset = -direction * local_radius
        else:
            offset = tangent * local_radius * rng.choice([-0.7, 0.7])

        pt = np.array([cx + offset[0], cy + offset[1], wipe_z], dtype=np.float32)
        pt[:2] += rng.uniform(-jitter_xy, jitter_xy, size=2)
        if step_and_check(pt, q_eef0):
            break
        
    success = env._check_success()

    # ---------- optionally save frontview videos ----------
    if render:
        os.makedirs(video_folder, exist_ok=True)
        out = os.path.join(video_folder, "wipe_heuristic_frontview.mp4")
        imageio.mimsave(out, frames, fps=env.control_freq)
        if verbose:
            print(f"Saved frontview video to {out}")

    return (
        np.stack(q_traj, axis=0),
        success,
        frames,
        init_qpos,
        env_setting,
        environment_parameters,
        np.stack(dynamic_traj, axis=0),
    )


if __name__ == "__main__":
    """
    Quick smoke test for the wipe heuristic.
    1) Build a Wipe env (offscreen).
    2) Run `generate_wipe_trajectory`.
    3) Optionally dump a video and print a short summary.
    """

    cams = ["frontview", "birdview", "robot0_eye_in_hand"]
    ctrl = load_composite_controller_config(robot="Panda")

    # Random placement for the wipe object/table if desired; set None for deterministic
    env = Wipe(
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=cams,
        camera_heights=[480] * 3,
        camera_widths =[640] * 3,
        camera_depths =[False] * 3,
        render_camera=None,
        control_freq=20,
    )
    env.task_config['contact_threshold'] = 0.01
    env.contact_threshold = 0.01
    env.task_config["num_markers"] = 25
    env.num_markers = 25
    env.task_config["table_offset"] = [0.3, 0, 1.0]
    env.table_offset = [0.3, 0, 1.0]
    env.task_config["table_full_size"] = [0.4, 0.6, 0.05]
    env.table_full_size = [0.4, 0.6, 0.05]

    traj, success, frames, init_qpos, env_state, env_param, dynamic_states = generate_wipe_trajectory(
        env,
        render=True,
        video_folder="Robot_simulation/videos",
        verbose=True,
    )

    if success:
        print("Wipe task success!")
    print("Trajectory length:", len(traj))
    print("Initial joint angles:", init_qpos)
    print("Environment parameters:", env_param)
    print("Dynamic state shape:", dynamic_states.shape)
    
    print("Env snapshot keys & shapes:")
    for k, v in env_state.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}")

    # Optionally persist snapshot
    # np.savez("wipe_env_snapshot.npz", **env_state)
