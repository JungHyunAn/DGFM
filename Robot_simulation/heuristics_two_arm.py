import os
import imageio
import numpy as np
from typing import List, Dict, Tuple

from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from robosuite.utils.transform_utils import mat2quat, quat_slerp, quat_multiply, quat_inverse


def generate_two_arm_trajectory(
    env,
    lift_steps: int = 100,
    render: bool = False,
    video_folder: str = "Robot_simulation/videos",
    verbose: bool = False,
) -> Tuple[np.ndarray, bool, List[np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """
    Heuristic trajectory generator for TwoArmLift, return the **joint-angle trajectory**.

    Trajectory Phases:
      • PHASE1‑1: 100‑step approach to a random offset around each handle
      • PHASE1‑2: 30-step descend to handle
      • PHASE2: 10-step grasping handle
      • PHASE3: (lift_steps)‑step lift (keep level)

    Args:
        env: robosuite TwoArmLift environment (already constructed).
        open_steps: Number of steps for lifting phase.
        render: If True, collect frames and save an MP4 to `video_folder`.
        video_folder: Output directory for the rendered video.
        verbose: Print extra info (e.g., video save path).

    Returns:
        q_traj : (T, dof*2+dof_gripper*2) np.ndarray
        success: bool
        frames : list[np.ndarray]
        init_qpos : np.ndarray
        env_setting : dict
    """

    # ---------- storage ----------
    q_traj, frames = [], []

    # ---------- reset & indices ----------
    env.reset()
    left, right = env.robots[0], env.robots[1]
    # arms+grippers
    left_arm   = left.robot_model.joints
    left_grip  = next(iter(left.gripper.values())).joints
    right_arm  = right.robot_model.joints
    right_grip = next(iter(right.gripper.values())).joints
    all_joints = left_arm + left_grip + right_arm + right_grip
    joint_idx  = [env.sim.model.get_joint_qpos_addr(n) for n in all_joints]
    # end effectors
    eefL = list(left.eef_site_id.values())[0]
    eefR = list(right.eef_site_id.values())[0]
    # handles
    handle_names = [n for n in env.sim.model.site_names if "handle" in n]
    handle_ids   = [env.sim.model.site_name2id(n) for n in handle_names]
    hidL, hidR = handle_ids[:2]

    adim       = env.action_dim

    # ---------- record initial states ----------
    init_qpos       = env.sim.data.qpos[joint_idx].copy()
    env_setting     = {
        "qpos":     env.sim.data.qpos.copy(),
        "qvel":     env.sim.data.qvel.copy(),
        "body_pos": env.sim.model.body_pos.copy(),
        "body_quat":env.sim.model.body_quat.copy(),
    }
    posL = env.sim.data.site_xpos[hidL].copy()
    posR = env.sim.data.site_xpos[hidR].copy()
    R0   = env.sim.data.site_xmat[hidL].reshape(3,3)
    yaw  = np.arctan2(R0[1,0], R0[0,0])
    environment_parameters = ((posL[0]+posR[0])/2, (posL[1]+posR[1])/2, yaw)

    # ---------- helper for recording ----------
    def record_q():
        q_traj.append(env.sim.data.qpos[joint_idx].copy())

    # ---------- begin recording ----------
    record_q()

    # ---------- compute pre-grasp pose (for PHASE1-1) ----------
    tgtL = posL + np.array([0,0,0.045]) # pre-grasp pose for left arm
    tgtR = posR + np.array([0,0,0.045]) # pre-grasp pose for right arm
    target_quat   = mat2quat(R0)

    for axis, ang in [(R0[:,0], np.pi), (R0[:, 2], -np.pi/2)]:
        axis = axis / np.linalg.norm(axis)
        q_rot = np.concatenate([axis * np.sin(ang/2), [np.cos(ang/2)]]).astype(np.float32)
        target_quat = quat_multiply(q_rot, target_quat)    

    start_pos_L  = env.sim.data.site_xpos[eefL].copy()
    start_pos_R  = env.sim.data.site_xpos[eefR].copy()
    start_quat_L = mat2quat(env.sim.data.site_xmat[eefL].reshape(3, 3))
    start_quat_R = mat2quat(env.sim.data.site_xmat[eefR].reshape(3, 3))

    # ---------- PHASE1‑1: 100‑step approach to pre-grasp pose ----------
    fracs = np.linspace(0.0, 1.0, 101)[1:]
    for f in fracs:
        p_des_L = (1 - f) * start_pos_L + f * tgtL
        p_des_R = (1 - f) * start_pos_R + f * tgtR
        q_des_L = quat_slerp(start_quat_L, target_quat, f)
        q_des_R = quat_slerp(start_quat_R, target_quat, f)

        a = np.zeros(adim)

        # position
        cur_pos_L = env.sim.data.site_xpos[eefL].copy()
        cur_pos_R = env.sim.data.site_xpos[eefR].copy()
        a[0:3] = (p_des_L - cur_pos_L) * 100.0
        a[7:10] = (p_des_R - cur_pos_R) * 100.0

        # orientation
        q_now_L = mat2quat(env.sim.data.site_xmat[eefL].reshape(3, 3))
        q_now_R = mat2quat(env.sim.data.site_xmat[eefR].reshape(3, 3))
        
        q_rel_L = quat_multiply(q_des_L, quat_inverse(q_now_L))
        q_rel_R = quat_multiply(q_des_R, quat_inverse(q_now_R))

        w_L     = q_rel_L[3]
        th_L    = 2 * np.arccos(np.clip(w_L, -1, 1))
        w_R     = q_rel_R[3]
        th_R    = 2 * np.arccos(np.clip(w_R, -1, 1))

        if abs(th_L) < 1e-6:
            axis_L = np.zeros(3)
        else:
            axis_L = q_rel_L[:3] / np.sin(th_L / 2)
        a[3:6] = axis_L * th_L

        if abs(th_R) < 1e-6:
            axis_R = np.zeros(3)
        else:
            axis_R = q_rel_R[:3] / np.sin(th_R / 2)
        a[10:13] = axis_R * th_R

        # grippers open
        a[6]  = -1
        a[13] = -1

        env.step(a)
        record_q()
        if render and frames is not None:
            img = env.sim.render(640, 480, camera_name="frontview")
            frames.append(np.flipud(img))
    # ---------- PHASE1‑2: 30-step descend to handle ----------
    for _ in range(30):
        a = np.zeros(adim)
        # small downward move
        a[2] = -0.205
        a[9] = -0.205
        # keep orientation & open gripper
        a[6]  = -1
        a[13] = -1
        env.step(a)
        record_q()
        if render:
            img = env.sim.render(640,480, camera_name="frontview")
            frames.append(np.flipud(img))

    # ---------- PHASE2: 10-step grasping handle ----------
    for _ in range(10):
        a = np.zeros(adim)
        # left gripper at index 6, right at 6+7=13
        a[6]  = 1.0
        a[13] = 1.0
        _, _, _, _ = env.step(a)
        record_q()
        if render:
            img = env.sim.render(640,480, camera_name="frontview")
            frames.append(np.flipud(img))

    # ---------- PHASE3: (lift_steps)-step lifting the pot ----------
    # average initial handle height
    hz0 = env.sim.data.site_xpos[handle_ids[0]][2]
    hz1 = env.sim.data.site_xpos[handle_ids[1]][2]
    base_z      = 0.5 * (hz0 + hz1)
    dz          = 0.2 / lift_steps  # lift by 20cm total

    # move x,y by cubic function
    ax, bx, cx, ay, by, cy = np.random.uniform(-0.001,0.001,6)

    # change in yaw
    max_yaw = np.deg2rad(30)
    freq    = np.random.uniform(1,3)
    phase   = np.random.uniform(0,2*np.pi)
    yaw_prev = 0.0

    for i in range(lift_steps):
        # get current eef positions
        p0 = env.sim.data.site_xpos[eefL].copy()
        p1 = env.sim.data.site_xpos[eefR].copy()

        # target: same orientation, random small xy jitter, rising z
        dx = ax*(float(i)/lift_steps)**3 + bx*(float(i)/lift_steps)**2 + cx*(float(i)/lift_steps)
        dy = ay*(float(i)/lift_steps)**3 + by*(float(i)/lift_steps)**2 + cy*(float(i)/lift_steps)
        t0 = np.array([p0[0]+dx, p0[1]+dy, base_z + dz*(i+1)])
        t1 = np.array([p1[0]+dx, p1[1]+dy, base_z + dz*(i+1)])

        # compute delta‐pos and pack into one action
        a = np.zeros(adim)
        a[0:3]   = (t0 - p0) * 100.0
        a[7:10]  = (t1 - p1) * 100.0

        # small yaw changes
        frac     = (i+1)/lift_steps
        yaw_cur  = max_yaw * np.sin(2*np.pi*freq*frac + phase)
        yaw_delta = yaw_cur - yaw_prev
        yaw_prev  = yaw_cur
        axis_L = env.sim.data.site_xmat[eefL].reshape(3,3)[:,2]
        axis_R = env.sim.data.site_xmat[eefR].reshape(3,3)[:,2]
        a[3:6]   = axis_L * yaw_delta
        a[10:13] = axis_R * yaw_delta

        # keep grippers closed
        a[6]  = 1.0
        a[13] = 1.0

        env.step(a)
        record_q()
        if render:
            img = env.sim.render(640,480, camera_name="frontview")
            frames.append(np.flipud(img))

    success = env._check_success()

    # ---------- optionally save frontview videos ----------
    if render:
        os.makedirs(video_folder, exist_ok=True)
        path = os.path.join(video_folder, "two_arm_lift_frontview.mp4")
        imageio.mimsave(path, frames, fps=env.control_freq)
        if verbose:
            print(f"Saved video to {path}")

    return np.stack(q_traj, axis=0), success, frames, init_qpos, env_setting, environment_parameters


if __name__ == "__main__":
    # 1) build env with all three offscreen cameras
    cams = ["frontview", "birdview", "robot0_eye_in_hand"]
    ctrl = load_composite_controller_config(robot="Panda")

    env = TwoArmLift(
        robots=["Panda", "Panda"],
        controller_configs=ctrl,
        env_configuration="parallel",
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=cams,
        camera_heights=[480]*3,
        camera_widths =[640]*3,
        camera_depths =[False]*3,
        render_camera=None,
        control_freq=20,
    )

    traj, success, _, init_qpos, env_state, env_param = generate_two_arm_trajectory(
        env,
        render=True,
        video_folder="Robot_simulation/videos",
        verbose=True,
    )

    if success:
        print("Two Arm task success!")
    print("Trajectory length:", len(traj))
    print("Initial joint angles:", init_qpos)
    print("Environment parameters:", env_param)

    print("Env snapshot keys & shapes:")
    for k, v in env_state.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}")
