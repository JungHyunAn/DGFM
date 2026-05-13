import os
import imageio
import numpy as np

from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from robosuite.utils.transform_utils import mat2quat, quat_multiply, quat_inverse
from Robot_simulation.heuristics_util import get_dynamic_state, step_towards

def generate_nut_trajectory(
    env,
    delta_z: float = 0.1,
    render: bool = False,
    video_folder: str = "Robot_simulation/videos",
    verbose: bool = False,
):
    """
    Heuristic trajectory generator for NutAssembly, return the **joint-angle trajectory**.

    Phases:
      • PHASE1‑1: 100‑step approach to pre-grasp pose
      • PHASE1‑2: 20‑step careful approach to nut handle
      • PHASE2: 10‑step grasping handle
      • PHASE3: 50‑step approach to peg end
      • PHASE4: 50‑step decend into peg
      • PHASE5: 10-step opening gripper

    Args:
        env: robosuite NutAssembly environment (already constructed).
        delta_x: how much the pegs are pulled toward the robot
        delta_z: how much the pegs should move up with the table
        render: If True, collect frames and save an MP4 to `video_folder`.
        video_folder: Output directory for the rendered video.
        verbose: Print extra info (e.g., video save path).

    Returns:
        q_traj : np.ndarray
            (T, dof_arm + dof_gripper) joint positions sampled every env.step().
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
    
    Notes:
        - environment_setting is recorded after some time to prevent the pegs from starting midair
    """

    # ---------- storage ----------
    q_traj = []
    dynamic_traj = []
    frames = []

    # ---------- reset & joint indices ----------
    env.reset()

    # randomize peg x and y coordinates & shift upward with the table (delta_z)
    """
    delta_x_range = [-0.015, 0.015]
    delta_y_range = [-0.015, 0.015]
    delta_x = np.random.uniform(delta_x_range[0], delta_x_range[1])
    delta_y = np.random.uniform(delta_y_range[0], delta_y_range[1])
    """

    # fixed peg position for now
    delta_x = -0.05
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

    # ---------- record robot initial state ----------
    init_qpos = env.sim.data.qpos[joint_idx].copy()

    # ---------- helper for recording ----------
    def record_q():
        q_traj.append(env.sim.data.qpos[joint_idx].copy())
        # TODO(nut): Replace empty placeholder with nut pose / peg-relative state.
        dynamic_traj.append(get_dynamic_state(env, "nut"))

    # ---------- begin recording ----------
    record_q()

    # ---------- get current & peg & nut positions/orientations ----------
    nut_pos = env.sim.data.site_xpos[nut_handle_id].copy()
    nut_pos[2] = getattr(env, "table_offset", np.zeros(3))[2] # since the pegs drop from midair

    peg_pos = env.sim.data.body_xpos[peg_id].copy()
    # print(peg_id, peg_pos)
    # print(env.peg2_body_id, env.sim.data.body_xpos[3])

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
                 gripper_val=-1,
                 render=render,
                 frames=frames,
                 camera_name="frontview")    
    # ---------- record environment setting (to record after nuts drop) ----------
    environment_setting = {
        "qpos":     env.sim.data.qpos.copy(),
        "qvel":     env.sim.data.qvel.copy(),
        "body_pos": env.sim.model.body_pos.copy(),
        "body_quat":env.sim.model.body_quat.copy(),
    }
    yaw = np.arctan2(R0[1,0], R0[0,0])
    # environment_parameters = (nut_pos[0], nut_pos[1], yaw, peg_pos[0], peg_pos[1]) # for peg variation
    environment_parameters = (nut_pos[0], nut_pos[1], yaw)

    # ---------- PHASE1‑2: 20‑step careful approach to nut handle ----------
    grasp_height = nut_pos + np.array([0.0, 0.0, 0.015], dtype=np.float32) # 15mm above handle
    step_towards(env, eef_id, adim, record_q,
                 target_pos=grasp_height,
                 target_quat=quat0,
                 steps=20,
                 gripper_val=-1,
                 render=render,
                 frames=frames,
                 camera_name="frontview")
    
    # ---------- PHASE2: 10‑step grasping handle ----------
    for _ in range(10):
        a = np.zeros(adim); a[6] = 1.0
        obs, _, _, _ = env.step(a)
        record_q()
        if render:
            img = env.sim.render(640, 480, camera_name="frontview")
            frames.append(np.flipud(img))

    # ---------- PHASE3: 50‑step approach to peg end ----------
    # Find closest alignment
    if angle > np.pi/2 and np.pi > angle:
        quat1 = quat_multiply(q_cur, np.array([0, 0, -1/np.sqrt(2), 1/np.sqrt(2)]))
        pre_insert = peg_pos + np.array([0.0, 0.05, 0.18], dtype=np.float32)
    elif angle < np.pi*3/2 and np.pi < angle:
        quat1 = quat_multiply(q_cur, np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]))
        pre_insert = peg_pos + np.array([0.0, -0.05, 0.18], dtype=np.float32)
    else:
        quat1 = q_cur
        pre_insert = peg_pos + np.array([-0.05, 0.0, 0.18], dtype=np.float32)
    step_towards(env, eef_id, adim, record_q,
                 target_pos=pre_insert,
                 target_quat=quat1,
                 steps=50,
                 gripper_val=1.0,
                 render=render,
                 frames=frames,
                 camera_name="frontview")

    # ---------- PHASE4: 50‑step decend into peg ----------
    insert_height = pre_insert.copy()
    insert_height[2] -= 0.15
    step_towards(env, eef_id, adim, record_q,
                 target_pos=insert_height,
                 target_quat=quat1,
                 steps=50,
                 gripper_val=1.0,
                 render=render,
                 frames=frames,
                 camera_name="frontview")
    
     # ---------- PHASE5: 10-step opening gripper ----------
    for _ in range(10):
        a = np.zeros(adim); a[6] = -1.0
        obs, _, _, _ = env.step(a)
        record_q()
        if render:
            img = env.sim.render(640, 480, camera_name="frontview")
            frames.append(np.flipud(img))

    success = env._check_success()

    # ---------- optionally save frontview videos ----------
    if render:
        os.makedirs(video_folder, exist_ok=True)
        path = os.path.join(video_folder, "nut_assembly_frontview.mp4")
        imageio.mimsave(path, frames, fps=env.control_freq)
        if verbose:
            print(f"Saved frontview video to {path}")

    return (
        np.stack(q_traj, axis=0),
        success,
        frames,
        init_qpos,
        environment_setting,
        environment_parameters,
        np.stack(dynamic_traj, axis=0),
    )


if __name__ == "__main__":
    # 1) build env with all three offscreen cameras
    cams = ["frontview", "birdview", "robot0_eye_in_hand"]
    ctrl = load_composite_controller_config(robot="Panda")

    env = NutAssembly(
        robots="Panda",
        single_object_mode=2,
        nut_type="square",
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=cams,
        camera_heights=[480]*3,
        camera_widths =[640]*3,
        camera_depths =[False]*3,
        render_camera=None,   # not used when offscreen
        control_freq=20,
    )
    delta_z = 0.1
    
    env.table_offset[2] += delta_z
    env.reset()    
    
    traj, success, _, init_qpos, env_state, env_param, dynamic_states = generate_nut_trajectory(
        env,
        delta_z,
        render=True,
        video_folder="Robot_simulation/videos",
        verbose=True
    )

    if success:
        print("Nut task success!")
    print("Trajectory length:", len(traj))
    print("Initial joint angles:", init_qpos)
    print("Environment parameters:", env_param)
    print("Dynamic state shape:", dynamic_states.shape)

    print("Env snapshot keys & shapes:")
    for k, v in env_state.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}")
