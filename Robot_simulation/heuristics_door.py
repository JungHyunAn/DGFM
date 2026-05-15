import os
import imageio
import numpy as np

from robosuite.environments.manipulation.door import Door
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from robosuite.utils.transform_utils import mat2quat, quat_multiply
from robosuite.utils.placement_samplers import UniformRandomSampler

from Robot_simulation.heuristics_util import get_dynamic_state, step_towards


def generate_door_trajectory(
    env,
    open_steps: int = 50,
    render: bool = False,
    video_folder: str = "Robot_simulation/videos",
    verbose: bool = False,
):
    """
    Heuristic trajectory generator for Door, return the **joint-angle trajectory**.

    Trajectory Phases:
      • PHASE1‑1: 100‑step approach to a pre-grasp pose
      • PHASE1‑2: 50‑step careful approach to handle
      • PHASE2: 10-step grasping handle
      • PHASE3: (open_steps)-step turning handle
      • PHASE4: (open_steps)-step pulling door

    Args:
        env: robosuite Door environment (already constructed).
        open_steps: Number of steps for handle rotation and door pulling phases.
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
        - Actions are constructed for the default OSC controller
          (pos/ori deltas + gripper scalar). If you switch controllers,
          update the action packing accordingly.
        - `mat2quat`, `quat_*` utilities assume MuJoCo’s [w,x,y,z] quaternion order.
        - `frontview_frames` can be large; avoid `render=True` when parallelizing.
    """

    # ---------- storage ----------
    q_traj = []
    gripper_traj = []
    dynamic_traj = []
    frontview_frames = []

    # ---------- reset & indices ----------
    env.reset()
    robot = env.robots[0]
    # arm+gripper
    arm_names  = robot.robot_model.joints
    grip_names = robot.gripper["right"].joints
    all_names  = arm_names + grip_names
    joint_idx = [env.sim.model.get_joint_qpos_addr(name) for name in all_names]
    # end effector
    eef_id    = list(robot.eef_site_id.values())[0]
    handle_id = env.door_handle_site_id

    adim  = env.action_dim

    # ---------- record initial states ----------
    init_qpos = env.sim.data.qpos[joint_idx].copy()
    qpos_all   = env.sim.data.qpos.copy()
    qvel_all   = env.sim.data.qvel.copy()
    body_pos   = env.sim.model.body_pos.copy()
    body_quat  = env.sim.model.body_quat.copy()
    environment_setting = {
        "qpos":      qpos_all,
        "qvel":      qvel_all,
        "body_pos":  body_pos,
        "body_quat": body_quat,
    }

    handle_pos = env._handle_xpos.copy()
    R_handle   = env.sim.data.site_xmat[handle_id].reshape(3,3)
    yaw = np.arctan2(R_handle[1,0], R_handle[0,0])
    environment_parameters = (handle_pos[0], handle_pos[1], yaw)

    # ---------- helper for recording ----------
    gripper_pose = 1.0

    def record_q():
        full = env.sim.data.qpos.copy()
        q_traj.append(full[joint_idx].copy())
        gripper_traj.append(gripper_pose)
        dynamic_traj.append(get_dynamic_state(env, "door"))

    # ---------- begin recording ----------
    record_q()

    # ---------- compute pre-grasp pose (for PHASE1-1) ----------
    curr_pos   = env.sim.data.site_xpos[eef_id].copy()
    
    handle_off = np.random.uniform(0.03, 0.08)
    pre_grasp  = handle_pos + np.array([handle_off, 0.1, 0.0])
    
    q_handle   = mat2quat(R_handle)
    for axis, ang in [(R_handle[:,0], -np.pi/2), (R_handle[:,1], np.pi/2)]:
        axis = axis / np.linalg.norm(axis)
        q_rot = np.concatenate([axis * np.sin(ang/2), [np.cos(ang/2)]]).astype(np.float32)
        q_handle = quat_multiply(q_rot, q_handle)    

    # ---------- PHASE1‑1: 100‑step approach to pre-grasp pose ----------
    gripper_pose = 1.0
    step_towards(env=env,
                 eef_id=eef_id,
                 adim=adim,
                 record_q=record_q,
                 target_pos=pre_grasp,
                 target_quat=q_handle,
                 steps=100,
                 gripper_val=-1,
                 render=render,
                 frames=frontview_frames,
                 camera_name="frontview")
    # ---------- PHASE1‑2: 50-step careful approach to handle ----------
    gripper_pose = 1.0
    for _ in range(50):
        a = np.zeros(adim)
        a[0:3] = [0, -0.21, 0]
        env.step(a)
        # print(env._gripper_to_handle)
        record_q()
        if render:
            img = env.sim.render(640,480, camera_name="frontview")
            frontview_frames.append(np.flipud(img))

    # ---------- PHASE2: 10-step grasping handle ----------
    gripper_pose = 0.0
    for _ in range(10):
        a = np.zeros(adim); a[6] = 1.0
        env.step(a)
        record_q()
        if render:
            img = env.sim.render(640,480, camera_name="frontview")
            frontview_frames.append(np.flipud(img))

    # ---------- PHASE3: (open_steps)-step turning handle ----------
    gripper_pose = 0.0
    angle_total = np.pi/2
    drot_step   = angle_total / open_steps
    rot_axis    = R_handle[:,1] / np.linalg.norm(R_handle[:,1])
    handle_ctr  = handle_pos.copy()
    eef_pos     = curr_pos.copy()
    radius_vec  = eef_pos - handle_ctr
    thetas      = np.linspace(drot_step, angle_total, open_steps)

    for theta in thetas:
        v = radius_vec; k = rot_axis
        v_rot = (v*np.cos(theta)
               + np.cross(k,v)*np.sin(theta)
               + k*(np.dot(k,v)*(1-np.cos(theta))))
        target_pos = handle_ctr + v_rot
        dpos       = target_pos - eef_pos

        a = np.zeros(adim)
        a[0:3] = dpos * 100
        a[3:6] = -rot_axis * drot_step * 15
        a[6]   = 1.0

        env.step(a)
        # print(env._gripper_to_handle)
        record_q()
        eef_pos = target_pos.copy()
        if render:
            img = env.sim.render(640,480, camera_name="frontview")
            frontview_frames.append(np.flipud(img))

    # ---------- PHASE4: (open_steps)-step pulling door ----------
    gripper_pose = 0.0
    for _ in range(open_steps):
        a = np.zeros(adim); a[0] = -60/open_steps; a[6] = 1.0
        env.step(a)
        record_q()
        if render:
            img = env.sim.render(640,480, camera_name="frontview")
            frontview_frames.append(np.flipud(img))

    success = env._check_success()

    # ---------- optionally save frontview videos ----------
    if render:
        os.makedirs(video_folder, exist_ok=True)
        out = os.path.join(video_folder, "door_heuristic_frontview.mp4")
        imageio.mimsave(out, frontview_frames, fps=env.control_freq)
        if verbose:
            print(f"Saved frontview video to {out}")

    return (
        np.stack(q_traj, axis=0),
        success,
        frontview_frames,
        init_qpos,
        environment_setting,
        environment_parameters,
        np.stack(dynamic_traj, axis=0),
        np.asarray(gripper_traj, dtype=np.float32),
    )


if __name__ == "__main__":
    # 1) build env with all three offscreen cameras
    cams = ["frontview", "birdview", "robot0_eye_in_hand"]
    ctrl = load_composite_controller_config(robot="Panda")

    door_sampler = UniformRandomSampler(
        name="door_placer",
        mujoco_objects="door_placer",
        x_range=[-0.05, 0.05],         # push along table x
        y_range=[-0.3, -0.1],
        rotation=(-np.pi/2 - 0.25, -np.pi/2),
        rotation_axis="z",
        reference_pos=(-0.2, -0.35, 0.8),  # same as env.table_offset
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
    )

    env = Door(
        robots="Panda",
        controller_configs=ctrl,
        placement_initializer=door_sampler,
        use_latch=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        initialization_noise={'magnitude': 0.2, 'type': "uniform"},
        # specify which cameras to instantiate
        camera_names=cams,
        camera_heights=[480]*3,
        camera_widths =[640]*3,
        camera_depths =[False]*3,
        render_camera=None,   # not used when offscreen
        control_freq=20,
    )

    result = generate_door_trajectory(
        env,
        open_steps=50,
        render=True,
        video_folder="Robot_simulation/videos",
        verbose=True
    )
    traj, success, _, init_qpos, env_state, env_param, dynamic_states = result[:7]

    if success:
        print("Door task success!")
    print("Trajectory length:", len(traj))
    print("Initial joint angles:", init_qpos)
    print("Environment parameters:", env_param)
    print("Dynamic state shape:", dynamic_states.shape)
    print("Env snapshot keys & shapes:")
    for k, v in env_state.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}")
