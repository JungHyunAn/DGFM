"""
Robosuite dataset generation & rendering utilities
==================================================

This model implements helpers to generate env and trajectories. It includes:

What this module provides
-------------------------
- **make_env**: Build task-specific RoboSuite envs with optional absolute
  JOINT_POSITION control for arms (for evaluation).
- **restore_environment**: Deterministically restore a saved MuJoCo state using
  ('qpos', 'qvel', 'body_pos', 'body_quat', ...).
- **compute_smooth_trajectory**: Convert sparse keyframes to high-rate trajectories 
  via cubic splines; may include grasp motion insertion.
- **compute_smooth_trajectory_with_q0**: Before upsampling, inserts a prefix sequence
  to smoothly move to the first action.
- **render_trajectory**: Renders the trajectory.

Key design choices & assumptions
--------------------------------
1) **Deterministic restoration**
   If `environment_setting` is passed to `make_env`, we copy the full MuJoCo state
   and set `env.placement_initializer = None` so subsequent `forward()` preserves
   placements.

2) **Controller mode for replay**
   With `use_joint_control=True`, Panda arms run an absolute 7-DoF
   JOINT_POSITION controller (linear interpolation). Action packing used by
   '_to_action_from_q' mirrors task interfaces:
     • door / nut:  7 arm joints + 1 gripper scalar (avg of the two fingers)
     • wipe:        7 arm joints
     • two_arm:     ([7 arm] + [1 grip]) x 2
   If you change controllers or action conventions, update `_to_action_from_q`.

3) **Spline upsampling**
   For gripper tasks, 'compute_smooth_trajectory_gripper' inserts a short
   closure segment by duplicating a keyframe and sets gripper joints to
   open/close values on either side, then fits per-joint cubic splines. The
   wrapper resolves gripper joint indices from the live env. Wipe uses a minimal
   spline without gripper logic.

4) **Rendering path**
   'render_trajectory' builds per-frame actions via '_to_action_from_q', steps
   the env, and captures off-screen frames with 'env.sim.render'.

Task-specific notes
-------------------
- Wipe: table size / offsets and marker count are set for consistency.
- Nut: table is slightly lifted to avoid initial penetrations.
"""

import imageio
import numpy as np
import logging
logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)  
robosuite_logger.propagate = False        
for h in list(robosuite_logger.handlers): 
    robosuite_logger.removeHandler(h)

from typing import List, Optional, Callable
from scipy.interpolate import CubicSpline
import mujoco

from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import mat2quat, quat_inverse, quat_multiply, quat_slerp


def _safe_qpos_by_joint_substring(env, substrings):
    """Return qpos values for joints whose names contain any requested substring."""
    values = []
    names = getattr(env.sim.model, "joint_names", [])
    for name in names:
        lname = name.lower()
        if any(s in lname for s in substrings):
            try:
                addr = env.sim.model.get_joint_qpos_addr(name)
                values.append(np.asarray(env.sim.data.qpos[addr]).reshape(-1)[0])
            except Exception:
                continue
    return values


def get_dynamic_state(env, task_name: str) -> np.ndarray:
    """Extract task dynamic state recorded alongside robot joint angles.

    Door records handle/latch angle and door hinge angle. Wipe records a
    per-marker remaining-dirt mask. Other tasks intentionally return an empty
    vector until task-specific dynamics are chosen.
    """
    if task_name == "door":
        handle_angle = None
        door_angle = None

        for attr in ("handle_qpos_addr", "latch_qpos_addr"):
            if hasattr(env, attr):
                try:
                    handle_angle = float(np.asarray(env.sim.data.qpos[getattr(env, attr)]).reshape(-1)[0])
                    break
                except Exception:
                    pass
        for attr in ("hinge_qpos_addr", "door_qpos_addr"):
            if hasattr(env, attr):
                try:
                    door_angle = float(np.asarray(env.sim.data.qpos[getattr(env, attr)]).reshape(-1)[0])
                    break
                except Exception:
                    pass

        if handle_angle is None:
            vals = _safe_qpos_by_joint_substring(env, ("handle", "latch"))
            handle_angle = vals[0] if vals else 0.0
        if door_angle is None:
            vals = _safe_qpos_by_joint_substring(env, ("hinge", "door"))
            door_angle = vals[0] if vals else 0.0

        return np.array([handle_angle, door_angle], dtype=np.float32)

    # TODO(two_arm): Record pot/handle dynamic pose or lifted height during rollout.
    # TODO(nut): Record nut pose relative to peg during rollout.
    if task_name == "wipe":
        markers = getattr(getattr(env.model, "mujoco_arena", None), "markers", ())
        if not markers:
            return np.zeros((0,), dtype=np.float32)

        marker_geom_ids = getattr(env, "_dgfm_wipe_marker_geom_ids", None)
        if marker_geom_ids is None or len(marker_geom_ids) != len(markers):
            marker_geom_ids = np.fromiter(
                (env.sim.model.geom_name2id(marker.visual_geoms[0]) for marker in markers),
                dtype=np.int32,
                count=len(markers),
            )
            env._dgfm_wipe_marker_geom_ids = marker_geom_ids

        return (env.sim.model.geom_rgba[marker_geom_ids, 3] > 0.0).astype(np.float32)

    return np.zeros((0,), dtype=np.float32)


SIG = (mujoco.mjtState.mjSTATE_INTEGRATION)
def save_mj_state(env):
    m = env.sim.model._model
    d = env.sim.data._data
    n = mujoco.mj_stateSize(m, int(SIG))
    buf = np.empty(n, dtype=np.float64)           # mjtNum is float64 in Python bindings
    mujoco.mj_getState(m, d, buf, int(SIG))
    return buf                                    # numpy array you can store
def restore_mj_state(env, buf: np.ndarray):
    m = env.sim.model._model
    d = env.sim.data._data
    mujoco.mj_setState(m, d, np.asarray(buf, dtype=np.float64), int(SIG))
    # finalize caches / kinematics
    env.sim.forward()


def step_towards(
    env,
    eef_id: int,
    adim: int,
    record_q,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    steps: int = 40,
    pos_gain: float = 100.0,
    ori_gain: float = 1.0,
    gripper_val: float | None = None,
    render: bool = False,
    frames: list | None = None,
    camera_name: str = "frontview",
):
    """
    Interpolate from current EEF pose to (target_pos, target_quat) over `steps`
    using OSC deltas. Works whether or not a gripper channel exists.

    Required outer-scope/args:
        env, eef_id, adim, record_q
    Optional:
        frames list if you want to append rendered images.
    """
    # Detect gripper
    has_gripper = adim > 6
    grip_idx = adim - 1 if has_gripper else None

    # Start pose
    start_pos  = env.sim.data.site_xpos[eef_id].copy()
    R_now      = env.sim.data.site_xmat[eef_id].reshape(3, 3)
    start_quat = mat2quat(R_now)

    fracs = np.linspace(0.0, 1.0, steps + 1)[1:]
    for f in fracs:
        p_des = (1 - f) * start_pos + f * target_pos
        q_des = quat_slerp(start_quat, target_quat, f)

        a = np.zeros(adim)

        # position
        cur_pos = env.sim.data.site_xpos[eef_id].copy()
        a[0:3] = (p_des - cur_pos) * pos_gain

        # orientation
        q_now = mat2quat(env.sim.data.site_xmat[eef_id].reshape(3, 3))
        q_rel = quat_multiply(q_des, quat_inverse(q_now))
        w     = q_rel[3]
        th    = 2 * np.arccos(np.clip(w, -1, 1))
        if abs(th) < 1e-6:
            axis = np.zeros(3)
        else:
            axis = q_rel[:3] / np.sin(th / 2)
        a[3:6] = axis * th * ori_gain

        if has_gripper and gripper_val is not None:
            a[grip_idx] = gripper_val

        env.step(a)
        record_q()
        if render and frames is not None:
            img = env.sim.render(640, 480, camera_name=camera_name)
            frames.append(np.flipud(img))


def restore_environment(
    env,
    environment_setting: Optional[List[float]],
):
    """
    Restores the robosuite environment based on environment _setting

    Args:
        env: Robosuite environment
        environment_setting: Dict with keys {"qpos","qvel","body_pos","body_quat"}
    """
    env.sim.model.body_pos[:] = np.copy(environment_setting["body_pos"])
    env.sim.model.body_quat[:]= np.copy(environment_setting["body_quat"])

    mujoco.mj_setConst(env.sim.model._model, env.sim.data._data)  # rebuild derived constants

    env.sim.data.qpos[:]      = np.copy(environment_setting["qpos"])
    env.sim.data.qvel[:]      = np.copy(environment_setting["qvel"])
    
    # Optional extras if present
    if "ctrl" in environment_setting:
        env.sim.data.ctrl[:] = environment_setting["ctrl"]
    if "act" in environment_setting:
        env.sim.data.act[:] = environment_setting["act"]
    if "mocap_pos" in environment_setting:
        env.sim.data.mocap_pos[:] = environment_setting["mocap_pos"]
    if "mocap_quat" in environment_setting:
        env.sim.data.mocap_quat[:] = environment_setting["mocap_quat"]

    env.sim.forward()
    

def make_env(
    task_name: str,
    control_freq: int = 20,
    has_renderer: bool = False,
    has_offscreen_renderer: bool = False,
    use_camera_obs: bool = False,
    use_joint_control: bool = False,
    environment_setting: Optional[List[float]] = None,
    training: bool = False,
):
    """
    Build and reset a robosuite environment (currently only 'door'), optionally
    switch the robot to JOINT_POSITION control for replay, and (if provided)
    restore a previously saved MuJoCo state.

    Args:
        task_name: Name of the task. Either 'door', 'wipe', 'two_arm', and 'nut'
        control_freq: Physics/control frequency for the env.
        has_renderer: Whether to create an on-screen viewer.
        has_offscreen_renderer: Whether to enable off-screen rendering.
        use_camera_obs: If True, image observations are returned by env.step().
        use_joint_control: If True, change the Panda controller to absolute
            joint-position control (useful for replaying trajectories).
        environment_setting: Dict with keys {"qpos","qvel","body_pos","body_quat"}
            (all np.ndarray) that fully specify a saved simulator state. When
            given, it is copied into the env after reset to recreate the scene.

    Returns:
        env: The initialized (and possibly restored) robosuite environment.

    Side Effects:
        - Calls env.reset() once to initialize simulator buffers.
        - If `environment_setting` is provided, disables further random placement
          by nulling `env.placement_initializer`.

    Notes:
        Writing only to `data.body_xpos` or `model.body_pos` is insufficient for
        reproducibility when bodies are attached via joints. Copying the entire
        state arrays is robust.
    """

    # 1) for task "door"
    assert (task_name in {"door", "wipe", "two_arm", "nut"}), f"Unsupported task for {task_name}"
    if task_name == "door":
        cfg = load_composite_controller_config(robot="Panda")
        door_sampler = UniformRandomSampler(
            name="door_placer",
            mujoco_objects=None,          # Door() will add the door object internally
            x_range=[-0.05, 0.05],         # push along table x
            y_range=[-0.3, -0.1],
            rotation=(-np.pi/2 - 0.25, -np.pi/2),
            rotation_axis="z",
            reference_pos=(-0.2, -0.35, 0.8),  # same as env.table_offset
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
        )
        if use_joint_control: # for rendering
            arm_key = next(iter(cfg["body_parts"]))
            orig = cfg["body_parts"][arm_key]
            grip_spec = orig["gripper"]
            bp = cfg["body_parts"][arm_key]
            bp.update({
                "type": "JOINT_POSITION",
                "kp": 40.0,
                "kd": 4.0,
                "interpolation": "linear",
                "ndim": 7,
                "input_type": "absolute",
                "input_max": [np.pi]*7,
                "input_min": [-np.pi]*7,
                "output_max": [np.pi]*7,
                "output_min": [-np.pi]*7,
                "gripper": grip_spec,
            })
        env = Door(
            robots="Panda",
            controller_configs=cfg,
            placement_initializer=door_sampler,
            use_latch=True,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            initialization_noise={'magnitude': 0.2, 'type': "uniform"},
            use_camera_obs=use_camera_obs,
            camera_names=["frontview"],
            camera_heights=[480],
            camera_widths=[640],
            camera_depths=[False],
            control_freq=control_freq,
            reward_shaping=True,
        )
        env.reset()
        if environment_setting is not None:
            restore_environment(env, environment_setting)
            env.placement_initializer = None
            env.sim.forward()   # now forward will KEEP it

    elif task_name == "wipe":
        cfg = load_composite_controller_config(robot="Panda")
        if use_joint_control:
            arm_key = next(iter(cfg["body_parts"]))
            orig = cfg["body_parts"][arm_key]
            grip_spec = orig["gripper"]
            bp = cfg["body_parts"][arm_key]
            bp.update({
                "type": "JOINT_POSITION",
                "kp": 40.0,
                "kd": 2.0,
                "interpolation": "linear",
                "ndim": 7,
                "input_type": "absolute",
                "input_max": [np.pi]*7,
                "input_min": [-np.pi]*7,
                "output_max": [np.pi]*7,
                "output_min": [-np.pi]*7,
                "gripper": grip_spec,
            })
        env = Wipe(
            robots="Panda",
            controller_configs=cfg,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            camera_names=["frontview"],
            camera_heights=[480] * 3,
            camera_widths =[640] * 3,
            camera_depths =[False] * 3,
            control_freq=control_freq,
        )
        env.task_config["num_markers"] = 50
        env.num_markers = 50
        env.task_config["table_full_size"] = [0.4, 0.6, 0.05]
        env.table_full_size = [0.4, 0.6, 0.05]
        env.task_config['contact_threshold'] = 0.01
        env.contact_threshold = 0.01
        env.task_config["table_offset"] = [0.3, 0, 1.0]
        env.table_offset = [0.3, 0, 1.0]
        
        env.reset()

        if environment_setting is not None:
            arena = env.model.mujoco_arena
            delta_z = 0.005
            restore_environment(env, environment_setting)
            for marker in arena.markers:
                bid = env.sim.model.body_name2id(marker.root_body)
                env.sim.model.body_pos[bid] += np.array([0.0, 0.0, delta_z])
            env.placement_initializer = None
            env.sim.forward()

    elif task_name == "two_arm":
        cfg = load_composite_controller_config(robot="Panda")
        if use_joint_control: # for rendering
            for arm_key, bp in cfg["body_parts"].items():
                grip_spec = bp["gripper"]
                bp.update({
                    "type": "JOINT_POSITION",
                    "kp": 40.0,
                    "kd": 2.0,
                    "interpolation": "linear",
                    "ndim": 7,
                    "input_type": "absolute",
                    "input_max": [np.pi]*7,
                    "input_min": [-np.pi]*7,
                    "output_max": [np.pi]*7,
                    "output_min": [-np.pi]*7,
                    "gripper": grip_spec,
                })
        env = TwoArmLift(
            robots=["Panda", "Panda"],
            controller_configs=cfg,
            env_configuration="parallel",
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            camera_names=["frontview"],
            camera_heights=[480],
            camera_widths=[640],
            camera_depths=[False],
            control_freq=control_freq,
        )
        env.reset()
        if environment_setting is not None:
            restore_environment(env, environment_setting)
            env.placement_initializer = None
            env.sim.forward()
    else:
        cfg = load_composite_controller_config(robot="Panda")
        if use_joint_control: # for rendering
            arm_key = next(iter(cfg["body_parts"]))
            orig = cfg["body_parts"][arm_key]
            grip_spec = orig["gripper"]
            bp = cfg["body_parts"][arm_key]
            bp.update({
                "type": "JOINT_POSITION",
                "kp": 40.0,
                "kd": 2.0,
                "interpolation": "linear",
                "ndim": 7,
                "input_type": "absolute",
                "input_max": [np.pi]*7,
                "input_min": [-np.pi]*7,
                "output_max": [np.pi]*7,
                "output_min": [-np.pi]*7,
                "gripper": grip_spec,
            })
        env = NutAssembly(
            robots="Panda",
            single_object_mode=2,
            nut_type="square",
            controller_configs=cfg,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            camera_names=["frontview"],
            camera_heights=[480],
            camera_widths =[640],
            camera_depths =[False],
            control_freq=20,
        )
        delta_z = 0.1       
        env.table_offset = [0, 0, 0.82 + delta_z]
        env.placement_initializer = None
        env.reset()
        if environment_setting is not None:
            if training and delta_z:
                m = env.sim.model
                for name in m.body_names:
                    n = name.lower()
                    if ("peg" in n) or ("stand" in n) or ("board" in n):
                        m.body_pos[m.body_name2id(name)][2] += delta_z
            mujoco.mj_setConst(env.sim.model._model, env.sim.data._data)

            restore_environment(env, environment_setting)

            # restore_mj_state(env, environment_setting) # restore via full mujoco setting
            env.sim.forward()
        
    return env


def write_grid_video(
    episodes_frames: List[List[np.ndarray]],
    path: str,
    grid_shape=(5,5),
    fps: int = 20
):
    """
    Stitch multiple episode videos (lists of RGB frames) into a single grid video.

    Args:
        episodes_frames: Outer list = episodes, inner list = frames (H*W*3).
        path: Output .mp4 file path.
        grid_shape: (rows, cols) layout of the grid.
        fps: Frames per second for the output video.

    Raises:
        AssertionError: If the number of episodes exceeds rows*cols.

    Behavior:
        - Truncates all episodes to the shortest length so they align in time.
        - Fills empty grid slots (if any) with black frames.
    """

    rows, cols = grid_shape
    num_eps = len(episodes_frames)
    assert num_eps <= rows*cols, "Too many episodes for grid!"
    min_len = min(len(frames) for frames in episodes_frames)
    H, W, _ = episodes_frames[0][0].shape

    grid_frames = []
    for t in range(min_len):
        rows_imgs = []
        for r in range(rows):
            cells = []
            for c in range(cols):
                idx = r*cols + c
                if idx < num_eps:
                    cells.append(episodes_frames[idx][t])
                else:
                    cells.append(np.zeros((H, W, 3), dtype=np.uint8))
            rows_imgs.append(np.concatenate(cells, axis=1))
        grid_frames.append(np.concatenate(rows_imgs, axis=0))

    imageio.mimsave(path, grid_frames, fps=fps)


# Wrapper for compute_smooth_trajectory_{gripper type}
def compute_smooth_trajectory(
    env_r,
    task_name,
    q_keys: np.ndarray,
    control_freq: int,
) -> np.ndarray:
    gripper_ids = [
        env_r.sim.model.get_joint_qpos_addr(joint_name)
        for robot in env_r.robots
        for gripper in robot.gripper.values()
        for joint_name in gripper.joints
    ]
    if task_name == "wipe":
        q_high = compute_smooth_trajectory_wipper(
            q_keys,
            control_freq=control_freq/20,
            render_freq=control_freq/2,
        )
    elif task_name == "nut":
        q_high = compute_smooth_trajectory_gripper(
            q_keys,
            gripper_ids=gripper_ids,
            control_freq=control_freq/15,
            render_freq=control_freq/2,
            open_after_end=True,
            closure_steps=1,
            closure_insertion=0
        )
    elif task_name == "door":
        q_high = compute_smooth_trajectory_gripper(
            q_keys,
            gripper_ids=gripper_ids,
            control_freq=control_freq/15,
            render_freq=control_freq/2,
            closure_steps=1,
            closure_insertion=9
        )
    else: # two arm
        q_high = compute_smooth_trajectory_gripper(
            q_keys,
            gripper_ids=gripper_ids,
            control_freq=control_freq/15,
            render_freq=control_freq/2,
            closure_steps=1,
            closure_insertion=12 # new insertion
        )
    
    return q_high


def compute_smooth_trajectory_gripper(
    joint_keys: np.ndarray,
    gripper_ids = (7, 8),
    control_freq: int = 20,
    render_freq: int = 120,
    closure_steps: int = 1,
    closure_insertion: int = 5,
    rest_steps: int = 0,
    open_after_end: bool = False,
) -> np.ndarray:
    """
    Upsample sparse joint keyframes to a higher rate using cubic splines for tasks with single gipper,
    inserting a short 'closure' segment that duplicates one keyframe (e.g., to
    hold the gripper closed) and shifts subsequent timestamps accordingly.

    Args:
        joint_keys: (N, dof) low-rate keyframes.
        control_freq: Sampling rate of the keyframes (Hz).
        render_freq: Desired high-rate sampling (Hz).
        closure_steps: Number of duplicated frames to insert.
        closure_insertion: 1-based index of the keyframe to insert.

    Returns:
        q_high: (T_high, dof) high-rate trajectory.
    """
    # print(control_freq, render_freq)

    dof    = joint_keys.shape[1]
    dt_low = 1.0 / control_freq
    dt_high= 1.0 / render_freq

    # original key times
    N = joint_keys.shape[0]
    t_low = np.arange(N) * dt_low

    # closure insertion
    t_cl = t_low[closure_insertion] + np.arange(0, closure_steps)*dt_low
    q_cl = np.tile((joint_keys[max(0, closure_insertion-1),:]), (closure_steps,1))

    t_low_after = closure_steps * dt_low + t_low[closure_insertion:]
    t_ext = np.concatenate([t_low[:closure_insertion], t_cl, t_low_after])
    q_ext = np.vstack([joint_keys[:closure_insertion], q_cl, joint_keys[closure_insertion:]])
    q_ext[:closure_insertion, gripper_ids] = -1.0
    q_ext[closure_insertion:, gripper_ids] = 1.0

    # splines
    splines = [CubicSpline(t_ext, q_ext[:,j], extrapolate=False) for j in range(dof)]

    # sample high‑rate
    t_high = np.arange(t_ext[0], t_ext[-1], dt_high)
    q_high = np.stack([[s(t) for s in splines] for t in t_high])

    if rest_steps:
        prefix = np.vstack([q_high[0]] * rest_steps)                # also (rest_steps, dof)
        q_high = np.vstack([prefix, q_high])                       # same result

    # open after end
    if open_after_end:
        open_steps = 30
        # take the very last configuration and repeat it
        last_cfg = q_high[-1:].copy()                         # shape (1, dof)
        pad      = np.repeat(last_cfg, open_steps, axis=0)    # shape (10, dof)
        # force gripper joints to “open” = –1.0
        for gid in gripper_ids:
            pad[:, gid] = -1.0
        # append
        q_high = np.vstack([q_high, pad])
    return q_high


def compute_smooth_trajectory_wipper(
    joint_keys: np.ndarray,
    control_freq: int = 20,
    render_freq: int = 120,
) -> np.ndarray:
    dof    = joint_keys.shape[1]
    dt_low = 1.0 / control_freq
    dt_high= 1.0 / render_freq

    # original key times
    N = joint_keys.shape[0]
    t_low = np.arange(N) * dt_low

    # splines
    splines = [CubicSpline(t_low, joint_keys[:,j], extrapolate=False) for j in range(dof)]

    # sample high‑rate
    t_high = np.arange(t_low[0], t_low[-1], dt_high)
    q_high = np.stack([[s(t) for s in splines] for t in t_high])
    return q_high


def _plan_linear(q0, q1, duration, hz):
    """
    Linear interpolation in joint space.
    Returns:
      t: (T,) timestamps in [0, duration]
      q: (T, D) waypoints with q[0]=q0, q[-1]=q1
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    assert q0.shape == q1.shape, "q0 and q1 must have same shape"
    D = q0.shape[0]

    if duration <= 0 or hz <= 0:
        # Degenerate: just return the target
        return np.array([0.0]), np.asarray(q1, dtype=float).reshape(1, D)

    # +1 to include both endpoints; ~1/hz spacing
    T = max(int(np.round(duration * hz)) + 1, 2)
    t = np.linspace(0.0, duration, T, endpoint=True)

    alpha = (t / duration)[:, None]  # (T, 1)
    q = (1.0 - alpha) * q0 + alpha * q1

    # Numerical safety
    q[0]  = q0
    q[-1] = q1
    return t, q
def compute_smooth_trajectory_with_q0(
    env_r,
    task_name,
    q_keys: np.ndarray,
    control_freq: int,
    q0: np.ndarray,
    approach_duration: float = 20.0,   # seconds; tweak as you like
    pause_steps: int = 50,            # number of frames to pause at first key
):
    """
    Prepend a linear interpolation from q0 -> q_keys[0] to the trajectory produced by
    compute_smooth_trajectory(...). Uses joint-position lerp only.

    Args:
        env_r: robosuite env (passed through to compute_smooth_trajectory)
        task_name: task selector for your internal logic
        q_keys: (T_low, D) or (D,) key poses
        control_freq: controller/sim stepping frequency (Hz)
        q0: (D,) starting joint configuration
        approach_duration: duration (s) for the initial linear segment

    Returns:
        q_high: (T_high, D) full trajectory starting from q0 and moving smoothly
                into the first keyframe, then following the smoothed plan.
    """
    q_keys = np.asarray(q_keys, dtype=float)
    q0     = np.asarray(q0,     dtype=float)

    # Normalize shapes
    if q_keys.ndim == 1:
        q_keys = q_keys[None, :]  # (1, D)

    # Basic shape checks
    assert q0.ndim == 1, "q0 must be shape (D,)"
    assert q_keys.ndim == 2, "q_keys must be shape (T, D) or (D,)"
    D = q_keys.shape[1]
    assert q0.shape[0] == D, f"Dim mismatch: q0 has {q0.shape[0]} dims, q_keys has {D}"

    # If already at the first key pose, skip the approach
    first_key = q_keys[0]
    if np.allclose(q0, first_key, atol=1e-9, rtol=0.0) or (approach_duration <= 0):
        return compute_smooth_trajectory(env_r, task_name, q_keys, control_freq)

    # 1) Linear prefix from q0 -> q_keys[0]
    #    We drop the last sample to avoid duplicating the first frame of the main segment.
    t_app, Q_app = _plan_linear(q0, first_key, duration=float(approach_duration), hz=int(control_freq))
    if Q_app.shape[0] > 1:
        Q_prefix = Q_app[:-1]
    else:
        Q_prefix = Q_app  # degenerate (very short) case

    # 2) Main segment (your existing builder)
    Q_main = compute_smooth_trajectory(env_r, task_name, q_keys, control_freq)

    if pause_steps > 0:
        pause_block = np.repeat(first_key[None, :], pause_steps, axis=0)
    else:
        pause_block = np.empty((0, D), dtype=float)

    if task_name in ["door", "two_arm"]:
        gripper_ids = [
            env_r.sim.model.get_joint_qpos_addr(joint_name)
            for robot in env_r.robots
            for gripper in robot.gripper.values()
            for joint_name in gripper.joints
        ]
        Q_prefix[:, gripper_ids] = -1.0
        pause_block[:, gripper_ids] = -1.0
    elif task_name == "nut":
        gripper_ids = [
            env_r.sim.model.get_joint_qpos_addr(joint_name)
            for robot in env_r.robots
            for gripper in robot.gripper.values()
            for joint_name in gripper.joints
        ]
        Q_prefix[:, gripper_ids] = 1.0
        pause_block[:, gripper_ids] = 1.0

    # Concatenate
    q_high = np.vstack([Q_prefix, pause_block, Q_main])
    return q_high


def _to_action_from_q(q, task_name):
    """Match exactly the action packing in env.step(...)."""
    if task_name in ["door", "nut"]:
        arm_q   = q[:7]
        grip_sc = float((q[7] + q[8]) / 2.0)   # your current compression
        return np.concatenate([arm_q, [grip_sc]])
    elif task_name == "wipe":
        return q[:7]
    else:  # two_arm
        arm1 = q[0:7]
        grip1 = float((q[7] + q[8]) / 2.0)
        arm2 = q[9:16]
        grip2 = float((q[16] + q[17]) / 2.0)
        return np.concatenate([arm1, [grip1], arm2, [grip2]])
def render_trajectory(
    env,
    task_name: str,
    q_high: np.ndarray,
    initial_pose: np.ndarray,
    fps: int = 60,
    camera_name: str = "frontview",
    hold_init: bool = False,
    set_init: bool = True,
) -> List[np.ndarray]:
    """
    Replay a high-rate joint trajectory in the environment and capture rendered frames.

    Args:
        env: The robosuite environment (already restored to the correct scene).
        q_high: (T, dof) joint trajectory produced by `compute_smooth_trajectory`.
        initial_pose: (dof,) joint configuration to reset the robot before replay.
        fps: Rendered video FPS (used only for consistency when saving later).
        camera_name: Name of the MuJoCo camera to render.
        hold_init: For stability, hold for 100 steps before execution.
        set_init: Force mujoco states (NOT IMPLEMENTED WELL)

    Returns:
        frames: List of RGB images (H*W*3), one per step.

    Notes:
        - Joint indices are resolved once, then we set `env.sim.data.qpos[joint_idx]`.
        - For a two-DOF gripper, we average the two joint values to produce a single
          scalar action (matches Door controller interface).
        - If you switched to a JOINT_POSITION controller, ensure the action format
          matches (absolute joint targets) instead of OSC deltas.
    """
    
    # restore robot + door
    if set_init:
        offset = 0
        for robot in env.robots:
            # collect this robot's joint names
            arm_names  = robot.robot_model.joints
            grip_names = next(iter(robot.gripper.values())).joints
            names      = arm_names + grip_names

            # how many values to pull from initial_pose
            n = len(names)

            # look up their indices in qpos
            idx = [env.sim.model.get_joint_qpos_addr(nm) for nm in names]

            # copy the slice of initial_pose into sim.data.qpos
            env.sim.data.qpos[idx] = initial_pose[offset:offset + n]
            env.sim.data.qvel[:] = 0

            offset += n
        env.sim.forward()

    terminated = False
    if hold_init:
        for _ in range(100):
            _, _, done, _ = env.step(_to_action_from_q(q_high[0], task_name))
            if done:
                terminated = True
                break

    frames = []
    if terminated:
        img = env.sim.render(640,480, camera_name=camera_name)
        frames.append(np.flipud(img))
        return frames

    for q in q_high:
        action = _to_action_from_q(q, task_name)
        
        _, _, done, _ = env.step(action)

        img = env.sim.render(640,480, camera_name=camera_name)
        frames.append(np.flipud(img))
        if done:
            break
    return frames
