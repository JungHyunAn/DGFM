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
     • door / nut:  7 arm joints + 1 normalized gripper pose
     • wipe:        7 arm joints
     • two_arm:     ([7 arm] + [1 normalized gripper pose]) x 2
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
- Two-arm lift: pot placement is restricted to a tighter region than RoboSuite's
  default sampler.
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
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.transform_utils import mat2quat, quat_inverse, quat_multiply, quat_slerp

PANDA_GRIPPER_OPEN_QPOS = 0.04
ACTION_REPRESENTATIONS = ("joint_space", "task_space")
DEFAULT_VISION_CAMERAS = ("frontview", "robot0_eye_in_hand")
DEFAULT_VISION_HEIGHT = 224
DEFAULT_VISION_WIDTH = 224
NUT_FRONTVIEW_POSITION = np.array([1.0, 0.0, 1.45], dtype=np.float64)


def capture_camera_views(
    env,
    camera_names=DEFAULT_VISION_CAMERAS,
    width: int = DEFAULT_VISION_WIDTH,
    height: int = DEFAULT_VISION_HEIGHT,
) -> dict[str, np.ndarray]:
    """Capture vertically-correct RGB images for all configured cameras."""
    return {
        name: np.flipud(env.sim.render(width, height, camera_name=name)).copy()
        for name in camera_names
    }


def validate_action_representation(action_representation: str) -> str:
    if action_representation not in ACTION_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported action_representation={action_representation!r}. "
            f"Expected one of {ACTION_REPRESENTATIONS}."
        )
    return action_representation




def _mat_to_roll_pitch_yaw(mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to xyz fixed-axis Euler angles."""
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    sy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
    singular = sy < 1e-6
    if singular:
        roll = np.arctan2(-mat[1, 2], mat[1, 1])
        pitch = np.arctan2(-mat[2, 0], sy)
        yaw = 0.0
    else:
        roll = np.arctan2(mat[2, 1], mat[2, 2])
        pitch = np.arctan2(-mat[2, 0], sy)
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _first_id(value):
    if isinstance(value, dict):
        return next(iter(value.values()))
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def get_square_nut_pose(env) -> np.ndarray:
    """Return square nut position and orientation as [x, y, z, roll, pitch, yaw]."""
    body_id = None
    for attr in ("obj_body_id", "object_body_ids"):
        if hasattr(env, attr):
            body_id = _first_id(getattr(env, attr))
            break

    if body_id is not None:
        nut_pos = env.sim.data.body_xpos[body_id].copy()
        nut_rot = env.sim.data.body_xmat[body_id].reshape(3, 3)
    else:
        nut_site_id = env.object_site_ids[0]
        nut_pos = env.sim.data.site_xpos[nut_site_id].copy()
        nut_rot = env.sim.data.site_xmat[nut_site_id].reshape(3, 3)
    return np.concatenate([nut_pos.astype(np.float32), _mat_to_roll_pitch_yaw(nut_rot)], axis=0)


def configure_nut_frontview(env, camera_name: str = "frontview") -> bool:
    """Tighten the nut-task view while keeping the Panda and target peg visible."""
    try:
        camera_id = env.sim.model.camera_name2id(camera_name)
    except (KeyError, ValueError):
        return False
    if camera_id < 0:
        return False

    # Keep RoboSuite's original camera orientation and move only along its
    # viewing axis. At x=1.0 the reset and insertion poses remain fully framed,
    # while the task occupies substantially more of a 4:3 or square image.
    env.sim.model.cam_pos[camera_id] = NUT_FRONTVIEW_POSITION
    env.sim.forward()
    return True


def configure_nut_pegs(env, delta_x: float = -0.05, delta_z: float = 0.1) -> None:
    """Apply the deterministic nut peg layout and task-specific camera view."""
    if not hasattr(env, "_dgfm_nut_peg1_base_pos"):
        env._dgfm_nut_peg1_base_pos = env.sim.model.body_pos[env.peg1_body_id].copy()
        env._dgfm_nut_peg2_base_pos = env.sim.model.body_pos[env.peg2_body_id].copy()

    peg1_pos = env._dgfm_nut_peg1_base_pos.copy()
    peg1_pos[0] += delta_x
    peg1_pos[2] += delta_z
    peg2_pos = env._dgfm_nut_peg2_base_pos.copy()
    peg2_pos[2] = 0.0

    env.sim.model.body_pos[env.peg1_body_id] = peg1_pos
    env.sim.model.body_pos[env.peg2_body_id] = peg2_pos
    env.sim.forward()
    configure_nut_frontview(env)


def _gripper_qpos_to_normalized(gripper_qpos: np.ndarray) -> np.ndarray:
    """Map Panda finger qpos to normalized gripper pose: 0 closed, 1 open."""
    gripper_qpos = np.asarray(gripper_qpos, dtype=np.float32)
    return np.clip(
        np.mean(np.abs(gripper_qpos), axis=-1) / PANDA_GRIPPER_OPEN_QPOS,
        0.0,
        1.0,
    )


def _gripper_qpos_trace_to_normalized(gripper_qpos: np.ndarray) -> np.ndarray:
    """Normalize a recorded gripper trace by aperture within the episode."""
    aperture = np.mean(np.abs(np.asarray(gripper_qpos, dtype=np.float32)), axis=-1, keepdims=True)
    lo = np.min(aperture, axis=0, keepdims=True)
    hi = np.max(aperture, axis=0, keepdims=True)
    span = hi - lo
    if float(np.max(span)) > 1e-6:
        return np.clip((aperture - lo) / span, 0.0, 1.0).astype(np.float32)
    return np.clip(aperture / PANDA_GRIPPER_OPEN_QPOS, 0.0, 1.0).astype(np.float32)


def _normalized_gripper_to_action(value) -> float:
    """Map normalized gripper pose to controller command: 0 closed, 1 open."""
    value = float(np.clip(value, 0.0, 1.0))
    return 1.0 - 2.0 * value


def _clip_policy_gripper_dims(q: np.ndarray, task_name: str) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).copy()
    if task_name in ["door", "nut"] and q.shape[-1] == 8:
        q[..., 7] = np.clip(q[..., 7], 0.0, 1.0)
    elif task_name == "two_arm" and q.shape[-1] == 16:
        q[..., 7] = np.clip(q[..., 7], 0.0, 1.0)
        q[..., 15] = np.clip(q[..., 15], 0.0, 1.0)
    return q


def _normalized_gripper_to_qpos(value) -> np.ndarray:
    value = float(np.clip(value, 0.0, 1.0))
    return np.full((2,), value * PANDA_GRIPPER_OPEN_QPOS, dtype=np.float32)


def _site_pose(env, site_id: int) -> np.ndarray:
    pos = env.sim.data.site_xpos[site_id].copy().astype(np.float32)
    quat = mat2quat(env.sim.data.site_xmat[site_id].reshape(3, 3)).astype(np.float32)
    return np.concatenate([pos, quat], axis=0)


def _robot_eef_pose(env, robot) -> np.ndarray:
    return _site_pose(env, list(robot.eef_site_id.values())[0])


def _eef_action_delta(env, site_id: int, target_pose: np.ndarray, pos_gain: float = 100.0, ori_gain: float = 1.0) -> np.ndarray:
    target_pose = np.asarray(target_pose, dtype=np.float32)
    target_pos = target_pose[:3]
    target_quat = target_pose[3:7]
    quat_norm = np.linalg.norm(target_quat)
    if quat_norm < 1e-6:
        target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        target_quat = target_quat / quat_norm
    cur_pos = env.sim.data.site_xpos[site_id].copy()
    cur_quat = mat2quat(env.sim.data.site_xmat[site_id].reshape(3, 3))
    if np.dot(target_quat, cur_quat) < 0.0:
        target_quat = -target_quat

    action = np.zeros(6, dtype=np.float32)
    action[0:3] = (target_pos - cur_pos) * pos_gain
    q_rel = quat_multiply(target_quat, quat_inverse(cur_quat))
    w = q_rel[3]
    th = 2 * np.arccos(np.clip(w, -1, 1))
    if abs(th) < 1e-6:
        axis = np.zeros(3, dtype=np.float32)
    else:
        axis = q_rel[:3] / np.sin(th / 2)
    action[3:6] = axis * th * ori_gain
    return action


def _canonicalize_quat_sequence(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float32).copy()
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = np.divide(quats, np.maximum(norms, 1e-6), out=quats, where=norms > 0)
    for i in range(1, quats.shape[0]):
        if np.dot(quats[i - 1], quats[i]) < 0.0:
            quats[i] = -quats[i]
    return quats


def compose_task_space_trajectory(
    task_name: str,
    eef_trace: np.ndarray,
    gripper_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Pack EEF pose traces into the policy trajectory representation."""
    eef_trace = np.asarray(eef_trace, dtype=np.float32).copy()
    if task_name in ["door", "nut"]:
        if eef_trace.shape[-1] != 7:
            raise ValueError(f"{task_name} task-space trajectory must have 7 EEF pose dims")
        if gripper_pose is None:
            raise ValueError(f"{task_name} task-space trajectory requires gripper_pose")
        eef_trace[:, 3:7] = _canonicalize_quat_sequence(eef_trace[:, 3:7])
        grip = np.clip(np.asarray(gripper_pose, dtype=np.float32).reshape(-1, 1), 0.0, 1.0)
        if grip.shape[0] != eef_trace.shape[0]:
            raise ValueError("gripper_pose length must match eef_trace length")
        return np.concatenate([eef_trace, grip], axis=-1).astype(np.float32)
    if task_name == "wipe":
        if eef_trace.shape[-1] != 7:
            raise ValueError("wipe task-space trajectory must have 7 EEF pose dims")
        eef_trace[:, 3:7] = _canonicalize_quat_sequence(eef_trace[:, 3:7])
        return eef_trace.astype(np.float32)
    if task_name == "two_arm":
        if eef_trace.shape[-1] != 14:
            raise ValueError("two_arm task-space trajectory must have 14 EEF pose dims")
        if gripper_pose is None:
            raise ValueError("two_arm task-space trajectory requires gripper_pose")
        eef_trace[:, 3:7] = _canonicalize_quat_sequence(eef_trace[:, 3:7])
        eef_trace[:, 10:14] = _canonicalize_quat_sequence(eef_trace[:, 10:14])
        grip = np.clip(np.asarray(gripper_pose, dtype=np.float32).reshape(eef_trace.shape[0], -1), 0.0, 1.0)
        if grip.shape[1] != 2:
            raise ValueError("two_arm gripper_pose must have shape (T, 2)")
        return np.concatenate(
            [eef_trace[:, 0:7], grip[:, 0:1], eef_trace[:, 7:14], grip[:, 1:2]],
            axis=-1,
        ).astype(np.float32)
    raise ValueError(f"Unsupported task for task-space trajectory: {task_name}")


def normalize_policy_trajectory(
    task_name: str,
    q_trace: np.ndarray,
    gripper_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw robot qpos traces to compact policy poses with normalized grippers.

    Shapes:
      door/nut:  raw 9  -> policy 8  = 7 arm + 1 normalized gripper
      two_arm:   raw 18 -> policy 16 = (7 arm + 1 normalized gripper) x 2
      wipe:      raw >=7 -> policy 7 = arm only
    """
    q_trace = np.asarray(q_trace, dtype=np.float32)
    if gripper_pose is not None:
        gripper_pose = np.asarray(gripper_pose, dtype=np.float32)
    if task_name in ["door", "nut"]:
        if q_trace.shape[-1] < 9:
            raise ValueError(f"{task_name} trajectory must have at least 9 raw qpos dims")
        if gripper_pose is not None:
            gripper_pose = gripper_pose.reshape(-1, 1)
            if gripper_pose.shape[0] != q_trace.shape[0]:
                raise ValueError("gripper_pose length must match q_trace length")
            grip = np.clip(gripper_pose, 0.0, 1.0)
        else:
            grip = _gripper_qpos_trace_to_normalized(q_trace[:, 7:9])
        return np.concatenate(
            [q_trace[:, :7], grip],
            axis=-1,
        ).astype(np.float32)
    if task_name == "two_arm":
        if q_trace.shape[-1] < 18:
            raise ValueError("two_arm trajectory must have at least 18 raw qpos dims")
        if gripper_pose is not None:
            gripper_pose = gripper_pose.reshape(q_trace.shape[0], -1)
            if gripper_pose.shape[1] != 2:
                raise ValueError("two_arm gripper_pose must have shape (T, 2)")
            left_grip = np.clip(gripper_pose[:, 0:1], 0.0, 1.0)
            right_grip = np.clip(gripper_pose[:, 1:2], 0.0, 1.0)
        else:
            left_grip = _gripper_qpos_trace_to_normalized(q_trace[:, 7:9])
            right_grip = _gripper_qpos_trace_to_normalized(q_trace[:, 16:18])
        return np.concatenate(
            [q_trace[:, 0:7], left_grip, q_trace[:, 9:16], right_grip],
            axis=-1,
        ).astype(np.float32)
    if task_name == "wipe":
        return q_trace[:, :7].astype(np.float32)
    raise ValueError(f"Unsupported task for policy trajectory normalization: {task_name}")


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


def get_environment_state(env, task_name: str) -> np.ndarray:
    """Return task oracle state recorded separately from robot joint angles.

    Door records handle/latch angle and door hinge angle. Wipe records the
    current remaining-dirt center and maximum radius. Nut records the live
    square nut pose as position plus roll/pitch/yaw.
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

    if task_name == "nut":
        return get_square_nut_pose(env)

    # TODO(two_arm): Record pot/handle dynamic pose or lifted height during rollout.
    if task_name == "wipe":
        try:
            max_radius, center, _ = env._get_wipe_information()
            state = np.array([center[0], center[1], max_radius], dtype=np.float32)
            if np.all(np.isfinite(state)):
                env._dgfm_last_wipe_dynamic_state = state
                return state
        except Exception:
            pass

        state = np.asarray(
            getattr(env, "_dgfm_last_wipe_dynamic_state", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ).copy()
        state[2] = 0.0
        return state

    return np.zeros((0,), dtype=np.float32)


def get_dynamic_state(env, task_name: str) -> np.ndarray:
    """Backward-compatible alias for datasets produced before environment_states."""
    return get_environment_state(env, task_name)


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
    action_representation: str = "joint_space",
    camera_names=None,
    camera_heights=None,
    camera_widths=None,
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
        action_representation: "joint_space" uses the current joint-space controller
            behavior. "task_space" keeps the default OSC controller for EEF targets.
        environment_setting: Dict with keys {"qpos","qvel","body_pos","body_quat"}
            (all np.ndarray) that fully specify a saved simulator state. When
            given, it is copied into the env after reset to recreate the scene.
        training: If the environment is used for training&eval stage

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

    camera_names = list(camera_names or DEFAULT_VISION_CAMERAS)
    camera_heights = list(camera_heights or [DEFAULT_VISION_HEIGHT] * len(camera_names))
    camera_widths = list(camera_widths or [DEFAULT_VISION_WIDTH] * len(camera_names))
    if not (len(camera_names) == len(camera_heights) == len(camera_widths)):
        raise ValueError("camera_names, camera_heights, and camera_widths must have equal lengths")
    camera_depths = [False] * len(camera_names)

    # 1) for task "door"
    assert (task_name in {"door", "wipe", "two_arm", "nut"}), f"Unsupported task for {task_name}"
    action_representation = validate_action_representation(action_representation)
    if action_representation == "task_space":
        use_joint_control = False
    if task_name == "door":
        cfg = load_composite_controller_config(robot="Panda")
        door_sampler = UniformRandomSampler(
            name="door_placer",
            mujoco_objects=None,          # Door() will add the door object internally
            x_range=[-0.075, 0.075],         # push along table x
            y_range=[-0.3, -0.1],
            rotation=(-np.pi/2 - np.pi/8, -np.pi/2),
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

        # initialization_noise_magnitude = 0.5 # more variance for dataset generation
        #if training:
        #    initialization_noise_magnitude = 0.2
        initialization_noise_magnitude = 0.2

        env = Door(
            robots="Panda",
            controller_configs=cfg,
            placement_initializer=door_sampler,
            use_latch=True,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            initialization_noise={'magnitude': initialization_noise_magnitude, 'type': "uniform"},
            use_camera_obs=use_camera_obs,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
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
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            control_freq=control_freq,
        )
        env.task_config["num_markers"] = 50
        env.num_markers = 50
        # env.task_config["two_clusters"] = True
        # env.two_clusters = True
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
        two_arm_sampler = UniformRandomSampler(
            name="pot_placer",
            mujoco_objects=None,  # TwoArmLift adds the pot object internally
            x_range=[-0.015, 0.015],
            y_range=[-0.015, 0.015],
            rotation=(np.pi - np.pi / 6, np.pi + np.pi / 6),
            rotation_axis="z",
            reference_pos=(0.0, 0.0, 0.8),  # TwoArmLift table offset
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
        )
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
            placement_initializer=two_arm_sampler,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            control_freq=control_freq,
        )
        env.reset()
        if environment_setting is not None:
            restore_environment(env, environment_setting)
            env.placement_initializer = None
            env.sim.forward()
    elif task_name == "nut":
        cfg = load_composite_controller_config(robot="Panda")
        nut_sampler = SequentialCompositeSampler(name="ObjectSampler")
        nut_sampler.append_sampler(
            sampler=UniformRandomSampler(
                name="SquareNutSampler",
                x_range=[-0.14, -0.11],
                y_range=[0.11, 0.14],
                rotation=(np.pi - np.pi / 6, np.pi + np.pi / 6),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0.0, 0.0, 0.82),
                z_offset=0.12,
            )
        )
        nut_sampler.append_sampler(
            sampler=UniformRandomSampler(
                name="RoundNutSampler",
                x_range=[-0.115, -0.11],
                y_range=[-0.225, -0.11],
                rotation=None,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0.0, 0.0, 0.82),
                z_offset=0.12,
            )
        )
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
            placement_initializer=nut_sampler,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            control_freq=20,
        )
        delta_z = 0.1       
        env.table_offset = [0, 0, 0.82 + delta_z]
        env.reset()
        if environment_setting is None:
            configure_nut_pegs(env, delta_x=-0.05, delta_z=delta_z)
        else:
            restore_environment(env, environment_setting)
            env.placement_initializer = None
            env.sim.forward()
            configure_nut_frontview(env)
    else:
        raise ValueError(f"Unsupported task type={task_name}.")

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
        - Runs until the longest episode finishes.
        - Holds shorter episodes on their final frame.
        - Fills empty grid slots (if any) with black frames.
    """

    rows, cols = grid_shape
    num_eps = len(episodes_frames)
    assert num_eps <= rows*cols, "Too many episodes for grid!"
    max_len = max(len(frames) for frames in episodes_frames)
    H, W, _ = episodes_frames[0][0].shape

    grid_frames = []
    for t in range(max_len):
        rows_imgs = []
        for r in range(rows):
            cells = []
            for c in range(cols):
                idx = r*cols + c
                if idx < num_eps:
                    frames = episodes_frames[idx]
                    cells.append(frames[min(t, len(frames) - 1)])
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


def _get_environment_params(
    env,
    task_name: str):
    """Extract compact task parameters used as static policy conditions.

    Each task exposes the object/target pose information needed by training and
    evaluation. Nut assembly has no static parameters; its live square-nut
    pose is recorded in dynamic_states.
    """
    assert (task_name in {"door", "wipe", "two_arm", "nut"}), f"Unsupported task for {task_name}"
    if task_name == "door":
        handle_id  = env.door_handle_site_id
        handle_pos = env._handle_xpos.copy()
        R_handle   = env.sim.data.site_xmat[handle_id].reshape(3,3)
        yaw = np.arctan2(R_handle[1,0], R_handle[0,0])
        environment_parameters = (handle_pos[0], handle_pos[1], yaw)
    elif task_name == "wipe":
        environment_parameters = ()
    elif task_name == "two_arm":
        handle_names = [n for n in env.sim.model.site_names if "handle" in n]
        handle_ids   = [env.sim.model.site_name2id(n) for n in handle_names]
        hidL, hidR = handle_ids[:2]
        posL = env.sim.data.site_xpos[hidL].copy()
        posR = env.sim.data.site_xpos[hidR].copy()
        R0   = env.sim.data.site_xmat[hidL].reshape(3,3)
        yaw  = np.arctan2(R0[1,0], R0[0,0])
        environment_parameters = ((posL[0]+posR[0])/2, (posL[1]+posR[1])/2, yaw)
    else:
        environment_parameters = ()

    return environment_parameters


def _current_robot_q(env, task_name: str, action_representation: str = "joint_space") -> np.ndarray:
    """Return the current robot pose in the compact policy representation.

    Gripper finger qpos values are collapsed into normalized open/close channels
    so state-conditioned policies see the same format they generate.
    """
    action_representation = validate_action_representation(action_representation)
    if action_representation == "task_space":
        if task_name in ["door", "nut"]:
            return np.concatenate([
                _robot_eef_pose(env, env.robots[0]),
                [float(_gripper_qpos_to_normalized(env.sim.data.qpos[7:9]))],
            ]).astype(np.float32)
        if task_name == "wipe":
            return _robot_eef_pose(env, env.robots[0]).astype(np.float32)
        return np.concatenate([
            _robot_eef_pose(env, env.robots[0]),
            [float(_gripper_qpos_to_normalized(env.sim.data.qpos[7:9]))],
            _robot_eef_pose(env, env.robots[1]),
            [float(_gripper_qpos_to_normalized(env.sim.data.qpos[16:18]))],
        ]).astype(np.float32)

    full_qpos = env.sim.data.qpos.copy()
    if task_name in ["door", "nut"]:
        return np.concatenate([full_qpos[:7], [float(_gripper_qpos_to_normalized(full_qpos[7:9]))]])
    if task_name == "wipe":
        return full_qpos[:7]
    return np.concatenate(
        [
            full_qpos[0:7],
            [float(_gripper_qpos_to_normalized(full_qpos[7:9]))],
            full_qpos[9:16],
            [float(_gripper_qpos_to_normalized(full_qpos[16:18]))],
        ]
    )


def is_two_arm_lift_success(env) -> bool:
    """Strict shared Two-arm lift criterion used by generation and evaluation."""
    if not env._check_success():
        return False
    z0 = float(env._handle0_xpos[2])
    z1 = float(env._handle1_xpos[2])
    return abs(z1 - z0) < 0.05


def _state_policy_success(env, task_name: str) -> bool:
    """Apply task-specific success checks for state-conditioned rollouts.

    Most tasks defer to robosuite success; two-arm lift adds a handle-height
    consistency check so partial or uneven grasps are not counted as success.
    """
    if task_name == "two_arm":
        return is_two_arm_lift_success(env)
    return bool(env._check_success())


def _to_action_from_q(q, task_name, action_representation: str = "joint_space", env=None):
    """Pack a compact policy pose into the action vector expected by env.step.

    This centralizes task-specific arm / gripper action layouts for replay and
    rollout code.
    """
    q = np.asarray(q, dtype=np.float32)
    action_representation = validate_action_representation(action_representation)
    if action_representation == "task_space":
        if env is None:
            raise ValueError("Task-space action conversion requires env")
        if task_name in ["door", "nut"]:
            robot = env.robots[0]
            eef_id = list(robot.eef_site_id.values())[0]
            grip_sc = _normalized_gripper_to_action(q[7]) if q.shape[0] >= 8 else 0.0
            return np.concatenate([_eef_action_delta(env, eef_id, q[:7]), [grip_sc]])
        if task_name == "wipe":
            robot = env.robots[0]
            eef_id = list(robot.eef_site_id.values())[0]
            return _eef_action_delta(env, eef_id, q[:7])
        left, right = env.robots[0], env.robots[1]
        eef_l = list(left.eef_site_id.values())[0]
        eef_r = list(right.eef_site_id.values())[0]
        grip_l = _normalized_gripper_to_action(q[7])
        grip_r = _normalized_gripper_to_action(q[15])
        return np.concatenate([
            _eef_action_delta(env, eef_l, q[0:7]),
            [grip_l],
            _eef_action_delta(env, eef_r, q[8:15]),
            [grip_r],
        ])

    if task_name in ["door", "nut"]:
        arm_q = q[:7]
        if q.shape[0] == 8:
            grip_sc = _normalized_gripper_to_action(q[7])
        else:
            grip_sc = float((q[7] + q[8]) / 2.0)
        return np.concatenate([arm_q, [grip_sc]])
    elif task_name == "wipe":
        return q[:7]
    else:  # two_arm
        arm1 = q[0:7]
        if q.shape[0] == 16:
            grip1 = _normalized_gripper_to_action(q[7])
            arm2 = q[8:15]
            grip2 = _normalized_gripper_to_action(q[15])
        else:
            grip1 = float((q[7] + q[8]) / 2.0)
            arm2 = q[9:16]
            grip2 = float((q[16] + q[17]) / 2.0)
        return np.concatenate([arm1, [grip1], arm2, [grip2]])


def _set_robot_qpos_from_policy_pose(env, task_name: str, pose: np.ndarray):
    pose = np.asarray(pose, dtype=np.float32)
    if task_name in ["door", "nut"] and pose.shape[0] == 8:
        robot = env.robots[0]
        arm_names = robot.robot_model.joints
        grip_names = next(iter(robot.gripper.values())).joints
        arm_idx = [env.sim.model.get_joint_qpos_addr(nm) for nm in arm_names]
        grip_idx = [env.sim.model.get_joint_qpos_addr(nm) for nm in grip_names]
        env.sim.data.qpos[arm_idx] = pose[:7]
        env.sim.data.qpos[grip_idx] = _normalized_gripper_to_qpos(pose[7])
        return True
    if task_name == "two_arm" and pose.shape[0] == 16:
        offset = 0
        for robot_i, robot in enumerate(env.robots):
            arm_names = robot.robot_model.joints
            grip_names = next(iter(robot.gripper.values())).joints
            arm_idx = [env.sim.model.get_joint_qpos_addr(nm) for nm in arm_names]
            grip_idx = [env.sim.model.get_joint_qpos_addr(nm) for nm in grip_names]
            arm_start = 0 if robot_i == 0 else 8
            grip_pos = 7 if robot_i == 0 else 15
            env.sim.data.qpos[arm_idx] = pose[arm_start:arm_start + 7]
            env.sim.data.qpos[grip_idx] = _normalized_gripper_to_qpos(pose[grip_pos])
            offset += len(arm_names) + len(grip_names)
        return True
    if task_name == "wipe" and pose.shape[0] == 7:
        robot = env.robots[0]
        arm_idx = [env.sim.model.get_joint_qpos_addr(nm) for nm in robot.robot_model.joints]
        env.sim.data.qpos[arm_idx] = pose[:7]
        return True
    return False


def render_trajectory(
    env,
    task_name: str,
    q_high: np.ndarray,
    initial_pose: np.ndarray,
    fps: int = 60,
    camera_name: str = "frontview",
    hold_init: bool = False,
    set_init: bool = True,
    action_representation: str = "joint_space",
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
        - Compact policy trajectories use one normalized gripper pose per gripper:
          0 closed, 1 open.
        - If you switched to a JOINT_POSITION controller, ensure the action format
          matches (absolute joint targets) instead of OSC deltas.
    """
    
    action_representation = validate_action_representation(action_representation)

    # restore robot + door
    if set_init and action_representation == "joint_space":
        if not _set_robot_qpos_from_policy_pose(env, task_name, initial_pose):
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
        env.sim.data.qvel[:] = 0
        env.sim.forward()

    terminated = False
    if hold_init:
        for _ in range(100):
            _, _, done, _ = env.step(_to_action_from_q(q_high[0], task_name, action_representation, env))
            if done:
                terminated = True
                break

    frames = []
    if terminated:
        img = env.sim.render(640,480, camera_name=camera_name)
        frames.append(np.flipud(img))
        return frames

    for q in q_high:
        action = _to_action_from_q(q, task_name, action_representation, env)
        
        _, _, done, _ = env.step(action)

        img = env.sim.render(640,480, camera_name=camera_name)
        frames.append(np.flipud(img))
        if done:
            break
    return frames
