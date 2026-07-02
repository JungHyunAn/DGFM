"""
Heuristic dataset generator for RoboSuite (parallel)
===================================================

This program generates successful demonstration episodes for RoboSuite tasks
using hand-coded heuristics, samples keyframes, and saves everything into a
single HDF5 dataset. It can also optionally replay a subset of the episodes and
export a grid video for quick inspection.

What this script does
---------------------
1) **Environment & heuristics**:
   - Builds a task-specific RoboSuite environment (`make_env`).
   - Calls the task’s heuristic policy to produce a full low-level joint
     trajectory plus a full simulator snapshot:
       door       → `generate_door_trajectory`
       wipe       → `generate_wipe_trajectory`
       two_arm    → `generate_two_arm_trajectory`
       nut        → `generate_nut_trajectory`

2) **Keyframe selection** (`sample_keyframes`):
   - Optionally downsample the full trajectory by a fixed stride
     (`DOWNSAMPLE_RATIOS[task]`), or, for *wipe*, infer the stride so that a
     given interval contributes exactly `n` frames (`-1` sentinel).
   - For each configured interval `(start, end, n)` in **original** indices
     (`KEYFRAME_INTERVALS[task]`), pick `n` uniformly spaced indices and map
     them back to the original trace.
   - Returns `(q_keys, key_inds)` where `q_keys` is (K, dof) and `key_inds` is (K,).

3) **HDF5 writing** (`save_episode`, `init_hdf5`):
   - Creates `/meta` with attributes:
       task, timestamp, robot, dof
   - For each successful episode, creates:
       /data/entire_episode_{i}/
         ├─ joint_angles          (K, dof)        # sampled policy keyframes (joint or task space)
         ├─ key_inds              (K,)            # original indices in the full trace
         ├─ initial_pose          (dof,)          # robot pose at episode start
         ├─ attrs: success=1, num_keyframes=K
         ├─ environment_setting/                # full simulator state to restore
         │    ├─ qpos
         │    ├─ qvel
         │    ├─ body_pos
         │    └─ body_quat
         ├─ environment_parameters/values       # task-specific static params (P,)
         └─ environment_states                      # per-step task dynamics

4) **Parallel collection** (`generate_data_parallel`):
   - Launches `num_workers` processes; each worker repeatedly calls the heuristic
     until it accumulates `chunk_size` successful episodes or hits a trial cap.
   - Continues resubmitting workers until `n` successes have been saved.

5) **Optional rendering**:
   - Recreates the environment from saved `environment_setting`, upsamples
     keyframes to a smooth high-rate trajectory via `compute_smooth_trajectory`,
     replays it with `render_trajectory`, and stitches per-episode videos into a
     grid MP4 (`write_grid_video`).

Task presets
------------
- **Downsample ratios** (`DOWNSAMPLE_RATIOS`):
    door=2, two_arm=2, nut=2, wipe=−1 (auto-stride computed from the first interval)
- **Keyframe intervals** (`KEYFRAME_INTERVALS`):
    Per-task lists of `(start_idx, end_idx, n_samples)` in original indices.
    Use `end = -1` to denote the last frame of the original trace.

CLI
---
python -m Robot_simulation.environments.generate_data
  --n 250
  --task_name door                # {door, wipe, two_arm, nut}
  --render                        # (optional) write grid MP4
  --num_workers 8
  --chunk_size 3
  --verbose

Important notes & assumptions
-----------------------------
- The **heuristic generator** must return:
    (q_traj, success, frames, initial_qpos, environment_setting, env_param, environment_states)
  where `environment_setting` contains full MuJoCo state arrays. Only episodes
  with `success=True` are saved.
- Keyframes are sampled from a potentially downsampled trace but `key_inds`
  are always in the **original** trace index space.
- `make_env(..., action_representation=...)` restores the exact scene for
  deterministic replay and uses joint-position replay only for joint-space data.
- The upsampler (`compute_smooth_trajectory`) handles task-specific gripper
  timing and rates. If you change the env `control_freq`, revisit its arguments.

Outputs
-------
- HDF5 dataset at: `Robot_simulation/heuristic_dataset/{task}_dataset_{n}.hdf5`
- (Optional) grid video: `{task}_grid_{n}.mp4` in the same folder.

Gotchas / tips
--------------
- Ensure the heuristic’s joint ordering matches the RoboSuite model joints.
- If rendering looks unstable at t≈0, verify the replay utilities teleport the
  robot to the first pose and zero velocities before stepping (handled in
  `render_trajectory` in the utilities module).
- For very long episodes, adjust `KEYFRAME_INTERVALS` to cover the parts you
  care about, and/or increase `DOWNSAMPLE_RATIOS`.
"""


import os
import time
import h5py
import imageio.v2 as imageio
import numpy as np
import math
import argparse
import logging
logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)  
robosuite_logger.propagate = False        
for h in list(robosuite_logger.handlers): 
    robosuite_logger.removeHandler(h)

from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from Robot_simulation.environments.heuristics_door import generate_door_trajectory
from Robot_simulation.environments.heuristics_wipe import generate_wipe_trajectory
from Robot_simulation.environments.heuristics_two_arm import TWO_ARM_VISION_CAMERAS, generate_two_arm_trajectory
from Robot_simulation.environments.heuristics_nut import generate_nut_trajectory
from Robot_simulation.env_util import make_env
from Robot_simulation.environments.heuristics_util import (
    compose_task_space_trajectory,
    normalize_policy_trajectory,
    render_trajectory,
    validate_action_representation,
    write_grid_video,
    DEFAULT_VISION_CAMERAS,
    DEFAULT_VISION_HEIGHT,
    DEFAULT_VISION_WIDTH,
)
from Robot_simulation import DEFAULT_DATASET_DIR

DOWNSAMPLE_RATIOS  = {"door"    : 2,
                      "wipe"    : -1,
                      "two_arm" : 2,
                      "nut"     : 2}
KEYFRAME_INTERVALS = {"door"    : [(100, 110, 1), # approaching
                                   (110, 120, 1), # approaching
                                   (120, 130, 1), # approaching
                                   (130, 140, 1), # approaching
                                   (140, 150, 3), # approaching very close 
                                   (150, 153, 2), # hovering on door handle
                                   (160, 210, 8), # turning door handle
                                   (210, 250, 8)],# pulling door

                      "wipe"    : [(60, -1, 25)], # wiping

                      "two_arm" : [(100, 105, 1), # approaching
                                   (105, 110, 1), # approaching
                                   (110, 115, 1), # approaching
                                   (115, 120, 1), # approaching
                                   (120, 123, 2), # approaching                            
                                   (123, 126, 2), # approaching                            
                                   (126, 130, 2), # approaching      
                                   (130, 133, 2), # hovering on handles
                                   (140, -1, 13)],# lifting

                      "nut"     : [(0, 80, 8),    # approaching nut
                                   (80, 120, 4),   # final approach
                                   (120, 132, 2),  # grasping nut
                                   (132, 180, 4),  # carrying to square peg
                                   (180, 230, 3)]  # inserting and releasing
                    }


def _default_camera_names_for_task(task_name: str):
    return TWO_ARM_VISION_CAMERAS if task_name == "two_arm" else DEFAULT_VISION_CAMERAS


def init_hdf5(
    path: str, task_name: str, env, action_representation: str = "joint_space",
    vision: bool = False, camera_names=DEFAULT_VISION_CAMERAS,
    image_height: int = DEFAULT_VISION_HEIGHT, image_width: int = DEFAULT_VISION_WIDTH,
):
    """
    Create and initialize the root HDF5 file for this dataset.

    Args:
        path: Output file path.
        task_name: Name of the task (e.g., "door").
        env: A live robosuite env, used to log robot meta info.

    Returns:
        An open h5py.File handle (caller is responsible for closing it).
    """

    f = h5py.File(path, "w")
    meta = f.create_group("meta")
    meta.attrs["task"] = task_name
    meta.attrs["timestamp"] = time.time()
    meta.attrs["robot"] = env.robots[0].robot_model.naming_prefix
    meta.attrs["robot_dof"] = env.robots[0].dof
    meta.attrs["action_representation"] = action_representation
    meta.attrs["observation_type"] = "vision" if vision else "state"
    meta.attrs["camera_names"] = np.asarray(camera_names, dtype="S")
    meta.attrs["image_height"] = int(image_height)
    meta.attrs["image_width"] = int(image_width)
    return f


def sample_keyframes(
    q_trace: np.ndarray,
    downsample_ratio: int = 1,
    intervals = [
        (100, 150,  3),
        (160, 210, 10),
        (210, 261, 10),
    ]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of joint poses as keyframes.

    Pipeline:
        1) Optionally downsample `q_trace` by taking every `downsample_ratio`-th frame.
        2) For each (start, end, n) interval (specified in ORIGINAL indices), pick
           `n` uniformly spaced indices, map them back to the original trace.

    Args:
        q_trace: (T, dof) full low-level trajectory.
        downsample_ratio: Integer stride for coarse downsampling before picking.
        intervals: List of tuples (start_idx, end_idx, num_samples) in ORIGINAL indices.

    Returns:
        q_keys:   (K, dof) sampled joint poses.
        idx_keys: (K,)     original indices (0-based) into `q_trace`.

    Notes:
        - Indices are clipped to the valid range in case of boundary effects.
        - Adjust `intervals` if your trajectories change length.
    """
    # 1) Downsample
    if downsample_ratio == -1:
        interval = intervals[0]
        downsample_ratio = int((interval[1] - interval[0])/interval[2])
        q_ds = q_trace[::downsample_ratio]
    elif downsample_ratio > 1:
        q_ds = q_trace[::downsample_ratio]
    else:
        q_ds = q_trace

    # 2) Sample from original intervals    
    ds_idxs = []
    for start, end, n in intervals:
        if end == -1:
            end = len(q_trace) - 1
        # Convert original bounds into downsampled indices
        ds_start = math.ceil(start / downsample_ratio)
        ds_end   = math.floor(end  / downsample_ratio)
        # Uniformly sample n points in [ds_start, ds_end]
        ds_idxs.append(
            np.linspace(ds_start, ds_end, n, dtype=int)
        )
    ds_idxs = np.concatenate(ds_idxs)            # (35,) indices into q_ds
    orig_idxs = ds_idxs * downsample_ratio       # back to original q_trace indices
    orig_idxs = np.clip(orig_idxs, 0, len(q_trace)-1)

    # 3) Gather the keyframes
    q_keys = q_ds[ds_idxs]                       # (35, dof)

    return q_keys, orig_idxs


def save_episode(
    hf: h5py.File,
    ep_idx: int,
    joint_angles: np.ndarray,
    key_inds: np.ndarray,
    success: bool,
    initial_pose: np.ndarray,
    env_setting,
    env_param,
    environment_states: Optional[np.ndarray] = None,
    image_paths: Optional[Dict[str, List[str]]] = None,
    action_representation: str = "joint_space",
):
    """
    Write one successful episode into the HDF5 file.

    Structure created:
        data/entire_episode_{ep_idx}/
            - joint_angles         (K, dof)
            - key_inds             (K,)
            - initial_pose         (dof,)
            - attrs: success, num_keyframes
            - environment_setting/ (group of arrays: qpos, qvel, body_pos, body_quat, ...)
            - environment_parameters/ (tuple of parameters for the environment
                                        - Door   : door handle x coordinate, y coordinate, yaw
                                        - Wipe   : none; current dirt center/radius is stored in environment_states
                                        - TwoArm : pot x coordinate, y coordinate, yaw
                                        - Nut    : none; live square-nut xyz/rpy is stored in environment_states)

    Args:
        hf: Open HDF5 file handle.
        ep_idx: Episode id (used in the group name).
        joint_angles: Keyframe policy trajectory values. The legacy dataset name is
            preserved for compatibility; inspect `action_representation` metadata.
        key_inds: Original indices of those keyframes.
        success: Boolean flag from the generator.
        initial_pose: Robot pose at the start of the episode (joint space).
        env_setting: Dict of simulator arrays (full-state snapshot).

    Returns:
        None
    """
    
    grp = hf.create_group(f"data/entire_episode_{ep_idx}")
    # these are arrays, so gzip is fine
    grp.create_dataset("joint_angles", data=joint_angles, compression="gzip")
    grp.create_dataset("key_inds",      data=key_inds,      compression="gzip")
    grp.attrs["success"]       = int(success)
    grp.attrs["num_keyframes"] = joint_angles.shape[0]
    grp.attrs["trajectory_format"] = "policy"
    grp.attrs["action_representation"] = action_representation
    grp.attrs["gripper_format"] = "normalized_pose_0_closed_1_open"
    grp.attrs["dof"] = joint_angles.shape[1]
    grp.create_dataset("initial_pose", data=initial_pose, compression="gzip")
    if environment_states is None:
        environment_states = np.zeros((joint_angles.shape[0], 0), dtype=np.float32)
    environment_state_ds = grp.create_dataset(
        "environment_states", data=environment_states, compression="gzip"
    )
    grp["dynamic_states"] = environment_state_ds
    if image_paths:
        image_grp = grp.create_group("image_paths")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for camera_name, paths in image_paths.items():
            image_grp.create_dataset(camera_name, data=np.asarray(paths, dtype=object), dtype=string_dtype)
    # save environment settings
    env_setting_grp = grp.create_group("environment_setting")
    for k, v in env_setting.items():
        env_setting_grp.create_dataset(k, data=v, compression="gzip")
    # save environment parameters
    env_param_grp = grp.create_group("environment_parameters")
    env_param_grp.create_dataset(
        "values",
        data=np.array(env_param, dtype=np.float32),
        compression="gzip"
    )


def worker_generate(
    worker_id: int,
    task_name: str,
    chunk_size: int,
    max_trials: int,
    seed: int,
    vision: bool = False,
    camera_names=None,
    image_height: int = DEFAULT_VISION_HEIGHT,
    image_width: int = DEFAULT_VISION_WIDTH,
    action_representation: str = "joint_space",
) -> List[Dict[str, Any]]:
    """
    Worker process entrypoint for multiprocessing.

    It repeatedly calls the appropriate heuristic generator until it collects
    `chunk_size` successful trajectories or hits `max_trials`.

    Args:
        worker_id: Integer id of this worker (used to offset RNG seed).
        task_name: One of {"door","wipe","two_arm","nut"}.
        chunk_size: Target number of successful episodes to return.
        max_trials: Hard cap on attempts.
        seed: Base RNG seed; worker_id is added to make streams independent.

    Returns:
        A dict containing the worker attempt count and successful trajectory
        entries:
            {
              "trials": int,
              "successes": [
                {
                  "joint_angles": (K, dof),
                  "key_inds":     (K,),
                  "initial_pose": (dof,),
                  "environment_setting": dict of arrays
                },
                ...
              ]
            }
    """

    gen_map = {
        "door": generate_door_trajectory,
        "wipe": generate_wipe_trajectory,
        "two_arm": generate_two_arm_trajectory,
        "nut": generate_nut_trajectory
    }
    generator = gen_map[task_name]
    action_representation = validate_action_representation(action_representation)
    if camera_names is None:
        camera_names = _default_camera_names_for_task(task_name)
    camera_names = list(camera_names)
    np.random.seed(seed + worker_id)

    env = make_env(
        task_name,
        has_offscreen_renderer=vision,
        use_camera_obs=False,
        camera_names=camera_names,
        camera_heights=[image_height] * len(camera_names),
        camera_widths=[image_width] * len(camera_names),
    )
    successes = []
    trials = 0

    while len(successes) < chunk_size and trials < max_trials:
        trials += 1
        # generator now returns (actions, success, frames, initial_qpos)
        result = generator(
            env, render=vision, save_video=False, camera_names=camera_names,
            image_height=image_height, image_width=image_width,
        )
        q_traj, success, frames, init_qpos, environment_setting, env_param, environment_states = result[:7]
        gripper_pose = None
        eef_traj = None
        if task_name == "wipe":
            eef_traj = result[7] if len(result) > 7 else None
        else:
            gripper_pose = result[7] if len(result) > 7 else None
            eef_traj = result[8] if len(result) > 8 else None
        if action_representation == "task_space":
            if eef_traj is None:
                raise ValueError(f"{task_name} heuristic did not return an EEF pose trace")
            q_policy = compose_task_space_trajectory(task_name, eef_traj, gripper_pose=gripper_pose)
        else:
            q_policy = normalize_policy_trajectory(task_name, q_traj, gripper_pose=gripper_pose)
        print(trials, success)
        if success:
            successes.append({
                "joint_angles": q_policy,
                "key_inds":      np.arange(len(q_policy), dtype=np.int64),
                "initial_pose":  q_policy[0],
                "environment_setting":  environment_setting,
                "environment_parameters": env_param,
                "environment_states": environment_states,
                "camera_frames": frames if vision else None,
            })

    env.close()
    return {
        "trials": trials,
        "successes": successes,
    }


def wait_first(futures):
    """
    Block until at least one Future in `futures` completes.

    Args:
        futures: Iterable of concurrent.futures.Future objects.

    Returns:
        done:    Set of futures that finished.
        pending: List of futures still running.

    Notes:
        - Simple polling loop with a short sleep to avoid busy-wait.
        - Useful to stream results as soon as any worker finishes.
    """

    done = set()
    while not done:
        for f in futures:
            if f.done():
                done.add(f)
        if not done:
            time.sleep(0.01)
    pending = [f for f in futures if f not in done]
    return done, pending


def _write_episode_images(
    h5_path: str, ep_idx: int, frames: Dict[str, List[np.ndarray]], jpeg_quality: int
) -> Dict[str, List[str]]:
    """Write episode RGB frames beside the HDF5 file and return relative paths."""
    image_root = os.path.splitext(h5_path)[0] + "_images"
    h5_dir = os.path.dirname(h5_path) or "."
    paths_by_camera = {}
    for camera_name, camera_frames in frames.items():
        camera_dir = os.path.join(image_root, f"episode_{ep_idx:06d}", camera_name)
        os.makedirs(camera_dir, exist_ok=True)
        camera_paths = []
        for frame_idx, frame in enumerate(camera_frames):
            image_path = os.path.join(camera_dir, f"{frame_idx:06d}.jpg")
            imageio.imwrite(image_path, np.asarray(frame, dtype=np.uint8), quality=jpeg_quality)
            camera_paths.append(os.path.relpath(image_path, h5_dir))
        paths_by_camera[camera_name] = camera_paths
    return paths_by_camera


def generate_data_parallel(
    n: int,
    task_name: str,
    render: bool = False,
    output_dir: str = DEFAULT_DATASET_DIR,
    hdf5_name: Optional[str] = None,
    num_workers: int = 4,
    chunk_size: int = 2,
    max_render_videos: int = 25,
    max_trials_per_worker: int = 200,
    base_seed: int = 12345,
    verbose: bool = True,
    vision: bool = False,
    camera_names=DEFAULT_VISION_CAMERAS,
    image_height: int = DEFAULT_VISION_HEIGHT,
    image_width: int = DEFAULT_VISION_WIDTH,
    jpeg_quality: int = 80,
    action_representation: str = "joint_space",
):
    """
    Orchestrate parallel trajectory generation, HDF5 serialization, and optional rendering.

    Steps:
        1) Spawn `num_workers` processes; each returns up to `chunk_size` successes.
        2) Keep submitting workers until `n` successful episodes are saved.
        3) Store per-episode data (joint keys, env snapshot, etc.) into the HDF5.
        4) Optionally, re-render a subset of episodes and stitch a grid video.

    Args:
        n: Total number of successful episodes to collect.
        task_name: Task name (must match the heuristic generator & env builder).
        render: If True, replay & render up to `max_render_videos` episodes.
        output_dir: Directory to save the HDF5 and video.
        hdf5_name: Custom filename; defaults to f"{task_name}_dataset.hdf5".
        num_workers: Number of parallel worker processes.
        chunk_size: How many successes each worker tries to return per batch.
        max_render_videos: Cap on episodes to render into the grid video.
        max_trials_per_worker: Per-worker attempt limit to avoid infinite loops.
        base_seed: Base RNG seed (worker_id is added).
        verbose: Print progress info.

    Returns:
        Path to the saved HDF5 file.

    Side effects:
        - Creates output_dir if missing.
        - Writes a grid MP4 if `render=True`.

    Caveats:
        - The replay step uses `control_freq/15` etc. Double-check those rates if
          your spline or controller assumptions change.
        - Ensure your heuristic generator returns a full env snapshot for
          deterministic replay.
    """

    action_representation = validate_action_representation(action_representation)
    if camera_names is None:
        camera_names = _default_camera_names_for_task(task_name)
    camera_names = list(camera_names)
    os.makedirs(output_dir, exist_ok=True)

    # Sample env for metadata & fps (no offscreen / no camera obs)
    sample_env = make_env(
        task_name, has_offscreen_renderer=False, use_camera_obs=False,
        camera_names=camera_names, camera_heights=[image_height] * len(camera_names),
        camera_widths=[image_width] * len(camera_names),
    )
    control_freq = sample_env.control_freq

    # HDF5 init
    if hdf5_name is None:
        hdf5_name = f"{task_name}_{action_representation}_dataset_{n}{'_vision' if vision else ''}.hdf5"
    h5_path = os.path.join(output_dir, hdf5_name)
    hf = init_hdf5(
        h5_path,
        task_name,
        sample_env,
        action_representation=action_representation,
        vision=vision, camera_names=camera_names,
        image_height=image_height, image_width=image_width,
    )
    sample_env.close()

    episode_idx = 0
    success_count = 0
    heuristic_trials = 0
    heuristic_successes = 0
    episodes_frames: List[List[np.ndarray]] = []

    if verbose:
        print(f"[generate_data_parallel] Target={n} | Task={task_name} | Workers={num_workers}")

    with ProcessPoolExecutor(max_workers=num_workers) as pool, \
         tqdm(total=n, desc="Successful Trajectories", unit="traj") as pbar:

        # Launch initial workers
        futures = [
            pool.submit(
                worker_generate,
                wid,
                task_name,
                chunk_size,
                max_trials_per_worker,
                base_seed,
                vision,
                camera_names,
                image_height,
                image_width,
                action_representation,
            )
            for wid in range(num_workers)
        ]

        # Collect until n successes
        while success_count < n and futures:
            done, futures = wait_first(futures)
            for fut in done:
                worker_result = fut.result()
                batch = worker_result["successes"]
                heuristic_trials += worker_result["trials"]
                heuristic_successes += len(batch)
                for entry in batch:
                    if success_count >= n:
                        break
                    if success_count == 0:
                        hf["meta"].attrs["policy_dof"] = entry["joint_angles"].shape[1]
                        hf["meta"].attrs["gripper_format"] = "normalized_pose_0_closed_1_open"
                        hf["meta"].attrs["action_representation"] = action_representation
                    image_paths = None
                    if vision:
                        frames = entry["camera_frames"]
                        expected_len = len(entry["joint_angles"])
                        bad = {name: len(values) for name, values in frames.items() if len(values) != expected_len}
                        if bad:
                            raise ValueError(
                                f"Episode {episode_idx} image/trajectory length mismatch: "
                                f"expected {expected_len}, got {bad}"
                            )
                        image_paths = _write_episode_images(
                            h5_path, episode_idx, frames, jpeg_quality
                        )
                    # Save actions + initial_pose
                    save_episode(
                        hf,
                        episode_idx,
                        entry["joint_angles"],
                        entry["key_inds"],
                        True,
                        entry["initial_pose"],
                        entry["environment_setting"],
                        entry["environment_parameters"],
                        entry["environment_states"],
                        image_paths,
                        action_representation=action_representation,
                    )

                    pbar.update(1)
                    pbar.set_postfix({"episode": episode_idx+1})
                    episode_idx += 1
                    success_count += 1
                # Resubmit worker if more needed
                if success_count < n:
                    wid = np.random.randint(0, num_workers)
                    futures.append(
                        pool.submit(
                            worker_generate,
                            wid,
                            task_name,
                            chunk_size,
                            max_trials_per_worker,
                            base_seed,
                            vision,
                            camera_names,
                            image_height,
                            image_width,
                            action_representation,
                        )
                    )

    hf.attrs["num_episodes"] = success_count
    hf.close()

    if heuristic_trials:
        heuristic_success_rate = heuristic_successes / heuristic_trials
        print(
            "Heuristic trajectory success rate: "
            f"{heuristic_success_rate:.2%} "
            f"({heuristic_successes}/{heuristic_trials})"
        )

    # Single-process render & grid video
    if render:
        render_episodes = min(success_count, max_render_videos)
        print("Restoring & Rendering saved trajectories . . .")
        for i in range(render_episodes):
            with h5py.File(h5_path, "r") as hf_read:
                q_traj  = hf_read[f"data/entire_episode_{i}/joint_angles"][:]
                init_q  = q_traj[0]
                g = hf_read[f"data/entire_episode_{i}/environment_setting"]
                environment_setting = {k: g[k][()] for k in g.keys()}   # dict of arrays
            # restore the initial pose
            env_r = make_env(task_name,
                             has_offscreen_renderer=True,
                             use_camera_obs=False,
                             use_joint_control=(action_representation == "joint_space"),
                             environment_setting=environment_setting,
                             action_representation=action_representation)

            frames = render_trajectory(
                env_r, task_name, q_traj, init_q,
                fps=control_freq * 3,
                camera_name="frontview",
                action_representation=action_representation,
                set_init=(action_representation == "joint_space"),
            )
            episodes_frames.append(frames)
            env_r.close()

        grid_path = os.path.join(output_dir, f"{task_name}_{action_representation}_grid_{n}.mp4")
        write_grid_video(episodes_frames, grid_path, grid_shape=(5,5), fps=control_freq)
        if verbose:
            print(f"Saved grid video: {grid_path}")

    if verbose:
        print(f"Saved dataset: {h5_path} ({success_count} episodes)")

    return h5_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate heuristic data for robosuite tasks in parallel."
    )
    parser.add_argument(
        "--n", type=int, default=25,
        help="Total number of successful episodes to collect"
    )
    parser.add_argument(
        "--task_name", type=str, default="door",
        choices=["door", "wipe", "two_arm", "nut"],
        help="Name of the task to generate data for"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Whether to replay and render videos of the collected episodes"
    )
    parser.add_argument(
        "--vision", action=argparse.BooleanOptionalAction, default=False,
        help="Save external camera images and image paths for vision-conditioned training"
    )
    parser.add_argument(
        "--camera_names", nargs="+", default=None,
        help="Camera views to save when --vision is enabled"
    )
    parser.add_argument("--image_height", type=int, default=DEFAULT_VISION_HEIGHT)
    parser.add_argument("--image_width", type=int, default=DEFAULT_VISION_WIDTH)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_DATASET_DIR,
        help="Directory to save generated HDF5 datasets"
    )
    parser.add_argument(
        "--num_workers", type=int, default=5,
        help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=3,
        help="Number of successes each worker returns per batch"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress and debug information"
    )
    parser.add_argument(
        "--action_representation", type=str, default="joint_space",
        choices=["joint_space", "task_space"],
        help="Policy trajectory representation to save"
    )

    args = parser.parse_args()

    generate_data_parallel(
        n=args.n,
        task_name=args.task_name,
        render=args.render,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
        vision=args.vision,
        camera_names=args.camera_names,
        image_height=args.image_height,
        image_width=args.image_width,
        jpeg_quality=args.jpeg_quality,
        action_representation=args.action_representation,
    )
