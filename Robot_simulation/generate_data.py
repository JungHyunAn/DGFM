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
         ├─ joint_angles          (K, dof)        # sampled keyframes (not full trace)
         ├─ key_inds              (K,)            # original indices in the full trace
         ├─ initial_pose          (dof,)          # robot pose at episode start
         ├─ attrs: success=1, num_keyframes=K
         ├─ environment_setting/                # full simulator state to restore
         │    ├─ qpos
         │    ├─ qvel
         │    ├─ body_pos
         │    └─ body_quat
         └─ environment_parameters/values       # task-specific (P,)

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
python -m Robot_simulation.generate_dataset_parallel
  --n 250
  --task_name door                # {door, wipe, two_arm, nut}
  --render                        # (optional) write grid MP4
  --num_workers 8
  --chunk_size 3
  --verbose

Important notes & assumptions
-----------------------------
- The **heuristic generator** must return:
    (q_traj, success, frames_unused, initial_qpos, environment_setting, env_param)
  where `environment_setting` contains full MuJoCo state arrays. Only episodes
  with `success=True` are saved.
- Keyframes are sampled from a potentially downsampled trace but `key_inds`
  are always in the **original** trace index space.
- `make_env(..., use_joint_control=True, environment_setting=...)` restores the
  exact scene for deterministic replay and disables random placement thereafter.
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

from Robot_simulation.heuristics_door import generate_door_trajectory
from Robot_simulation.heuristics_wipe import generate_wipe_trajectory
from Robot_simulation.heuristics_two_arm import generate_two_arm_trajectory
from Robot_simulation.heuristics_nut import generate_nut_trajectory
from Robot_simulation.heuristics_util import (make_env, 
                                              write_grid_video, 
                                              compute_smooth_trajectory, 
                                              render_trajectory)

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

                      "nut"     : [(130, 140, 2), # carrying to peg end
                                   (140, 150, 2), # carrying to peg end
                                   (150, 160, 2), # carrying to peg end
                                   (160, 170, 2), # carrying to peg end
                                   (170, 180, 2), # carrying to peg end
                                   (180, 190, 3), # entering peg
                                   (190, 202, 3), # entering peg
                                   (202, 207, 3), # entering peg
                                   (207, 215, 3), # entering peg
                                   (215, 230, 3)] # entering peg
                    }


def init_hdf5(path: str, task_name: str, env):
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
    meta.attrs["dof"] = env.robots[0].dof
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
                                        - Wipe   : dirt x coordinate, y coordinate, maximum radius
                                        - TwoArm : pot x coordinate, y coordinate, yaw
                                        - Nut    : nut handle x coordinate, y coordinate, yaw)

    Args:
        hf: Open HDF5 file handle.
        ep_idx: Episode id (used in the group name).
        joint_angles: Keyframe joint angles.
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
    grp.create_dataset("initial_pose", data=initial_pose, compression="gzip")
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
        A list of dicts, each containing:
            {
              "joint_angles": (K, dof),
              "key_inds":     (K,),
              "initial_pose": (dof,),
              "environment_setting": dict of arrays
            }
    """

    gen_map = {
        "door": generate_door_trajectory,
        "wipe": generate_wipe_trajectory,
        "two_arm": generate_two_arm_trajectory,
        "nut": generate_nut_trajectory
    }
    generator = gen_map[task_name]
    np.random.seed(seed + worker_id)

    env = make_env(task_name, has_offscreen_renderer=False, use_camera_obs=False)
    successes = []
    trials = 0

    while len(successes) < chunk_size and trials < max_trials:
        trials += 1
        # generator now returns (actions, success, frames, initial_qpos)
        q_traj, success, _, init_qpos, environment_setting, env_param = generator(env, render=False)
        # print(trials, success)
        if success:
            # door_bid = env.object_body_ids["door"]
            # print("Passed door position: ", env.sim.data.body_xpos[door_bid])
            q_keys, idx = sample_keyframes(q_traj, 
                                           downsample_ratio=DOWNSAMPLE_RATIOS[task_name], 
                                           intervals=KEYFRAME_INTERVALS[task_name])
            successes.append({
                "joint_angles": q_keys,         # (35,dof)
                "key_inds":      idx,           # (35,)
                "initial_pose":  init_qpos,
                "environment_setting":  environment_setting,
                "environment_parameters": env_param,
            })

    env.close()
    return successes


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


def generate_data_parallel(
    n: int,
    task_name: str,
    render: bool = False,
    output_dir: str = "Robot_simulation/heuristic_dataset",
    hdf5_name: Optional[str] = None,
    num_workers: int = 4,
    chunk_size: int = 2,
    max_render_videos: int = 25,
    max_trials_per_worker: int = 200,
    base_seed: int = 12345,
    verbose: bool = True
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

    os.makedirs(output_dir, exist_ok=True)

    # Sample env for metadata & fps (no offscreen / no camera obs)
    sample_env = make_env(task_name, has_offscreen_renderer=False, use_camera_obs=False)
    control_freq = sample_env.control_freq

    # HDF5 init
    if hdf5_name is None:
        hdf5_name = f"{task_name}_dataset_{n}.hdf5"
    h5_path = os.path.join(output_dir, hdf5_name)
    hf = init_hdf5(
        h5_path,
        task_name,
        sample_env
    )
    sample_env.close()

    episode_idx = 0
    success_count = 0
    episodes_frames: List[List[np.ndarray]] = []

    if verbose:
        print(f"[generate_data_parallel] Target={n} | Task={task_name} | Workers={num_workers}")

    with ProcessPoolExecutor(max_workers=num_workers) as pool, \
         tqdm(total=n, desc="Successful Trajectories", unit="traj") as pbar:

        # Launch initial workers
        futures = [
            pool.submit(worker_generate, wid, task_name, chunk_size, max_trials_per_worker, base_seed)
            for wid in range(num_workers)
        ]

        # Collect until n successes
        while success_count < n and futures:
            done, futures = wait_first(futures)
            for fut in done:
                batch = fut.result()
                for entry in batch:
                    if success_count >= n:
                        break
                    # Save actions + initial_pose
                    save_episode(
                        hf,
                        episode_idx,
                        entry["joint_angles"],
                        entry["key_inds"],
                        True,
                        entry["initial_pose"],
                        entry["environment_setting"],
                        entry["environment_parameters"]
                    )

                    pbar.update(1)
                    pbar.set_postfix({"episode": episode_idx+1})
                    episode_idx += 1
                    success_count += 1
                # Resubmit worker if more needed
                if success_count < n:
                    wid = np.random.randint(0, num_workers)
                    futures.append(
                        pool.submit(worker_generate, wid, task_name, chunk_size, max_trials_per_worker, base_seed)
                    )

    hf.attrs["num_episodes"] = success_count
    hf.close()

    # Single-process render & grid video
    if render:
        render_episodes = min(success_count, max_render_videos)
        print("Restoring & Rendering saved trajectories . . .")
        for i in range(render_episodes):
            with h5py.File(h5_path, "r") as hf_read:
                q_keys  = hf_read[f"data/entire_episode_{i}/joint_angles"][:]
                init_q  = q_keys[0]
                g = hf_read[f"data/entire_episode_{i}/environment_setting"]
                environment_setting = {k: g[k][()] for k in g.keys()}   # dict of arrays
            # restore the initial pose            
            env_r = make_env(task_name, 
                             has_offscreen_renderer=True, 
                             use_camera_obs=False, 
                             use_joint_control=True, 
                             environment_setting=environment_setting)
            
            q_high = compute_smooth_trajectory(env_r, task_name, q_keys, control_freq)

            frames = render_trajectory(
                env_r, task_name, q_high, init_q,
                fps=control_freq * 3,
                camera_name="frontview"
            )
            episodes_frames.append(frames)
        env_r.close()

        grid_path = os.path.join(output_dir, f"{task_name}_grid_{n}.mp4")
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

    args = parser.parse_args()

    generate_data_parallel(
        n=args.n,
        task_name=args.task_name,
        render=args.render,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        verbose=args.verbose
    )