"""
Train & evaluate Flow-Matching (FM) models on RoboSuite datasets
================================================================

This program loads an HDF5 dataset of keyframe joint trajectories and
environment parameters, trains a conditional vector-field model (1D U-Net with
FiLM) using one of three FM variants, evaluates the trained policy in RoboSuite,
and saves artifacts (metrics, plots, videos, and model weights).

What this script does
---------------------
1) **Load dataset** (`--dataset_path`):
   - Expects HDF5 layout produced by the data generator, e.g.:
       /data/entire_episode_{i}/joint_angles                  → (T, dof)
       /data/entire_episode_{i}/environment_parameters/values → (P,)
   - Reads all episodes, then subsamples **N** randomly (seeded).

2) **Build model & optimizer**:
   - Model: `VectorField(seq_len, dof, param_len, gripper_idx)` (1D U-Net + FiLM).
   - Optimizer: Adam(lr=1e-4, weight_decay=1e-6).
   - Scheduler: cosine with linear warm-up (`get_cosine_schedule_with_warmup`).

3) **Train** one of:
   - `UniformFM`: uniform t∈[0,1].
   - `ShiftedFM`: back-loaded t via Beta, emphasizing late times.
   - `DGFM`: Dimension-Guided FM that builds a low-rank mixture over (x, c) and
     mixes **global** and **local** flow phases (requires more args; see below).

4) **Evaluate**:
   - Uses `eval_model` to run many trials in parallel RoboSuite envs and
     compute mean reward and success rate. Optionally renders sample rollouts to
     a grid MP4 in the timestamped results folder.

5) **Save artifacts** under `--results_path/<Asia/Seoul timestamp>/`:
   - `results.json`: full config + metrics across training.
   - `success_rates.png`: success / loss vs. epoch.
   - `model.pt`: best model weights (PyTorch).
   - `*_grid_*.mp4`: optional rendered grids from evaluation.

Command-line arguments
----------------------
Required:
- `--FM_type`          ∈ {UniformFM, ShiftedFM, DGFM}
- `--N`                Number of episodes to draw from the dataset
- `--dataset_path`     Path to HDF5 produced by the data generator
- `--task_name`        ∈ {door, wipe, two_arm, nut}
- `--results_path`     Directory to create a timestamped experiment folder

Common options:
- `--device`           "cuda", "cuda:0", or "cpu" (auto-checked)
- `--batch_size`       Mini-batch size (default: 200)
- `--max_epochs`       Total training epochs (default: 1000)
- `--warmup_steps`     Warm-up epochs for the cosine schedule (default: 200)
- `--val_period`       Validate & log every K epochs (default: 5)
- `--early_stopping`   Enable early stopping (flag)
- `--stop_criteria`    Allowed consecutive non-improvements (default: 3)
- `--evaluation_samples` Number of trials in each evaluation (default: 1000)

FM-specific:
- **UniformFM / ShiftedFM / DGFM**: `--n_t` (per-sample t-replications).
- `--time_sampling` can be `uniform` or `shifted`; shifted uses `--beta_a` / `--beta_b`.
- **DGFM** uses `--interpolation_path` (default: `piecewise-linear-midpoint`).
- DGFM cluster rank / size are auto-set per task from `(seq_len)`; see code.

Task conventions
----------------
- Gripper indices are auto-set:
  - door / nut: [7]
  - two_arm   : [7, 15]
  - wipe      : no gripper
- Shapes:
  - Trajectories tensor: (B, T, dof)
  - Env parameters:      (B, P)

Outputs & folder structure
--------------------------
`<results_path>/<YYYYMMDDTHHMMSS+09:00>/`
  ├── results.json
  ├── success_rates.png
  ├── model.pt
  ├── (optional) door_grid_best.mp4
  └── (optional) door_grid_last.mp4

Example invocations
-------------------
Uniform FM:
  python -m Robot_simulation.run_eval \
    --FM_type UniformFM --N 5000 \
    --dataset_path Robot_simulation/heuristic_dataset/door_dataset_10000.hdf5 \
    --task_name door --results_path Robot_simulation/eval_results \
    --device cuda --val_period 5 --batch_size 250 --n_t 4 \
    --max_epochs 800 --warmup_steps 200

Shifted FM:
  python -m Robot_simulation.run_eval \
    --FM_type ShiftedFM --N 5000 ... (same as above)

DGFM:
  python -m Robot_simulation.run_eval \
    --FM_type DGFM --n_t 4 --interpolation_path piecewise-linear-midpoint \
    --N 5000 ... (paths as above)

Behavioral notes & tips
-----------------------
- The dataset loader assumes **all** episodes share the same (T, dof) and P.
- Random subsampling and training are seeded for reproducibility of
  the episode subset; CUDA nondeterminism may still affect training dynamics.
- Evaluation renders a subset of episodes into grids; tune `render_width` and
  `render_num` inside `eval_model`. (by default 4*4 and successful trials occupy half)
- The Asia/Seoul timestamp is used for folder naming for easier experiment
  bookkeeping.
"""

import os
import numpy as np
import random
import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
from typing import Tuple, List
import json
import h5py
import argparse
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import logging
logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)  
robosuite_logger.propagate = False        
for h in list(robosuite_logger.handlers): 
    robosuite_logger.removeHandler(h)

from Robot_simulation.FM_util import (
    ShiftedFM,
    UniformFM,
    VectorField,
    _align_handle_to_nut,
    _get_environment_params,
    build_state_conditioned_windows,
    eval_model,
)
from Robot_simulation.DGFM_util import DGFM
from Robot_simulation.env_util import make_env
from Robot_simulation import DEFAULT_DATASET_DIR, DEFAULT_RECORDS_DIR


def _load_json_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return config


def _apply_config_defaults(parser: argparse.ArgumentParser, config: dict) -> None:
    valid = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    parser.set_defaults(**config)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_scale=0.05, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = max(0.0, min(1.0, progress))  # clamp to [0,1]
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _spawn_env_once(task_name: str, seed: int, idx: int,
                    use_vision: bool = False, camera_name: str = "frontview"):
    # keep workers single-threaded & headless
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # per-worker seeds for determinism
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    setting, params, vision = None, None, None
    
    if task_name == "nut":
        for i in range(100):
            env = make_env(task_name, has_offscreen_renderer=use_vision)
            env.reset()

            setting, params, check_grasp, vision = _align_handle_to_nut(env, use_vision=use_vision, camera_name=camera_name)
            
            env.close()
            
            if check_grasp:
                break
            if i == 99:
                print("Nut environment failed grasping!")

    else:
        env = make_env(task_name)
        env.reset()

        setting = {
            "qpos":      env.sim.data.qpos.copy(),
            "qvel":      env.sim.data.qvel.copy(),
            "body_pos":  env.sim.model.body_pos.copy(),
            "body_quat": env.sim.model.body_quat.copy(),
        }
        params = np.asarray(_get_environment_params(env, task_name), dtype=np.float32)  # (Dc,)
        if (use_vision):
            vision = env.sim.render(640, 480, camera_name=camera_name)
        else:
            vision = None

        env.close()

    return idx, setting, params, vision
        

def train_and_eval_FM(
    FM_type: str,
    N: int,
    dataset_path: str,
    task_name: str,
    device,
    results_path: str,
    mf: int = 4,
    model_path: str = None,
    n_t: int = None,
    n_t_global: int = None,
    n_t_local: int = None,
    time_sampling: str = "uniform",
    beta_a: float = 1.5,
    beta_b: float = 1.0,
    interpolation_path: str = "piecewise-linear-midpoint",
    max_epochs: int = 1000,
    batch_size: int = 200,
    warmup_steps: int = 200,
    val_period: int = 5,
    early_stopping: bool = True,
    stop_criteria: int = 3,
    evaluation_samples: int = 1000,
    val_trials: int = 5,
    seed: int = 2002,
    horizon: int = 32,
    window_stride: int = 1,
    max_policy_steps: int = 20,
    cluster_partition: int = 5,
    cluster_jaccard_thresh: float = 0.8,
    cluster_merge_k: int = 10,
    cluster_standardize: bool = True,
    cluster_scale_x: float = 1.0,
    cluster_scale_c: float = 1.0,
    cluster_eps: float = 1e-3,
    cluster_outlier_q: float = 0.9,
    max_pca_samples: int = 2000,
    pca_n_jobs: int = -1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    mixture_reg: float = 1e-6,
    mixture_orth_sigma: float = 0.0,
    dgfm_truncated: bool = True,
    dgfm_trunc_low: float = -1.5,
    dgfm_trunc_high: float = 1.5,
):
    flow_class_map = {
        "UniformFM": UniformFM,
        "ShiftedFM": ShiftedFM,
        "DGFM": DGFM,
    }
    if FM_type not in flow_class_map:
        raise ValueError(f"Unsupported FM_type={FM_type}. Expected one of {sorted(flow_class_map)}")

    timestamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime("%Y%m%dT%H%M%S")

    if model_path is None:
        exp_dir = os.path.join(results_path, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
    else:
        exp_dir = results_path

    # load dataset
    with h5py.File(dataset_path, "r") as hf:
        data_grp = hf["data"]
        # collect episode subgroup names in order:
        ep_keys = sorted(data_grp.keys(), key=lambda s: int(s.split("_")[-1]))
        total_N = len(ep_keys)
        selected_ep_keys = ep_keys[:min(N, total_N)]
        if not selected_ep_keys:
            raise ValueError(f"No episodes found in dataset: {dataset_path}")

        # read the first to get shapes/dtypes
        first = data_grp[selected_ep_keys[0]]
        traj0 = first["joint_angles"][:]                         # shape (T, dof)
        param0 = first["environment_parameters"]["values"][:]    # shape (P,)
        dyn0 = first["dynamic_states"][:] if "dynamic_states" in first else np.zeros((traj0.shape[0], 0), dtype=np.float32)

        full_len, dof = traj0.shape
        dyn_dim = dyn0.shape[1]
        seq_len = min(horizon, full_len)
        data_trajectories = []
        data_dynamic = []
        data_static_env = []
        # fill
        for i, ep in enumerate(selected_ep_keys):
            grp = data_grp[ep]
            q_ep = grp["joint_angles"][:]
            data_trajectories.append(q_ep)
            if "dynamic_states" in grp:
                data_dynamic.append(grp["dynamic_states"][:])
            else:
                data_dynamic.append(np.zeros((q_ep.shape[0], dyn_dim), dtype=np.float32))
            data_static_env.append(grp["environment_parameters"]["values"][:])
        data_static_env = np.asarray(data_static_env, dtype=param0.dtype)

        window_traj, window_cond = build_state_conditioned_windows(
            data_trajectories,
            data_dynamic,
            data_static_env,
            horizon=seq_len,
            stride=window_stride,
        )
        data_trajectories = torch.from_numpy(window_traj).float().to(device)
        data_env_params = torch.from_numpy(window_cond).float().to(device)
        param_len = window_cond.shape[1]
        num_demos = len(selected_ep_keys)
        num_windows = window_traj.shape[0]

    # sample the target trajectories & its environment parameters
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    perm_idx = torch.randperm(data_trajectories.shape[0], device=device)
    target_trajectories = data_trajectories[perm_idx]
    env_params = data_env_params[perm_idx]
    train_N = target_trajectories.shape[0]

    # obtain gripper indexes
    gripper_idx = None
    if task_name in ["door", "nut"]:
        gripper_idx = [7]
    elif task_name == "two_arm":
        gripper_idx = [7, 15]

    # define models
    print(
        "[config] "
        f"FM_type={FM_type} | task={task_name} | demos={num_demos} | windows={num_windows} | "
        f"seq_len={seq_len} | dof={dof} | param_len={param_len} | gripper_idx={gripper_idx} | "
        f"horizon_arg={horizon} | window_stride={window_stride} | "
        f"max_policy_steps={max_policy_steps} | cluster_partition={cluster_partition} | "
        f"learning_rate={learning_rate} | weight_decay={weight_decay} | "
        f"max_epochs={max_epochs} | batch_size={batch_size} | warmup_steps={warmup_steps} | "
        f"val_period={val_period} | val_trials={val_trials} | eval_samples={evaluation_samples} | "
        f"n_t={n_t} | time_sampling={time_sampling} | interpolation_path={interpolation_path} | "
        f"seed={seed} | device={device}"
    )
    model = VectorField(seq_len, dof, param_len, gripper_idx=gripper_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_epochs=warmup_steps,
        total_epochs=max_epochs
    )
    flow = flow_class_map[FM_type](
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        task_name=task_name,
        horizon=seq_len,
        dof=dof,
        condition_dim=param_len,
        gripper_idx=gripper_idx,
        time_sampling=time_sampling,
        beta_a=beta_a,
        beta_b=beta_b,
        device=device,
    )

    # train model
    print(
        f"[{datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}] "
        f"Starting {FM_type} training for {task_name} with {num_demos} demos -> {train_N} windows..."
    )
    
    if model_path is None:
        best_model = None
        mixture_sampler = None
        cluster_d = None
        cluster_size = None

        if FM_type == "DGFM":
            if cluster_partition <= 0:
                raise ValueError(f"cluster_partition must be positive, got {cluster_partition}")

            full_dimension = seq_len * dof
            if task_name == "door":
                cluster_d = seq_len * 2 + param_len # end effector stays on 1-dimension path + env_params
            elif task_name == "wipe":
                cluster_d = seq_len * 3 # 75 | end effector stays on 2-dimension path (only x, y movement)
            elif task_name == "two_arm":
                # cluster_d = seq_len * 6 + 3 # 153 | two end effectors stays on 4-dimension path (free x,y,z and z-rotation)
                cluster_d = seq_len * 3 + 3 # 78
            elif task_name == "nut":
                cluster_d = int(seq_len * 3.2)  # 80 | for 10/25=0.4 portion, end effector stays on 4-dimension path (free x,y,z and z-rotation)
                                                #      for the rest 0.6 portion, end effector stays on 1=dimension path
            else:
                cluster_d = None

            cluster_size = max(int(train_N/cluster_partition), full_dimension + 5) if cluster_d is not None else None

            best_model, last_model, recs, mixture_sampler = flow.train(
                target_trajectories=target_trajectories,
                conditions=env_params,
                n_t=n_t,
                cluster_size=cluster_size,
                cluster_d=cluster_d,
                max_epochs=max_epochs,
                batch_size=batch_size,
                interpolation_path=interpolation_path,
                scale_x=cluster_scale_x,
                scale_c=cluster_scale_c,
                cluster_jaccard_thresh=cluster_jaccard_thresh,
                cluster_merge_k=cluster_merge_k,
                cluster_standardize=cluster_standardize,
                cluster_eps=cluster_eps,
                cluster_outlier_q=cluster_outlier_q,
                max_pca_samples=max_pca_samples,
                pca_n_jobs=pca_n_jobs,
                mixture_reg=mixture_reg,
                mixture_orth_sigma=mixture_orth_sigma,
                dgfm_truncated=dgfm_truncated,
                dgfm_trunc_low=dgfm_trunc_low,
                dgfm_trunc_high=dgfm_trunc_high,
                val_period=val_period,
                early_stopping=early_stopping,
                stop_criteria=stop_criteria,
                val_trials=val_trials,
            )
        else:
            best_model, last_model, recs = flow.train(
                target_trajectories=target_trajectories,
                conditions=env_params,
                n_t=n_t,
                max_epochs=max_epochs,
                batch_size=batch_size,
                val_period=val_period,
                early_stopping=early_stopping,
                stop_criteria=stop_criteria,
                val_trials=val_trials,
            )
    else:
        best_model = VectorField(seq_len, dof, param_len, gripper_idx=gripper_idx).to(device)
        state = torch.load(model_path, map_location=device)
        best_model.load_state_dict(state)
        best_model.eval()

    # evaluate model
    print(f"Training finished, evaluating for {evaluation_samples} trials . . .")

    # ====== PARALLEL ENV GENERATION ======
    env_params_list: list[np.ndarray] = [None] * evaluation_samples
    env_settings_all: list[dict]      = [None] * evaluation_samples

    # choose worker count (env creation is CPU-bound)
    ENV_WORKERS = min(
        evaluation_samples,
        max(1, (os.cpu_count() or 4) - 2),
        int(os.getenv("EVAL_ENV_WORKERS", "10"))
    )

    ctx = get_context("spawn")  # safe with MuJoCo/OpenGL
    with ProcessPoolExecutor(max_workers=ENV_WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(_spawn_env_once, task_name, seed + i, i) for i in range(evaluation_samples)]
        for fut in as_completed(futs):
            idx, setting, params, _ = fut.result() # vision not used for training+evaluation
            env_settings_all[idx] = setting
            env_params_list[idx]  = params

    # stack to (N, Dc) float32 (order matches idx)
    eval_params = np.asarray(env_params_list, dtype=np.float32)

    success_rate_best, avg_reward_best = eval_model(model=best_model,
                                                model_class=VectorField,
                                                task_name=task_name,
                                                seq_len=seq_len,
                                                dof=dof,
                                                param_len=param_len,
                                                gripper_idx=gripper_idx,
                                                render_dir=exp_dir,
                                                video_name="best",
                                                val_params=eval_params,
                                                env_settings_all=env_settings_all,
                                                device=device,
                                                trials=evaluation_samples,
                                                render_width=4,
                                                render_num=8,
                                                base_seed=seed+1,
                                                q_low=None,
                                                base_mixture=False,
                                                max_policy_steps=max_policy_steps)
    print(f"Success rate : {success_rate_best:.3f}, Average reward : {avg_reward_best:.3f}")


    # write results    
    if model_path is None:
        json_path = os.path.join(exp_dir, 'results.json')
        output = {
            "timestamp":       datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "seed":            seed,
            "FM_type":         FM_type,
            "N":               N,
            "num_demos":       num_demos,
            "num_windows":     num_windows,
            "horizon":         seq_len,
            "window_stride":   window_stride,
            "max_policy_steps": max_policy_steps,
            "task_name":       task_name,
            "n_t":             n_t,
            "time_sampling":   time_sampling,
            "beta_a":          beta_a,
            "beta_b":          beta_b,
            "interpolation_path": interpolation_path if FM_type == "DGFM" else None,
            "mf":              None,
            "learning_rate":   learning_rate,
            "weight_decay":    weight_decay,
            "cluster_d":       cluster_d if FM_type == "DGFM" else None,
            "cluster_partition": cluster_partition if FM_type == "DGFM" else None,
            "full_dimension":  seq_len * dof if FM_type == "DGFM" else None,
            "cluster_size":    cluster_size if FM_type == "DGFM" else None,
            "cluster_jaccard_thresh": cluster_jaccard_thresh if FM_type == "DGFM" else None,
            "cluster_merge_k": cluster_merge_k if FM_type == "DGFM" else None,
            "cluster_standardize": cluster_standardize if FM_type == "DGFM" else None,
            "cluster_scale_x": cluster_scale_x if FM_type == "DGFM" else None,
            "cluster_scale_c": cluster_scale_c if FM_type == "DGFM" else None,
            "cluster_eps":     cluster_eps if FM_type == "DGFM" else None,
            "cluster_outlier_q": cluster_outlier_q if FM_type == "DGFM" else None,
            "max_pca_samples": max_pca_samples if FM_type == "DGFM" else None,
            "pca_n_jobs":      pca_n_jobs if FM_type == "DGFM" else None,
            "mixture_reg":     mixture_reg if FM_type == "DGFM" else None,
            "mixture_orth_sigma": mixture_orth_sigma if FM_type == "DGFM" else None,
            "dgfm_truncated":  dgfm_truncated if FM_type == "DGFM" else None,
            "dgfm_trunc_low":  dgfm_trunc_low if FM_type == "DGFM" else None,
            "dgfm_trunc_high": dgfm_trunc_high if FM_type == "DGFM" else None,
            "n_t_global":      None,
            "n_t_local":       None,
            "maximum epoch":   max_epochs,
            "warmup steps":    warmup_steps,
            "batch_size":      batch_size,
            "val_period":      val_period,
            "val_trials":      val_trials,
            "early_stopping":  early_stopping,
            "stop_criteria":   stop_criteria,
            "eval_samples":    evaluation_samples,
            "success_rate_best": success_rate_best,
            "average_reward_best": avg_reward_best,
            "records":         recs,
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[saved results to {json_path}]")
        
        # plot and save records
        epochs = sorted(recs.keys())
        if not epochs:
            print("[Skipped training curve plot: no in-training validation records]")
        else:
            rates  = [recs[e].get("success_rate", float("nan")) for e in epochs]

            # first axis for success rate
            fig, ax1 = plt.subplots()
            ln1 = ax1.plot(epochs, rates, linewidth=2, label="success_rate")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Success Rate")
            ax1.set_title(f"{FM_type} Training Curves")

            # second axis for losses
            ax2 = ax1.twinx()
            loss_lines = []
            loss = [recs[e].get("loss") for e in epochs]
            if any(v is not None for v in loss):
                ln2 = ax2.plot(epochs, loss, linestyle="--", label="loss")
                loss_lines += ln2
            ax2.set_ylabel("Loss")

            # combined legend
            lines  = ln1 + loss_lines
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="best")

            plot_path = os.path.join(exp_dir, "success_rates.png")  # keep original filename
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"[Saved plot to {plot_path}]")

        # save model parameters
        model_path = os.path.join(exp_dir, 'model.pt')
        torch.save(best_model.state_dict(), model_path)
        print(f"[Saved model to {model_path}]")
    else:
        recs = None
        json_path = os.path.join(exp_dir, 'results_eval_trained.json')
        output = {
            "timestamp":       datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "success_rate_best": success_rate_best,
            "average_reward_best": avg_reward_best
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[saved results to {json_path}]")

    return best_model, recs


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser("train_and_eval_FM")
    parser.add_argument("--config",         type=str,   default=None,
                        help="JSON config with all run options except dataset_path.")
    parser.add_argument("--FM_type",        type=str,   default=None,
                        choices=["UniformFM","ShiftedFM","DGFM"])
    parser.add_argument("--N",              type=int,   default=None)
    parser.add_argument("--dataset_path",   type=str,   default=None)
    parser.add_argument("--task_name",      type=str,   default=None,
                        choices=["door","wipe","two_arm","nut"])
    parser.add_argument("--results_path",   type=str,   default=DEFAULT_RECORDS_DIR,
                        help="Base directory under which a timestamped experiment folder will be created.")
    parser.add_argument("--mf",             type=int,   default=4)
    parser.add_argument("--model_path",     type=str,   default=None)
    parser.add_argument("--device",         type=str,   default="cuda")
    parser.add_argument("--n_t",            type=int,   default=1)
    parser.add_argument("--n_t_global",     type=int,   default=1)
    parser.add_argument("--n_t_local",      type=int,   default=1)
    parser.add_argument("--time_sampling",  type=str,   default="uniform", choices=["uniform", "shifted"])
    parser.add_argument("--beta_a",         type=float, default=1.5)
    parser.add_argument("--beta_b",         type=float, default=1.0)
    parser.add_argument("--interpolation_path", type=str, default="piecewise-linear-midpoint")
    parser.add_argument("--max_epochs",     type=int,   default=1000)
    parser.add_argument("--batch_size",     type=int,   default=200)
    parser.add_argument("--warmup_steps",    type=int,   default=200)
    parser.add_argument("--val_period",     type=int,   default=5)
    parser.add_argument("--val_trials",     type=int,   default=5)
    parser.add_argument("--stop_criteria",  type=int,   default=3)
    parser.add_argument("--evaluation_samples", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--max_policy_steps", type=int, default=20)
    parser.add_argument("--cluster_partition", type=int, default=5)
    parser.add_argument("--cluster_jaccard_thresh", type=float, default=0.8)
    parser.add_argument("--cluster_merge_k", type=int, default=10)
    parser.add_argument("--cluster_standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster_scale_x", type=float, default=1.0)
    parser.add_argument("--cluster_scale_c", type=float, default=1.0)
    parser.add_argument("--cluster_eps", type=float, default=1e-3)
    parser.add_argument("--cluster_outlier_q", type=float, default=0.9)
    parser.add_argument("--max_pca_samples", type=int, default=2000)
    parser.add_argument("--pca_n_jobs", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--mixture_reg", type=float, default=1e-6)
    parser.add_argument("--mixture_orth_sigma", type=float, default=0.0)
    parser.add_argument("--dgfm_truncated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dgfm_trunc_low", type=float, default=-1.5)
    parser.add_argument("--dgfm_trunc_high", type=float, default=1.5)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=2002)

    _apply_config_defaults(parser, _load_json_config(config_args.config))
    args = parser.parse_args()

    missing = [name for name in ("FM_type", "N", "task_name") if getattr(args, name) is None]
    if missing:
        parser.error(f"Missing required option(s): {', '.join('--' + name for name in missing)}")
    if args.N <= 0:
        parser.error("--N must be positive")

    # device setup
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
    else:
        dev = torch.device("cpu")

    print(f"Running on {dev}\n")

    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = os.path.join(DEFAULT_DATASET_DIR, f"{args.task_name}_dataset_{args.N}.hdf5")

    train_and_eval_FM(
        FM_type            = args.FM_type,
        N                  = args.N,
        dataset_path       = dataset_path,
        task_name          = args.task_name,
        device             = dev,
        results_path       = args.results_path,
        mf                 = args.mf,
        model_path         = args.model_path,
        n_t                = args.n_t,
        n_t_global         = args.n_t_global,
        n_t_local          = args.n_t_local,
        time_sampling      = args.time_sampling,
        beta_a             = args.beta_a,
        beta_b             = args.beta_b,
        interpolation_path = args.interpolation_path,
        max_epochs         = args.max_epochs,
        batch_size         = args.batch_size,
        warmup_steps       = args.warmup_steps,
        val_period         = args.val_period,
        val_trials         = args.val_trials,
        early_stopping     = args.early_stopping,
        stop_criteria      = args.stop_criteria,
        evaluation_samples = args.evaluation_samples,
        seed               = args.seed,
        horizon            = args.horizon,
        window_stride      = args.window_stride,
        max_policy_steps   = args.max_policy_steps,
        cluster_partition  = args.cluster_partition,
        cluster_jaccard_thresh = args.cluster_jaccard_thresh,
        cluster_merge_k    = args.cluster_merge_k,
        cluster_standardize = args.cluster_standardize,
        cluster_scale_x    = args.cluster_scale_x,
        cluster_scale_c    = args.cluster_scale_c,
        cluster_eps        = args.cluster_eps,
        cluster_outlier_q  = args.cluster_outlier_q,
        max_pca_samples    = args.max_pca_samples,
        pca_n_jobs         = args.pca_n_jobs,
        learning_rate      = args.learning_rate,
        weight_decay       = args.weight_decay,
        mixture_reg        = args.mixture_reg,
        mixture_orth_sigma = args.mixture_orth_sigma,
        dgfm_truncated     = args.dgfm_truncated,
        dgfm_trunc_low     = args.dgfm_trunc_low,
        dgfm_trunc_high    = args.dgfm_trunc_high,
    )
