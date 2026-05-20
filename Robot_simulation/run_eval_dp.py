"""
Train & evaluate a diffusion-based policy on RoboSuite datasets
==============================================================

This script mirrors `Robot_simulation/run_eval.py`, but trains an epsilon-
prediction diffusion policy using the same `VectorField` backbone as the
flow-matching baselines for a fair comparison.
"""

import os
import math
import json
import h5py
import random
import argparse
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)
robosuite_logger.propagate = False
for h in list(robosuite_logger.handlers):
    robosuite_logger.removeHandler(h)

from Robot_simulation.models.VanillaFM_class import VectorField
from Robot_simulation.models.FM_util import _align_handle_to_nut
from Robot_simulation.models.DP_class import train_DP, eval_model_DP
from Robot_simulation.environments.heuristics_util import _get_environment_params, make_env


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_scale=0.05, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _spawn_env_once(task_name: str, seed: int, idx: int,
                    use_vision: bool = False, camera_name: str = "frontview"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    setting, params, vision = None, None, None

    if task_name == "nut":
        for i in range(100):
            env = make_env(task_name, has_offscreen_renderer=use_vision, training=True)
            env.reset()

            setting, params, check_grasp, vision = _align_handle_to_nut(
                env, use_vision=use_vision, camera_name=camera_name
            )
            env.close()

            if check_grasp:
                break
            if i == 99:
                print("Nut environment failed grasping!")
    else:
        env = make_env(task_name, training=True)
        env.reset()

        setting = {
            "qpos": env.sim.data.qpos.copy(),
            "qvel": env.sim.data.qvel.copy(),
            "body_pos": env.sim.model.body_pos.copy(),
            "body_quat": env.sim.model.body_quat.copy(),
        }
        params = np.asarray(_get_environment_params(env, task_name), dtype=np.float32)
        if use_vision:
            vision = env.sim.render(640, 480, camera_name=camera_name)
        else:
            vision = None

        env.close()

    return idx, setting, params, vision


def train_and_eval_DP(
    N: int,
    dataset_path: str,
    task_name: str,
    device,
    results_path: str,
    model_path: str = None,
    n_t: int = 1,
    max_epochs: int = 3000,
    batch_size: int = 200,
    warmup_steps: int = 600,
    val_period: int = 5,
    early_stopping: bool = True,
    stop_criteria: int = 3,
    evaluation_samples: int = 1000,
    seed: int = 2002,
):
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")

    if model_path is None:
        exp_dir = os.path.join(results_path, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
    else:
        exp_dir = results_path

    with h5py.File(dataset_path, "r") as hf:
        data_grp = hf["data"]
        ep_keys = sorted(data_grp.keys(), key=lambda s: int(s.split("_")[-1]))
        total_N = len(ep_keys)

        first = data_grp[ep_keys[0]]
        traj0 = first["joint_angles"][:]
        param0 = first["environment_parameters"]["values"][:]

        seq_len, dof = traj0.shape
        param_len = param0.shape[0]

        data_trajectories = np.empty((total_N, seq_len, dof), dtype=traj0.dtype)
        data_env_params = np.empty((total_N, param_len), dtype=param0.dtype)
        for i, ep in enumerate(ep_keys):
            grp = data_grp[ep]
            data_trajectories[i] = grp["joint_angles"][:]
            data_env_params[i] = grp["environment_parameters"]["values"][:]

        data_trajectories = torch.from_numpy(data_trajectories).float().to(device)
        data_env_params = torch.from_numpy(data_env_params).float().to(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    perm_idx = torch.randperm(N, device=device)
    target_trajectories = data_trajectories[perm_idx]
    env_params = data_env_params[perm_idx]

    gripper_idx = None
    if task_name in ["door", "nut"]:
        gripper_idx = [7]
    elif task_name == "two_arm":
        gripper_idx = [7, 15]

    print("parameter length: ", param_len)
    model = VectorField(seq_len, dof, param_len, gripper_idx=gripper_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_epochs=warmup_steps,
        total_epochs=max_epochs,
    )

    print(f"[{datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}] Starting DP training for {task_name} with {N} samples…")

    training_thread_time_seconds = None

    if model_path is None:
        train_time_start = time.thread_time()
        best_model, last_model, recs = train_DP(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            task_name=task_name,
            target_trajectories=target_trajectories,
            environment_parameters=env_params,
            seq_len=seq_len,
            dof=dof,
            param_len=param_len,
            gripper_idx=gripper_idx,
            n_t=n_t,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            val_period=val_period,
            early_stopping=early_stopping,
            stop_criteria=stop_criteria,
        )
        training_thread_time_seconds = time.thread_time() - train_time_start
        print(f"Training thread time: {training_thread_time_seconds:.3f}s")
    else:
        best_model = VectorField(seq_len, dof, param_len, gripper_idx=gripper_idx).to(device)
        state = torch.load(model_path, map_location=device)
        best_model.load_state_dict(state)
        best_model.eval()
        recs = None

    print(f"Training finished, evaluating for {evaluation_samples} trials . . .")

    env_params_list = [None] * evaluation_samples
    env_settings_all = [None] * evaluation_samples

    env_workers = min(
        evaluation_samples,
        max(1, (os.cpu_count() or 4) - 2),
        int(os.getenv("EVAL_ENV_WORKERS", "10")),
    )

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=env_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_spawn_env_once, task_name, seed + i, i) for i in range(evaluation_samples)]
        for fut in as_completed(futs):
            idx, setting, params, _ = fut.result()
            env_settings_all[idx] = setting
            env_params_list[idx] = params

    eval_params = np.asarray(env_params_list, dtype=np.float32)

    success_rate_best, avg_reward_best = eval_model_DP(
        model=best_model,
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
        base_seed=seed + 1,
    )
    print(f"Success rate : {success_rate_best:.3f}, Average reward : {avg_reward_best:.3f}")

    if model_path is None:
        json_path = os.path.join(exp_dir, "results.json")
        output = {
            "timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "seed": seed,
            "policy_type": "DP",
            "N": N,
            "task_name": task_name,
            "n_t": n_t,
            "maximum epoch": max_epochs,
            "warmup steps": warmup_steps,
            "batch_size": batch_size,
            "early_stopping": early_stopping,
            "stop_criteria": stop_criteria,
            "eval_samples": evaluation_samples,
            "success_rate_best": success_rate_best,
            "average_reward_best": avg_reward_best,
            "training_thread_time_seconds": training_thread_time_seconds,
            "records": recs,
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[saved results to {json_path}]")

        epochs = sorted(recs.keys())
        rates = [recs[e].get("success_rate", float("nan")) for e in epochs]

        fig, ax1 = plt.subplots()
        ln1 = ax1.plot(epochs, rates, linewidth=2, label="success_rate")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Success Rate")
        ax1.set_title("DP Training Curves")

        ax2 = ax1.twinx()
        loss_lines = []
        loss = [recs[e].get("loss") for e in epochs]
        if any(v is not None for v in loss):
            ln2 = ax2.plot(epochs, loss, linestyle="--", label="loss")
            loss_lines += ln2
        ax2.set_ylabel("Loss")

        lines = ln1 + loss_lines
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")

        plot_path = os.path.join(exp_dir, "success_rates.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"[Saved plot to {plot_path}]")

        saved_model_path = os.path.join(exp_dir, "model.pt")
        torch.save(best_model.state_dict(), saved_model_path)
        print(f"[Saved model to {saved_model_path}]")
    else:
        json_path = os.path.join(exp_dir, "results_eval_trained.json")
        output = {
            "timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "success_rate_best": success_rate_best,
            "average_reward_best": avg_reward_best,
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[saved results to {json_path}]")

    return best_model, recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train_and_eval_DP")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True, choices=["door", "wipe", "two_arm", "nut"])
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Base directory under which a timestamped experiment folder will be created.",
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_t", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=600)
    parser.add_argument("--val_period", type=int, default=5)
    parser.add_argument("--stop_criteria", type=int, default=3)
    parser.add_argument("--evaluation_samples", type=int, default=100)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=2002)
    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
    else:
        dev = torch.device("cpu")

    print(f"Running on {dev}\n")

    train_and_eval_DP(
        N=args.N,
        dataset_path=args.dataset_path,
        task_name=args.task_name,
        device=dev,
        results_path=args.results_path,
        model_path=args.model_path,
        n_t=args.n_t,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        val_period=args.val_period,
        early_stopping=args.early_stopping,
        stop_criteria=args.stop_criteria,
        evaluation_samples=args.evaluation_samples,
        seed=args.seed,
    )
