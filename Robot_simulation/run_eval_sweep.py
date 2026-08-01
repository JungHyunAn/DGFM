"""Run fixed train/eval sweeps for Robot_simulation policies.

The sweep intentionally exposes only task_name and seed. Per-task demo sizes,
training epochs, shared training/evaluation options, and method-specific
options live in dictionaries below so experiment settings stay easy to audit.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

import h5py
from pathlib import Path
from typing import Any

from Robot_simulation.reproducibility import (
    DP_EVAL_POLICY_SEED_SCHEME, DP_EVAL_SAMPLING_MODE, TASK_EVAL_BASE_SEEDS, dataset_fingerprint,
    selected_episode_indices, stable_hash,
)


METHODS = ("UniformFM", "DP", "DGFMv2")
FINAL_TASKS = ("door", "two_arm")
FINAL_TRAINING_SEEDS = (1000, 2000, 3000)
EXPERIMENT_VERSION = "clean_v1"
PREFERRED_DGFM_ROOT = Path("/PublicHDD/ajh916/DGFM")
FALLBACK_DGFM_ROOT = Path(__file__).resolve().parents[1]


def dgfm_path(relative_path: str) -> str:
    root = PREFERRED_DGFM_ROOT if PREFERRED_DGFM_ROOT.exists() else FALLBACK_DGFM_ROOT
    return str(root / relative_path)


DEMO_SIZES_BY_TASK = {
    "door": (20, 40, 80),
    "wipe": (),
    "two_arm": (20, 40, 80),
    "nut": (20, 40, 80),
}

MAX_EPOCHS_BY_TASK = {
    "door": (4000, 2000, 1000),
    "wipe": (),
    "two_arm": (4000, 2000, 1000),
    "nut": (4000, 2000, 1000),
}

VAL_PERIODS_BY_TASK = {
    "door": (80, 40, 20),
    "wipe": (),
    "two_arm": (80, 40, 20),
    "nut": (80, 40, 20),
}

CLUSTER_PARTITIONS_BY_TASK = {
    "door": (5, 5, 5),
    "wipe": (),
    "two_arm": (10, 10, 10),
    "nut": (5, 5, 5),
}

DATASET_PATH_BY_TASK = {
    "door": dgfm_path("Robot_simulation/heuristic_dataset_clean_v1/door_joint_space_dataset_1000_vision.hdf5"),
    "wipe": None,
    "two_arm": dgfm_path("Robot_simulation/heuristic_dataset_clean_v1/two_arm_joint_space_dataset_1000_vision_strict_v1.hdf5"),
    "nut": dgfm_path("Robot_simulation/heuristic_dataset_clean_v1/nut_joint_space_dataset_1000_vision.hdf5"),
}

CAMERA_NAMES_BY_TASK = {
    "door": ["frontview", "robot0_eye_in_hand"],
    "wipe": None,
    "two_arm": ["frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
    "nut": ["frontview", "robot0_eye_in_hand"],
}

SLEEP_SECONDS_BETWEEN_RUNS = 500


SHARED_CONFIG: dict[str, Any] = {
    "use_ema": True,
    "results_path": dgfm_path("Robot_simulation/eval_results_clean_v1/{task_name}/sweep_{seed}"),
    "device": "cuda",
    "n_t": 1,
    "learning_rate": 0.0001,
    "weight_decay": 1e-6,
    "batch_size": 500,
    "val_trials": 50,
    "stop_criteria": 3,
    "evaluation_samples": 100,
    "horizon": 16,
    "executed_horizon": 8,
    "window_stride": 1,
    "recorded_control_freq": 20,
    "trajectory_control_freq": 10,
    "max_policy_steps": 40,
    "observation_horizon": 1,
    "observation_type": "vision",
    "vision_batch_size": 500,
    "condition_embed_dim": 256,
    "vision_finetune": True,
    "vision_finetune_mode": "layer4",
    "vision_train_bn": False,
    "vision_pool": "spatial_softmax",
    "vision_spatial_softmax_temperature": 0.5,
    "vision_feature_proj_dim": 256,
    "vision_feature_norm": "layernorm",
    "vision_aug": False,
    "vision_random_shift": 4,
    "vision_color_jitter": 0.1,
    "vision_encoder_lr_scale": 0.1,
    "vision_projection_lr_scale": 1.0,
    "vision_warmup_freeze_epochs": 0,
    "early_stopping": False,
    "normalize_data": True,
    "eval_fresh": False,
}

METHOD_CONFIGS: dict[str, dict[str, Any]] = {
    "UniformFM": {
        "FM_type": "UniformFM",
        "time_sampling": "uniform",
    },
    "DP": {
        "FM_type": "DP",
        "dp_T_diff": 100,
        "dp_schedule_type": "cosine",
        "dp_ddim_steps": None,
        "dp_eta": 1.0,
        "dp_pred_type": "epsilon",
        "dp_clip_sample": True,
        "dp_clip_sample_range": 1.0,
    },
    "DGFMv2": {
        "FM_type": "DGFMv2",
        "cluster_jaccard_thresh": 0.8,
        "cluster_merge_k": 10,
        "cluster_standardize": True,
        "cluster_scale_x": 1.0,
        "cluster_scale_c": 1.0,
        "cluster_eps": 0.001,
        "cluster_outlier_q": 0.9,
        "max_pca_samples": 2000,
        "pca_n_jobs": -1,
        "mixture_reg": 1e-6,
        "mixture_orth_sigma": 0.0,
        "dgfm_truncated": True,
        "dgfm_trunc_low": -1.5,
        "dgfm_trunc_high": 1.5,
        "time_sampling": "uniform",
        "interpolation_path": "beizer",
        "residual_lambda": 0.2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "run_eval_sweep",
        description="Sweep UniformFM, DP, and DGFMv2 for a task and seed.",
    )
    parser.add_argument("task_name", choices=sorted(DEMO_SIZES_BY_TASK))
    parser.add_argument("seed", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Methods to run. Defaults to all methods.",
    )
    parser.add_argument("--residual_lambda", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", help="Skip completed runs in the sweep results folder.")
    parser.add_argument("--validation_backend", choices=["local", "remote", "none"], default=None)
    parser.add_argument("--remote_eval_config", type=str, default=None)
    parser.add_argument("--remote_eval_mode", choices=["direct", "queue"], default=None)
    parser.add_argument("--remote_eval_render_best", action="store_true")
    parser.add_argument("--remote_eval_timeout_sec", type=int, default=None)
    parser.add_argument("--remote_eval_poll_interval_sec", type=float, default=None)
    args = parser.parse_args()
    if args.validation_backend is None and args.remote_eval_config is not None:
        args.validation_backend = "remote"
    if args.validation_backend == "remote" and args.remote_eval_config is None:
        parser.error("--remote_eval_config is required when --validation_backend=remote")
    return args


def build_config(
    task_name: str,
    seed: int,
    method: str,
    num_demos: int,
    max_epochs: int,
    val_period: int,
    cluster_partition: int,
    residual_lambda: float = 0.2,
) -> dict[str, Any]:
    config = {
        **SHARED_CONFIG,
        **METHOD_CONFIGS[method],
        "N": num_demos,
        "task_name": task_name,
        "dataset_path": DATASET_PATH_BY_TASK[task_name],
        "results_path": SHARED_CONFIG["results_path"].format(task_name=task_name, seed=seed),
        "camera_names": CAMERA_NAMES_BY_TASK[task_name],
        "max_epochs": max_epochs,
        "warmup_steps": int(max_epochs * 0.2),
        "val_period": val_period,
        "seed": seed,
        "eval_base_seed": TASK_EVAL_BASE_SEEDS[task_name],
        "experiment_version": EXPERIMENT_VERSION,
    }
    if method == "DGFMv2":
        config["cluster_partition"] = cluster_partition
        config["residual_lambda"] = residual_lambda
    return config


def build_final_sweep() -> list[dict[str, Any]]:
    """Return the intended 2 tasks x 4 budgets x 3 methods x 3 seeds payload."""
    return [
        config
        for task_name in FINAL_TASKS
        for seed in FINAL_TRAINING_SEEDS
        for config in build_sweep(task_name, seed)
    ]


def build_sweep(
    task_name: str,
    seed: int,
    methods: list[str] | tuple[str, ...] = METHODS,
    residual_lambda: float = 0.2,
) -> list[dict[str, Any]]:
    demo_sizes = DEMO_SIZES_BY_TASK[task_name]
    max_epochs = MAX_EPOCHS_BY_TASK[task_name]
    val_periods = VAL_PERIODS_BY_TASK[task_name]
    cluster_partitions = CLUSTER_PARTITIONS_BY_TASK[task_name]
    if not demo_sizes or not max_epochs:
        raise ValueError(f"No sweep settings are configured for task_name={task_name!r}.")
    if (
        len(demo_sizes) != len(max_epochs)
        or len(demo_sizes) != len(val_periods)
        or len(demo_sizes) != len(cluster_partitions)
    ):
        raise ValueError(
            f"Demo-size, epoch, val-period, and cluster-partition schedules disagree for "
            f"task_name={task_name!r}: {len(demo_sizes)}, {len(max_epochs)}, "
            f"{len(val_periods)}, {len(cluster_partitions)}"
        )

    return [
        build_config(
            task_name,
            seed,
            method,
            num_demos,
            epochs,
            val_period,
            cluster_partition,
            residual_lambda,
        )
        for num_demos, epochs, val_period, cluster_partition in zip(
            demo_sizes, max_epochs, val_periods, cluster_partitions, strict=True
        )
        for method in methods
    ]


BOOLEAN_OPTIONAL_KEYS = {
    "use_ema",
    "vision_finetune",
    "vision_train_bn",
    "vision_aug",
    "cluster_standardize",
    "dgfm_truncated",
    "dp_clip_sample",
    "normalize_data",
    "eval_fresh",
    "remote_eval_render_best",
}


def apply_validation_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    if args.validation_backend is not None:
        config["validation_backend"] = args.validation_backend
    if args.remote_eval_config is not None:
        config["remote_eval_config"] = args.remote_eval_config
    if args.remote_eval_mode is not None:
        config["remote_eval_mode"] = args.remote_eval_mode
    if args.remote_eval_render_best:
        config["remote_eval_render_best"] = True
    if args.remote_eval_timeout_sec is not None:
        config["remote_eval_timeout_sec"] = args.remote_eval_timeout_sec
    if args.remote_eval_poll_interval_sec is not None:
        config["remote_eval_poll_interval_sec"] = args.remote_eval_poll_interval_sec
    return config


def config_to_cli_args(config: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if key == "early_stopping":
                if value:
                    args.append(flag)
            elif key in BOOLEAN_OPTIONAL_KEYS:
                args.append(flag if value else f"--no-{key}")
            else:
                raise ValueError(f"Boolean config key does not have CLI handling: {key}")
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(item) for item in value)
        else:
            args.extend([flag, str(value)])
    return args


def method_metadata_compatible(result: dict[str, Any], config: dict[str, Any]) -> bool:
    """Apply only metadata requirements relevant to the selected method/task."""
    method = config["FM_type"]
    task_name = config["task_name"]

    if config.get("use_ema"):
        if result.get("use_ema") is not True:
            return False
        ema_applied = result.get("ema_applied")
        if ema_applied is False:
            return False
        # Legacy clean_v1 UniformFM is provably EMA-backed by its existing
        # implementation and use_ema signature, so the new field is optional.
        if ema_applied is None and method != "UniformFM":
            return False

    if method == "DP":
        if result.get("dp_eta") != 1.0:
            return False
        if result.get("eval_policy_rng_isolated") is not True:
            return False
        if result.get("eval_policy_seed_scheme") != DP_EVAL_POLICY_SEED_SCHEME:
            return False
        if result.get("dp_eval_sampling_mode") != DP_EVAL_SAMPLING_MODE:
            return False

    if method == "DGFMv2":
        if result.get("ema_applied") is not True:
            return False
        if result.get("pca_covariance_mode") != "full_rank_regularized":
            return False
        if result.get("pca_rank_truncation") is not False:
            return False

    if task_name == "two_arm" and result.get("success_criterion") != "two_arm_lift_strict_v1":
        return False
    return True


def completed_result_path(config: dict[str, Any]) -> Path | None:
    results_path = Path(config["results_path"])
    if not results_path.exists():
        return None

    for result_path in sorted(results_path.glob("*/results.json")):
        try:
            with result_path.open("r") as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(result, dict):
            continue
        if result.get("experiment_version") != EXPERIMENT_VERSION:
            continue
        if not method_metadata_compatible(result, config):
            continue
        if result.get("sweep_config_sha256") != config.get("sweep_config_sha256"):
            continue
        if not isinstance(result.get("run_signature"), dict) or not result.get("run_signature_sha256"):
            continue
        if stable_hash(result["run_signature"]) != result["run_signature_sha256"]:
            continue
        expected_dataset = dataset_fingerprint(config["dataset_path"])
        if result.get("dataset_sha256") != expected_dataset["dataset_sha256"]:
            continue
        with h5py.File(config["dataset_path"], "r") as dataset:
            expected_indices = selected_episode_indices(
                len(dataset["data"]), config["seed"], config["N"]
            )
        if result.get("selected_episode_indices") != expected_indices:
            continue
        if result.get("model_type") != config["FM_type"]:
            continue
        if result.get("task_name") != config["task_name"]:
            continue
        if result.get("N") != config["N"]:
            continue
        if result.get("seed") != config["seed"]:
            continue
        if result.get("maximum epoch") != config["max_epochs"]:
            continue
        if result.get("warmup steps") != config["warmup_steps"]:
            continue
        if config["FM_type"] == "DGFMv2":
            if result.get("cluster_partition") != config["cluster_partition"]:
                continue
            if result.get("residual_lambda", 0.2) != config["residual_lambda"]:
                continue
        return result_path
    return None


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sweep = build_sweep(
        args.task_name,
        args.seed,
        args.methods,
        residual_lambda=args.residual_lambda,
    )
    sweep = [apply_validation_overrides(config, args) for config in sweep]
    for config in sweep:
        config["sweep_config_sha256"] = stable_hash(config)
    sweep_results_path = Path(sweep[0]["results_path"])
    sweep_results_path.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] Prepared {len(sweep)} runs. Results path: {sweep_results_path}")
    if args.resume:
        print("[sweep] Resume enabled: completed matching runs will be skipped.")
    if args.validation_backend is not None:
        print(f"[sweep] validation_backend={args.validation_backend}")

    for run_idx, config in enumerate(sweep, start=1):
        completed_path = completed_result_path(config) if args.resume else None
        if completed_path is not None:
            print(
                f"[sweep] ({run_idx}/{len(sweep)}) skipping completed "
                f"{config['FM_type']} task={config['task_name']} N={config['N']} "
                f"seed={config['seed']} -> {completed_path}"
            )
            continue

        print(
            f"[sweep] ({run_idx}/{len(sweep)}) running "
            f"{config['FM_type']} task={config['task_name']} N={config['N']} "
            f"epochs={config['max_epochs']} warmup={config['warmup_steps']} "
            f"seed={config['seed']}"
        )
        command = [sys.executable, "-m", "Robot_simulation.run_eval", *config_to_cli_args(config)]
        env = os.environ.copy()
        if args.task_name == "two_arm":
            env["VISION_EVAL_WORKERS"] = "5"
        try:
            subprocess.run(command, cwd=repo_root, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            print(
                f"[sweep] Run failed with exit code {exc.returncode}: "
                f"{config['FM_type']} task={config['task_name']} N={config['N']} "
                f"seed={config['seed']}",
                file=sys.stderr,
            )
            print(
                "[sweep] Re-run this sweep with --resume after fixing the underlying "
                "run_eval.py error.",
                file=sys.stderr,
            )
            print(f"[sweep] Command: {shlex.join(command)}", file=sys.stderr)
            raise SystemExit(exc.returncode) from exc

        if run_idx < len(sweep):
            print(f"[sweep] Sleeping {SLEEP_SECONDS_BETWEEN_RUNS} seconds before the next run.")
            time.sleep(SLEEP_SECONDS_BETWEEN_RUNS)


if __name__ == "__main__":
    main()
