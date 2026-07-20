"""Run the fixed Synthetic_data evaluation sweep through run_eval.py."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SAMPLE_SIZES = (320, 640, 1280, 2560, 5120)
CLUSTER_NUMS = (8, 16, 32, 64, 128)
TARGET_DISTRIBUTIONS = ("6", "7", "8")
TARGET_DISTRIBUTION_NAMES = {
    "6": "SwissRoll",
    "7": "TwoMoon",
    "8": "PinWheel",
}
TRIALS = 5

BASE_CONFIG = {
    "ambient_dim": 40,
    "latent_dim": 10,
    "total_steps": 1000,
    "batch_size": 50,
    "batch_num": 8,
    "total_n_t": 1,
    "global_n_t": 1,
    "local_n_t": 1,
    "noise_std": 1e-4,
    "truncation": 1.5,
    "int_inject": 0.5,
    "interpolation_path": "residual-cosine-midpoint",
    "residual_lambda": 0.4,
    "early_stopping": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, nargs="?", default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Reuse a specific sweep timestamp, primarily with --resume.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--results_path",
        type=Path,
        default=None,
        help="Defaults to Synthetic_data/eval_results/sweep_<seed>.",
    )
    return parser.parse_args()


def build_sweep(
    seed: int,
    results_root: Path,
    timestamp: str | None = None,
) -> list[dict]:
    if len(SAMPLE_SIZES) != len(CLUSTER_NUMS):
        raise ValueError("SAMPLE_SIZES and CLUSTER_NUMS must have matching lengths")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return [
        {
            **BASE_CONFIG,
            "sample_size": sample_size,
            "cluster_num": cluster_num,
            "target_distribution": distribution,
            "seed": seed,
            "trials": TRIALS,
            "sweep_seed": seed,
            "aggregate_file": results_root
            / f"{TARGET_DISTRIBUTION_NAMES[distribution]}_{timestamp}.json",
        }
        for distribution in TARGET_DISTRIBUTIONS
        for sample_size, cluster_num in zip(SAMPLE_SIZES, CLUSTER_NUMS, strict=True)
    ]


def config_to_cli(config: dict) -> list[str]:
    cli = []
    for key, value in config.items():
        if isinstance(value, bool):
            cli.append(f"--{key}" if value else f"--no-{key}")
        else:
            cli.extend((f"--{key}", str(value)))
    return cli


def is_complete(config: dict) -> bool:
    output_path = Path(config["aggregate_file"])
    try:
        with output_path.open() as source:
            result = json.load(source)
    except (OSError, json.JSONDecodeError):
        return False
    sample_result = result.get("sample_sizes", {}).get(str(config["sample_size"]), {})
    trials = sample_result.get("trials", {})
    return (
        result.get("seed") == config["sweep_seed"]
        and all(result.get("config", {}).get(key) == value for key, value in BASE_CONFIG.items())
        and sample_result.get("cluster_num") == config["cluster_num"]
        and all(
            trials.get(str(trial_idx), {}).get("seed") == config["seed"] + trial_idx
            and set(trials.get(str(trial_idx), {}).get("results", {}))
            == {"UniformFM", "DGFMv2"}
            for trial_idx in range(config["trials"])
        )
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    results_root = args.results_path or Path(
        f"Synthetic_data/eval_results/sweep_{args.seed}"
    )
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    timestamp = args.timestamp
    if timestamp is None and args.resume:
        prior_files = sorted(results_root.glob("SwissRoll_*.json"))
        if prior_files:
            timestamp = prior_files[-1].stem.removeprefix("SwissRoll_")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    sweep = build_sweep(args.seed, results_root, timestamp)
    print(
        f"[sweep] Prepared {len(sweep)} runs under {results_root} "
        f"(timestamp={timestamp})"
    )

    for run_idx, config in enumerate(sweep, start=1):
        if args.resume and is_complete(config):
            print(
                f"[sweep] ({run_idx}/{len(sweep)}) skipping distribution="
                f"{config['target_distribution']} N={config['sample_size']} "
                f"trials={config['trials']}"
            )
            continue

        command_config = dict(config)
        if args.device is not None:
            command_config["device"] = args.device
        command_config["no_progress"] = True
        command = [
            sys.executable,
            "-m",
            "Synthetic_data.run_eval",
            *config_to_cli(command_config),
        ]
        print(
            f"[sweep] ({run_idx}/{len(sweep)}) distribution="
            f"{config['target_distribution']} N={config['sample_size']} "
            f"clusters={config['cluster_num']} trials={config['trials']} "
            f"seed={config['seed']}"
        )
        try:
            subprocess.run(command, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[sweep] Failed command: {shlex.join(command)}", file=sys.stderr)
            raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
