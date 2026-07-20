"""Run the fixed Synthetic_data evaluation sweep through run_eval.py."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


SAMPLE_SIZES = (320, 640, 1280, 2560, 5120)
CLUSTER_NUMS = (8, 16, 32, 64, 128)
TARGET_DISTRIBUTIONS = ("6", "7", "8")
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
    "early_stopping": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, nargs="?", default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--results_path",
        type=Path,
        default=None,
        help="Defaults to Synthetic_data/eval_results/sweep_<seed>.",
    )
    return parser.parse_args()


def build_sweep(seed: int, results_root: Path) -> list[dict]:
    if len(SAMPLE_SIZES) != len(CLUSTER_NUMS):
        raise ValueError("SAMPLE_SIZES and CLUSTER_NUMS must have matching lengths")
    return [
        {
            **BASE_CONFIG,
            "sample_size": sample_size,
            "cluster_num": cluster_num,
            "target_distribution": distribution,
            "trial_idx": trial_idx,
            "seed": seed + trial_idx,
            "results_path": results_root
            / f"distribution_{distribution}"
            / f"sample_size_{sample_size}"
            / f"trial_{trial_idx}",
        }
        for distribution in TARGET_DISTRIBUTIONS
        for sample_size, cluster_num in zip(SAMPLE_SIZES, CLUSTER_NUMS, strict=True)
        for trial_idx in range(TRIALS)
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
    output_path = Path(config["results_path"]) / "results.json"
    try:
        with output_path.open() as source:
            result = json.load(source)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("seed") == config["seed"]
        and result.get("trial_idx") == config["trial_idx"]
        and result.get("sample_size") == config["sample_size"]
        and result.get("cluster_num") == config["cluster_num"]
        and set(result.get("results", {})) == {"UniformFM", "DGFMv2"}
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    results_root = args.results_path or Path(
        f"Synthetic_data/eval_results/sweep_{args.seed}"
    )
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    sweep = build_sweep(args.seed, results_root)
    print(f"[sweep] Prepared {len(sweep)} runs under {results_root}")

    for run_idx, config in enumerate(sweep, start=1):
        if args.resume and is_complete(config):
            print(
                f"[sweep] ({run_idx}/{len(sweep)}) skipping distribution="
                f"{config['target_distribution']} N={config['sample_size']} "
                f"trial={config['trial_idx']}"
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
            f"clusters={config['cluster_num']} trial={config['trial_idx']} "
            f"seed={config['seed']}"
        )
        try:
            subprocess.run(command, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[sweep] Failed command: {shlex.join(command)}", file=sys.stderr)
            raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
