"""Pure reproducibility helpers shared by data generation and experiments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TASK_EVAL_BASE_SEEDS = {
    "door": 410_000,
    "wipe": 420_000,
    "two_arm": 430_000,
    "nut": 440_000,
}

TASK_ENVIRONMENT_RANGES = {
    "door": {
        "x_range": [-0.075, 0.075],
        "y_range": [-0.3, -0.1],
        "yaw_range": [-float(np.pi / 2 + np.pi / 8), -float(np.pi / 2)],
    },
    "two_arm": {
        "x_range": [-0.015, 0.015],
        "y_range": [-0.015, 0.015],
        "yaw_range": [float(np.pi - np.pi / 6), float(np.pi + np.pi / 6)],
    },
    "nut": {
        "square_nut_x_range": [-0.12, -0.11],
        "square_nut_y_range": [0.11, 0.14],
        "square_nut_yaw_range": [float(np.pi / 2 - np.pi / 6), float(np.pi / 2 + np.pi / 6)],
        "round_nut_x_range": [-0.115, -0.11],
        "round_nut_y_range": [-0.225, -0.11],
    },
    "wipe": {
        "table_full_size": [0.4, 0.6, 0.05],
        "table_offset": [0.3, 0.0, 1.0],
        "num_markers": 50,
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def episode_seed(dataset_base_seed: int, episode_id: int) -> int:
    """Derive one deterministic uint32 seed from a dataset and episode identity."""
    if episode_id < 0:
        raise ValueError("episode_id must be non-negative")
    sequence = np.random.SeedSequence([int(dataset_base_seed), int(episode_id)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def episode_seed_plan(dataset_base_seed: int, episode_ids: Iterable[int]) -> list[int]:
    return [episode_seed(dataset_base_seed, episode_id) for episode_id in episode_ids]


def selected_episode_indices(total_episodes: int, training_seed: int, budget: int) -> list[int]:
    """Return the ordered prefix of one permutation, making budgets nested."""
    if total_episodes < 0 or budget < 0:
        raise ValueError("total_episodes and budget must be non-negative")
    permutation = np.random.default_rng(int(training_seed)).permutation(total_episodes)
    return permutation[: min(budget, total_episodes)].astype(np.int64).tolist()


def validation_suite_spec(
    task_name: str,
    trial_count: int = 50,
    eval_base_seed: int | None = None,
) -> dict[str, Any]:
    if trial_count < 0:
        raise ValueError("trial_count must be non-negative")
    resolved_seed = (
        TASK_EVAL_BASE_SEEDS[task_name] if eval_base_seed is None else int(eval_base_seed)
    )
    trial_seeds = [resolved_seed + idx for idx in range(trial_count)]
    identifier = f"{task_name}-validation-v1-{resolved_seed}-{trial_count}"
    return {
        "eval_base_seed": resolved_seed,
        "eval_trial_seeds": trial_seeds,
        "validation_suite_id": identifier,
        "validation_suite_path": None,
        "validation_trial_count": trial_count,
        "validation_suite_hash": stable_hash(
            {"task_name": task_name, "version": 1, "trial_seeds": trial_seeds}
        ),
    }


def git_commit(repo_root: str | os.PathLike[str] | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Hash a dataset, reusing a stat-validated sidecar cache when available."""
    dataset_path = Path(path).expanduser().resolve()
    stat = dataset_path.stat()
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    cached = None
    try:
        with manifest_path.open("r") as stream:
            cached = json.load(stream)
    except (OSError, json.JSONDecodeError):
        pass
    stat_fields = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if isinstance(cached, dict) and all(cached.get(k) == v for k, v in stat_fields.items()):
        digest = cached.get("sha256")
    else:
        digest = sha256_file(dataset_path)
        cached = {**stat_fields, "sha256": digest, "dataset_path": str(dataset_path)}
        try:
            with manifest_path.open("w") as stream:
                json.dump(cached, stream, indent=2)
        except OSError:
            pass
    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": digest,
        "dataset_size": stat.st_size,
        "dataset_manifest_path": str(manifest_path),
    }
