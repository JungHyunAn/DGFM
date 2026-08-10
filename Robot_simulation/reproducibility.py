"""Pure reproducibility helpers shared by data generation and experiments."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DP_EVAL_POLICY_SEED_SCHEME = (
    "uint64 SeedSequence([eval_base_seed, trial_idx, sha256(dp_policy)]) "
    "with one torch.Generator per rollout"
)
DP_EVAL_SAMPLING_MODE = "per_trial_rng_batched_inference_v1"

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

TASK_ENVIRONMENT_PARAMETER_RANGE_KEYS = {
    "door": ("x_range", "y_range", "yaw_range"),
    "wipe": (),
    "two_arm": ("x_range", "y_range", "yaw_range"),
    "nut": (
        "square_nut_x_range",
        "square_nut_y_range",
        "square_nut_yaw_range",
        "round_nut_x_range",
        "round_nut_y_range",
    ),
}

ENVIRONMENT_GRID_BINS_PER_DIMENSION = 3
ENVIRONMENT_GRID_SAMPLER_SCHEME = "fixed_task_range_3_bin_round_robin_v1"


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


def derive_seed(base_seed: int, index: int, namespace: str) -> int:
    """Derive a stable uint64 seed for an independent named RNG stream."""
    namespace_words = np.frombuffer(
        hashlib.sha256(namespace.encode("utf-8")).digest()[:16], dtype=np.uint32
    )
    sequence = np.random.SeedSequence(
        [int(base_seed), int(index), *(int(word) for word in namespace_words)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def evaluation_policy_seed_plan(
    eval_base_seed: int, trial_count: int, namespace: str = "dp_policy"
) -> list[int]:
    return [derive_seed(eval_base_seed, trial_idx, namespace) for trial_idx in range(trial_count)]


def validation_result_is_better(
    success_rate: float,
    avg_reward: float,
    best_success_rate: float,
    best_avg_reward: float,
    *,
    has_best: bool,
) -> bool:
    """Select by success rate, then reward; retain the earlier exact tie."""
    return (
        not has_best
        or success_rate > best_success_rate
        or (success_rate == best_success_rate and avg_reward > best_avg_reward)
    )


def summarize_validation_records(records: dict, tail_count: int = 10) -> dict[str, Any]:
    """Summarize fixed-suite records with trainer-consistent tie handling."""
    if not records:
        raise ValueError("Validation records are empty")
    ordered = []
    for epoch in sorted(records, key=int):
        record = records[epoch]
        rate = float(record.get("success_rate", float("nan")))
        reward = float(record.get("avg_reward", float("nan")))
        if np.isfinite(rate):
            ordered.append((int(epoch), rate, reward))
    if not ordered:
        raise ValueError("Validation records contain no finite success rates")
    best = ordered[0]
    for candidate in ordered[1:]:
        if validation_result_is_better(
            candidate[1], candidate[2], best[1], best[2], has_best=True
        ):
            best = candidate
    tail = ordered[-tail_count:]
    return {
        "avg_success_rate": float(np.mean([item[1] for item in tail])),
        "avg_success_rate_num_checkpoints": len(tail),
        "final_validation_success_rates": [item[1] for item in tail],
        "final_validation_epochs": [item[0] for item in tail],
        "max_success_rate": max(item[1] for item in ordered),
        "best_validation_epoch": best[0],
        "best_validation_success_rate": best[1],
        "best_validation_reward": best[2],
    }


def selected_episode_indices(total_episodes: int, training_seed: int, budget: int) -> list[int]:
    """Return the ordered prefix of one permutation, making budgets nested."""
    if total_episodes < 0 or budget < 0:
        raise ValueError("total_episodes and budget must be non-negative")
    permutation = np.random.default_rng(int(training_seed)).permutation(total_episodes)
    return permutation[: min(budget, total_episodes)].astype(np.int64).tolist()


def environment_grid_episode_indices(
    environment_parameters: np.ndarray,
    training_seed: int,
    budget: int,
    task_name: str,
) -> list[int]:
    """Return a budget prefix of one fixed, seeded task-grid episode ordering."""
    ordering = environment_grid_episode_order(
        environment_parameters, task_name, training_seed
    )
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if budget > len(ordering):
        raise ValueError(
            f"Requested {budget} demonstrations, but the dataset contains only "
            f"{len(ordering)} episodes"
        )
    return ordering[:budget]


def _environment_grid_cell_ids(
    environment_parameters: np.ndarray,
    task_name: str,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    parameters = np.asarray(environment_parameters)
    if parameters.ndim != 2:
        raise ValueError(
            "environment_parameters must have shape (episodes, dimensions), "
            f"got {parameters.shape}"
        )
    _, environment_dim = parameters.shape
    if task_name not in TASK_ENVIRONMENT_PARAMETER_RANGE_KEYS:
        raise ValueError(f"Unsupported task_name={task_name!r}")
    range_keys = TASK_ENVIRONMENT_PARAMETER_RANGE_KEYS[task_name]
    if environment_dim == 0:
        range_keys = ()
    if environment_dim != len(range_keys):
        raise ValueError(
            f"Task {task_name!r} has {len(range_keys)} configured environment-parameter "
            f"ranges, but the dataset contains {environment_dim} dimensions"
        )
    if not np.all(np.isfinite(parameters)):
        raise ValueError("environment_parameters must contain only finite values")
    if environment_dim == 0:
        return np.zeros(len(parameters), dtype=np.int64), range_keys, np.empty((0, 2))

    ranges = np.asarray(
        [TASK_ENVIRONMENT_RANGES[task_name][key] for key in range_keys],
        dtype=np.float64,
    )
    lower, upper = ranges[:, 0], ranges[:, 1]
    spans = upper - lower
    if not np.all(np.isfinite(ranges)) or np.any(spans <= 0):
        raise ValueError(f"Invalid configured environment ranges for task {task_name!r}")

    grid_parameters = parameters.astype(np.float64, copy=True)
    # atan2 stores angles in [-pi, pi], while a configured yaw interval may cross pi.
    for dimension, key in enumerate(range_keys):
        if "yaw" in key:
            center = (lower[dimension] + upper[dimension]) / 2.0
            grid_parameters[:, dimension] += 2.0 * np.pi * np.round(
                (center - grid_parameters[:, dimension]) / (2.0 * np.pi)
            )

    # Existing datasets may store world-space observations while their configured
    # ranges describe placement offsets. Clamp those values to the boundary bins
    # instead of rejecting an otherwise valid dataset.
    normalized = np.clip((grid_parameters - lower) / spans, 0.0, 1.0)
    coordinates = np.floor(
        normalized * ENVIRONMENT_GRID_BINS_PER_DIMENSION
    ).astype(np.int64)
    np.clip(
        coordinates,
        0,
        ENVIRONMENT_GRID_BINS_PER_DIMENSION - 1,
        out=coordinates,
    )
    cell_ids = np.ravel_multi_index(
        coordinates.T,
        (ENVIRONMENT_GRID_BINS_PER_DIMENSION,) * environment_dim,
    )
    return cell_ids, range_keys, ranges


def environment_grid_episode_order(
    environment_parameters: np.ndarray,
    task_name: str,
    sampling_seed: int,
) -> list[int]:
    """Order every episode by seeded round-robin traversal of occupied grid cells."""
    cell_ids, _, _ = _environment_grid_cell_ids(environment_parameters, task_name)
    rng = np.random.default_rng(int(sampling_seed))
    occupied_cells = np.unique(cell_ids)
    rng.shuffle(occupied_cells)

    episodes_by_cell = {}
    for cell_id in occupied_cells:
        episodes = np.flatnonzero(cell_ids == cell_id)
        rng.shuffle(episodes)
        episodes_by_cell[int(cell_id)] = episodes.tolist()

    ordering = []
    round_index = 0
    while len(ordering) < len(cell_ids):
        for cell_id in occupied_cells:
            episodes = episodes_by_cell[int(cell_id)]
            if round_index < len(episodes):
                ordering.append(int(episodes[round_index]))
        round_index += 1
    return ordering


def environment_grid_sampler_metadata(
    environment_parameters: np.ndarray,
    task_name: str,
    sampling_seed: int,
    ordering: list[int] | None = None,
) -> dict[str, Any]:
    """Describe the fixed grid and full ordering used for an experiment."""
    cell_ids, range_keys, ranges = _environment_grid_cell_ids(
        environment_parameters, task_name
    )
    if ordering is None:
        ordering = environment_grid_episode_order(
            environment_parameters, task_name, sampling_seed
        )
    return {
        "scheme": ENVIRONMENT_GRID_SAMPLER_SCHEME,
        "sampling_seed": int(sampling_seed),
        "bins_per_dimension": ENVIRONMENT_GRID_BINS_PER_DIMENSION,
        "parameter_dimensions": len(range_keys),
        "parameter_range_keys": list(range_keys),
        "parameter_ranges": ranges.tolist(),
        "total_grid_cells": ENVIRONMENT_GRID_BINS_PER_DIMENSION ** len(range_keys),
        "occupied_grid_cells": int(len(np.unique(cell_ids))),
        "episode_order_length": len(ordering),
        "episode_order_sha256": stable_hash(ordering),
    }


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
