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
    *,
    bins_per_dimension: int = 3,
) -> list[int]:
    """Return a nested, grid-balanced prefix of a deterministic episode ordering.

    Each environment-parameter dimension is split into a fixed number of bins.
    Occupied cells and the episodes within each cell are deterministically
    shuffled from training_seed. Episodes are then selected round-robin across
    cells, so every budget is a prefix of the same ordering and cell counts
    remain as balanced as the available data permits.
    """
    parameters = np.asarray(environment_parameters)
    if parameters.ndim != 2:
        raise ValueError(
            "environment_parameters must have shape (episodes, dimensions), "
            f"got {parameters.shape}"
        )
    total_episodes, environment_dim = parameters.shape
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if budget > total_episodes:
        raise ValueError(
            f"Requested {budget} demonstrations, but the dataset contains only "
            f"{total_episodes} episodes"
        )
    if not isinstance(bins_per_dimension, (int, np.integer)) or bins_per_dimension <= 0:
        raise ValueError(
            "bins_per_dimension must be a positive integer, got "
            f"{bins_per_dimension!r}"
        )
    if budget == 0:
        return []
    if environment_dim == 0:
        return selected_episode_indices(total_episodes, training_seed, budget)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("environment_parameters must contain only finite values")

    lower = parameters.min(axis=0)
    spans = parameters.max(axis=0) - lower
    constant_dimensions = np.flatnonzero(spans == 0)
    if constant_dimensions.size:
        raise ValueError(
            "Cannot construct an environment grid from constant parameter "
            f"dimensions {constant_dimensions.tolist()}"
        )

    normalized_parameters = (parameters - lower) / spans
    coordinates = np.floor(
        normalized_parameters * bins_per_dimension
    ).astype(np.int64)
    np.clip(coordinates, 0, bins_per_dimension - 1, out=coordinates)
    cell_ids = np.ravel_multi_index(
        coordinates.T, (bins_per_dimension,) * environment_dim
    )

    rng = np.random.default_rng(int(training_seed))
    episode_permutation = rng.permutation(total_episodes)
    episode_ranks = np.empty(total_episodes, dtype=np.int64)
    episode_ranks[episode_permutation] = np.arange(total_episodes)

    occupied_cells = np.unique(cell_ids)
    cell_order = rng.permutation(occupied_cells)
    cell_episodes = {
        int(cell): sorted(
            np.flatnonzero(cell_ids == cell).tolist(),
            key=episode_ranks.__getitem__,
        )
        for cell in cell_order
    }

    selected: list[int] = []
    round_idx = 0
    while len(selected) < budget:
        added_this_round = False
        for cell in cell_order:
            episodes = cell_episodes[int(cell)]
            if round_idx >= len(episodes):
                continue
            selected.append(int(episodes[round_idx]))
            added_this_round = True
            if len(selected) == budget:
                return selected
        if not added_this_round:
            raise RuntimeError(
                "Environment-grid ordering exhausted before reaching the requested budget"
            )
        round_idx += 1
    return selected


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
