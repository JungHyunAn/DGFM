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
