"""Summarize smoothness of saved simulation rollout trajectories.

The rollout pickle written by ``run_eval.py`` stores the commands actually sent
to the simulator.  For receding-horizon evaluation, each executed policy prefix
is upsampled and appended to the previous prefix.  This module distinguishes
second differences wholly inside a prefix from those that cross a replan
boundary.

Example:
    python -m Robot_simulation.tests.test_trajectory_smoothness \
        Robot_simulation/eval_results/two_arm/sweep_1000/20260703T234111/best_validation_rollouts.pkl
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class _NumpyRolloutUnpickler(pickle.Unpickler):
    """Load plain containers and NumPy arrays without arbitrary class loading."""

    _ALLOWED_GLOBALS = {
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._ALLOWED_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"Refusing unsupported pickle global {module}.{name}"
            ) from error


def load_rollouts(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load and validate a repository-generated rollout pickle."""
    rollout_path = Path(path).expanduser().resolve()
    if not rollout_path.is_file():
        raise FileNotFoundError(f"Rollout file does not exist: {rollout_path}")
    with rollout_path.open("rb") as stream:
        payload = _NumpyRolloutUnpickler(stream).load()

    if not isinstance(payload, dict):
        raise ValueError("Rollout pickle root must be a dictionary")
    unexpected = set(payload).difference({"success", "failure"})
    if unexpected:
        raise ValueError(f"Unexpected rollout groups: {sorted(unexpected)}")

    validated: dict[str, list[dict[str, Any]]] = {}
    for group in ("success", "failure"):
        records = payload.get(group, [])
        if not isinstance(records, list):
            raise ValueError(f"Rollout group {group!r} must be a list")
        validated[group] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict) or "traj" not in record:
                raise ValueError(f"{group}[{index}] must be a dictionary containing 'traj'")
            trajectory = np.asarray(record["traj"])
            if trajectory.ndim != 2 or trajectory.shape[0] < 3 or trajectory.shape[1] == 0:
                raise ValueError(
                    f"{group}[{index}].traj must have shape (T >= 3, D > 0), "
                    f"got {trajectory.shape}"
                )
            if trajectory.dtype.kind not in "fiu" or not np.isfinite(trajectory).all():
                raise ValueError(f"{group}[{index}].traj must contain finite numeric values")
            validated[group].append(record)
    if not validated["success"] and not validated["failure"]:
        raise ValueError("Rollout pickle contains no trajectories")
    return validated


def load_adjacent_metadata(path: str | Path) -> dict[str, Any]:
    """Read ``results.json`` next to a saved rollout, when available."""
    results_path = Path(path).expanduser().resolve().parent / "results.json"
    if not results_path.is_file():
        return {}
    with results_path.open() as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {results_path}")
    return metadata


def infer_executed_chunk_length(
    metadata: dict[str, Any],
    explicit_chunk_length: int | None = None,
) -> int:
    """Infer the number of saved commands contributed by each policy replan."""
    if explicit_chunk_length is not None:
        if explicit_chunk_length < 2:
            raise ValueError("chunk_length must be at least two")
        return int(explicit_chunk_length)

    required = ("executed_horizon", "recorded_control_freq", "trajectory_control_freq")
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise ValueError(
            "Cannot infer replan boundaries; adjacent results.json is missing "
            f"{', '.join(missing)}. Pass --chunk-length explicitly."
        )
    executed_horizon = int(metadata["executed_horizon"])
    recorded_hz = float(metadata["recorded_control_freq"])
    trajectory_hz = float(metadata["trajectory_control_freq"])
    if executed_horizon < 2 or recorded_hz <= 0 or trajectory_hz <= 0:
        raise ValueError("Invalid execution-horizon or control-frequency metadata")
    ratio = recorded_hz / trajectory_hz
    sample_step = int(round(ratio))
    if sample_step < 1 or not math.isclose(ratio, sample_step, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "recorded_control_freq must be an integer multiple of "
            "trajectory_control_freq"
        )
    # Matches env_util._upsample_policy_trajectory.
    return (executed_horizon - 1) * sample_step + 1


def default_action_indices(
    action_dimension: int,
    *,
    include_gripper: bool = False,
) -> list[int]:
    """Select arm coordinates, assuming each joint-space robot has 7+1 DoF."""
    if action_dimension <= 0:
        raise ValueError("action_dimension must be positive")
    if include_gripper or action_dimension % 8 != 0:
        return list(range(action_dimension))
    gripper_indices = set(range(7, action_dimension, 8))
    return [index for index in range(action_dimension) if index not in gripper_indices]


def trajectory_smoothness(
    trajectory: np.ndarray,
    *,
    chunk_length: int,
    action_indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Compute one rollout's first/second-difference L2 smoothness metrics."""
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] == 0:
        raise ValueError(f"Expected trajectory shaped (T >= 3, D > 0), got {values.shape}")
    if chunk_length < 2:
        raise ValueError("chunk_length must be at least two")
    if action_indices is not None:
        values = values[:, list(action_indices)]
    if values.shape[1] == 0:
        raise ValueError("At least one action coordinate must be selected")

    first_l2 = np.linalg.norm(np.diff(values, axis=0), axis=1)
    second_l2 = np.linalg.norm(np.diff(values, n=2, axis=0), axis=1)

    # A first difference crosses a boundary if its right endpoint begins a new
    # prefix. A second difference crosses one if either constituent first
    # difference does so; this assigns two acceleration samples per boundary.
    second_index = np.arange(second_l2.size)
    boundary_mask = (
        ((second_index + 1) % chunk_length == 0)
        | ((second_index + 2) % chunk_length == 0)
    )
    if not boundary_mask.any():
        raise ValueError(
            f"Trajectory length {len(values)} contains no replan boundary for "
            f"chunk_length={chunk_length}"
        )
    return {
        "mean_first_difference_l2": float(first_l2.mean()),
        "mean_second_difference_l2": float(second_l2.mean()),
        "within_chunk_jitter": float(second_l2[~boundary_mask].mean()),
        "replan_boundary_jitter": float(second_l2[boundary_mask].mean()),
    }


def summarize_trajectory_smoothness(
    rollout_path: str | Path,
    *,
    subset: str = "all",
    chunk_length: int | None = None,
    include_gripper: bool = False,
) -> dict[str, Any]:
    """Return equal-rollout-mean jitter metrics for a saved rollout file."""
    if subset not in {"all", "success", "failure"}:
        raise ValueError("subset must be one of: all, success, failure")
    rollouts = load_rollouts(rollout_path)
    metadata = load_adjacent_metadata(rollout_path)
    resolved_chunk_length = infer_executed_chunk_length(metadata, chunk_length)
    records = (
        rollouts["success"] + rollouts["failure"]
        if subset == "all"
        else rollouts[subset]
    )
    if not records:
        raise ValueError(f"No {subset} trajectories are available")

    action_dimensions = {np.asarray(record["traj"]).shape[1] for record in records}
    if len(action_dimensions) != 1:
        raise ValueError(f"Action dimensions differ across trajectories: {action_dimensions}")
    action_dimension = action_dimensions.pop()
    action_indices = default_action_indices(
        action_dimension,
        include_gripper=include_gripper,
    )
    per_rollout = [
        trajectory_smoothness(
            record["traj"],
            chunk_length=resolved_chunk_length,
            action_indices=action_indices,
        )
        for record in records
    ]
    metric_names = tuple(per_rollout[0])
    summary = {
        name: float(np.mean([metrics[name] for metrics in per_rollout]))
        for name in metric_names
    }
    return {
        "trajectory_path": str(Path(rollout_path).expanduser().resolve()),
        "model_type": metadata.get("model_type"),
        "subset": subset,
        "rollout_count": len(records),
        "success_count": len(rollouts["success"]),
        "failure_count": len(rollouts["failure"]),
        "action_dimension": action_dimension,
        "evaluated_action_indices": action_indices,
        "include_gripper": include_gripper,
        "executed_chunk_length": resolved_chunk_length,
        "aggregation": "equal_rollout_mean",
        **summary,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Trajectory: {summary['trajectory_path']}")
    if summary["model_type"] is not None:
        print(f"Model: {summary['model_type']}")
    print(
        f"Subset: {summary['subset']} ({summary['rollout_count']} rollouts; "
        f"{summary['success_count']} success, {summary['failure_count']} failure)"
    )
    print(
        f"Action coordinates: {len(summary['evaluated_action_indices'])}/"
        f"{summary['action_dimension']} | executed chunk length: "
        f"{summary['executed_chunk_length']}"
    )
    print("Jitter summary (equal mean across rollouts)")
    print(f"  mean ||delta q||:          {summary['mean_first_difference_l2']:.8f}")
    print(f"  mean ||delta^2 q||:        {summary['mean_second_difference_l2']:.8f}")
    print(f"  within-chunk jitter:       {summary['within_chunk_jitter']:.8f}")
    print(f"  replan-boundary jitter:    {summary['replan_boundary_jitter']:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_path", type=Path, help="Path to a rollout .pkl/.pkt file")
    parser.add_argument(
        "--subset",
        choices=("all", "success", "failure"),
        default="all",
        help="Which saved rollouts to summarize (default: all)",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        help="Saved commands per executed prefix; inferred from adjacent results.json by default",
    )
    parser.add_argument(
        "--include-gripper",
        action="store_true",
        help="Include gripper coordinates in the L2 metrics",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_trajectory_smoothness(
        args.trajectory_path,
        subset=args.subset,
        chunk_length=args.chunk_length,
        include_gripper=args.include_gripper,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_summary(summary)


class TrajectorySmoothnessTest(unittest.TestCase):
    def test_linear_chunks_isolate_replan_boundary_jitter(self):
        trajectory = np.asarray([0, 1, 2, 3, 10, 11, 12, 13], dtype=np.float32)[:, None]
        metrics = trajectory_smoothness(trajectory, chunk_length=4)
        self.assertAlmostEqual(metrics["mean_first_difference_l2"], 13.0 / 7.0)
        self.assertEqual(metrics["mean_second_difference_l2"], 2.0)
        self.assertEqual(metrics["within_chunk_jitter"], 0.0)
        self.assertEqual(metrics["replan_boundary_jitter"], 6.0)

    def test_default_two_arm_indices_exclude_grippers(self):
        self.assertEqual(
            default_action_indices(16),
            list(range(7)) + list(range(8, 15)),
        )
        self.assertEqual(default_action_indices(16, include_gripper=True), list(range(16)))

    def test_chunk_length_matches_cubic_spline_output_length(self):
        metadata = {
            "executed_horizon": 8,
            "recorded_control_freq": 20.0,
            "trajectory_control_freq": 10.0,
        }
        self.assertEqual(infer_executed_chunk_length(metadata), 15)

    def test_restricted_loader_accepts_numpy_rollouts(self):
        payload = {
            "success": [{"traj": np.zeros((5, 2), dtype=np.float32), "setting": {}}],
            "failure": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.pkl"
            with path.open("wb") as stream:
                pickle.dump(payload, stream)
            loaded = load_rollouts(path)
        self.assertTrue(np.array_equal(loaded["success"][0]["traj"], payload["success"][0]["traj"]))

    def test_restricted_loader_rejects_arbitrary_globals(self):
        with self.assertRaises(pickle.UnpicklingError):
            _NumpyRolloutUnpickler(io.BytesIO(pickle.dumps(eval))).load()


if __name__ == "__main__":
    main()
