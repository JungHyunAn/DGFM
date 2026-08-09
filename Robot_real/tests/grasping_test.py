"""Offline grasp/carry diagnostic for real pick-and-place policies.

The script samples identical held-out observations during demonstrated closed
gripper holds, asks every policy for an action chunk, and measures false gripper
reopen commands plus arm-chunk smoothness.  It only reads data and checkpoints;
it never imports a robot driver or executes an action.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Robot_real.dataset_utils import (  # noqa: E402
    PICK_AND_PLACE_GRIPPER_NAME,
    load_aligned_demos,
    select_train_validation_demos,
)
from Robot_real.rollout_model import RealRobotPolicy  # noqa: E402
from Robot_real.tests.smoothness_test import (  # noqa: E402
    METHOD_SPECS,
    _absolute_path,
    _observation_fingerprint,
    resolve_checkpoints,
    resolve_dataset,
    resolve_training_configs,
    validate_artifact_compatibility,
)
from Robot_real.train_model import set_seed  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "grasping_task_results"
DEFAULT_OFFSETS_SEC = (0.0, 0.2, 0.8)
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class GripperLevels:
    close: float
    open: float
    threshold: float


@dataclass(frozen=True)
class GraspEvent:
    close_index: int
    release_index: int


def infer_gripper_levels(demos: Sequence[Any]) -> GripperLevels:
    """Infer low=close and high=open command levels robustly from demos."""
    if not demos:
        raise ValueError("At least one demonstration is required")
    values = np.concatenate(
        [np.asarray(demo.actions[:, -1], dtype=np.float64) for demo in demos]
    )
    close = float(np.quantile(values, 0.05))
    open_value = float(np.quantile(values, 0.95))
    if not np.isfinite(close) or not np.isfinite(open_value):
        raise ValueError("Gripper actions contain non-finite values")
    if open_value - close < 1e-4:
        raise ValueError(
            "Could not separate open and close gripper commands: "
            f"q05={close:.8f}, q95={open_value:.8f}"
        )
    return GripperLevels(close, open_value, 0.5 * (close + open_value))


def _sustained(mask: np.ndarray, start: int, value: bool, run: int) -> bool:
    stop = start + run
    return stop <= len(mask) and bool(np.all(mask[start:stop] == value))


def find_grasp_event(
    gripper_actions: np.ndarray,
    levels: GripperLevels,
    *,
    minimum_run: int = 3,
) -> GraspEvent:
    """Return the first sustained open->close->open command sequence."""
    values = np.asarray(gripper_actions, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected one gripper trajectory, got {values.shape}")
    if minimum_run <= 0:
        raise ValueError("minimum_run must be positive")
    closed = values < levels.threshold
    close_index = None
    for index in range(1, len(closed)):
        if not closed[index - 1] and _sustained(closed, index, True, minimum_run):
            close_index = index
            break
    if close_index is None:
        raise ValueError("No sustained open-to-close command transition was found")

    release_index = None
    for index in range(close_index + minimum_run, len(closed)):
        if closed[index - 1] and _sustained(closed, index, False, minimum_run):
            release_index = index
            break
    if release_index is None:
        raise ValueError("No sustained release after the grasp command was found")
    return GraspEvent(close_index, release_index)


def resolve_held_out_indices(
    demos: Sequence[Any],
    metadata_by_method: dict[str, dict[str, Any]],
    reference_config: dict[str, Any],
    training_demos: int,
) -> list[int]:
    """Use the checkpoint's validation split, with deterministic fallback."""
    recorded = {
        tuple(int(index) for index in metadata.get("val_indices") or ())
        for metadata in metadata_by_method.values()
    }
    if len(recorded) != 1:
        raise ValueError(f"Checkpoints disagree on validation indices: {sorted(recorded)}")
    indices = list(next(iter(recorded)))
    if not indices:
        _, _, _, indices = select_train_validation_demos(
            demos,
            training_demos,
            validation=bool(reference_config.get("validation", True)),
            val_samples=int(reference_config.get("val_samples", 5)),
        )
    if not indices:
        raise ValueError("No held-out demonstrations are available")
    invalid = [index for index in indices if index < 0 or index >= len(demos)]
    if invalid:
        raise ValueError(
            f"Checkpoint validation indices are outside a {len(demos)}-demo dataset: {invalid}"
        )
    return indices


def _offset_label(offset_sec: float) -> str:
    return f"{offset_sec:+.1f}s"


def is_closed_target_chunk(target: np.ndarray, threshold: float) -> bool:
    """Return whether every gripper target in a full action chunk is closed."""
    target = np.asarray(target)
    if target.ndim != 2 or target.shape[0] == 0 or target.shape[1] == 0:
        raise ValueError(f"Expected a non-empty (H, D) target chunk, got {target.shape}")
    return bool(np.all(target[:, -1] < threshold))


def build_grasp_samples(
    demos: Sequence[Any],
    demo_indices: Sequence[int],
    *,
    levels: GripperLevels,
    horizon: int,
    observation_horizon: int,
    observation_dt_sec: float,
    offsets_sec: Sequence[float],
    minimum_run: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for demo_index in demo_indices:
        demo = demos[demo_index]
        event = find_grasp_event(
            demo.actions[:, -1], levels, minimum_run=minimum_run
        )
        held_steps = event.release_index - event.close_index
        events.append(
            {
                "demo_index": int(demo_index),
                "demo_name": demo.name,
                "close_index": int(event.close_index),
                "release_index": int(event.release_index),
                "held_steps": int(held_steps),
                "held_seconds_from_indices": float(held_steps * observation_dt_sec),
                "held_seconds_from_timestamps": float(
                    demo.timestamps_s[event.release_index]
                    - demo.timestamps_s[event.close_index]
                ),
            }
        )
        for offset_sec in offsets_sec:
            offset_steps = int(round(offset_sec / observation_dt_sec))
            start = event.close_index + offset_steps
            observation_indices = np.maximum(
                0,
                np.arange(
                    start - observation_horizon + 1,
                    start + 1,
                    dtype=np.int64,
                ),
            )
            if start < 0 or start + horizon > len(demo):
                raise ValueError(
                    f"{demo.name}: offset {_offset_label(offset_sec)} produces invalid "
                    f"chunk [{start}, {start + horizon}) for length {len(demo)}"
                )
            if not np.all(demo.camera_frame_valid[observation_indices]):
                raise ValueError(
                    f"{demo.name}: offset {_offset_label(offset_sec)} uses an invalid camera frame"
                )
            image_history = [
                [str(path) for path in demo.image_paths[index]]
                for index in observation_indices
            ]
            state_history = demo.states[observation_indices].copy()
            target = demo.actions[start : start + horizon].copy()
            if not is_closed_target_chunk(target, levels.threshold):
                raise ValueError(
                    f"{demo.name}: offset {_offset_label(offset_sec)} does not keep "
                    f"the gripper target closed for the full {horizon}-step chunk"
                )
            samples.append(
                {
                    "demo_index": int(demo_index),
                    "demo_name": demo.name,
                    "close_index": int(event.close_index),
                    "release_index": int(event.release_index),
                    "start": int(start),
                    "offset_sec": float(offset_sec),
                    "offset_label": _offset_label(offset_sec),
                    "observation_indices": observation_indices.tolist(),
                    "images": image_history,
                    "states": state_history,
                    "target": target,
                    "fingerprint": _observation_fingerprint(image_history, state_history),
                }
            )
    return samples, events


def chunk_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    latest_state: np.ndarray,
    *,
    gripper_threshold: float,
    execution_horizon: int,
    guard_steps: int,
    open_wait_steps: int,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    latest_state = np.asarray(latest_state, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            f"Prediction/target must have equal (H, D) shapes, got "
            f"{prediction.shape} and {target.shape}"
        )
    if not 0 < execution_horizon <= len(prediction):
        raise ValueError("execution_horizon must lie within the predicted chunk")
    if guard_steps < 0 or guard_steps >= execution_horizon:
        raise ValueError("guard_steps must lie in [0, execution_horizon)")
    if open_wait_steps <= 0 or open_wait_steps > execution_horizon:
        raise ValueError("open_wait_steps must lie in [1, execution_horizon]")

    prefix = prediction[:execution_horizon]
    target_prefix = target[:execution_horizon]
    predicted_open = prefix[:, -1] >= gripper_threshold
    target_closed = target_prefix[:, -1] < gripper_threshold
    false_open = predicted_open & target_closed
    false_open_after_wait = any(
        _sustained(false_open, index, True, open_wait_steps)
        for index in range(len(false_open))
    )
    arm_prediction = prefix[:, :-1]
    arm_target = target_prefix[:, :-1]
    second = np.diff(arm_prediction, n=2, axis=0)
    return {
        "predicted_open_in_executed_prefix": bool(np.any(predicted_open)),
        "predicted_open_after_guard": bool(np.any(predicted_open[guard_steps:])),
        "false_open_in_executed_prefix": bool(np.any(false_open)),
        "false_open_after_guard": bool(np.any(false_open[guard_steps:])),
        "false_open_after_switch_wait": bool(false_open_after_wait),
        "false_open_step_count": int(false_open.sum()),
        "target_closed_step_count": int(target_closed.sum()),
        "arm_mean_second_difference_l2": float(
            np.linalg.norm(second, axis=1).mean()
        ),
        "arm_first_action_state_jump_l2": float(
            np.linalg.norm(arm_prediction[0] - latest_state[:-1])
        ),
        "arm_mae": float(np.abs(arm_prediction - arm_target).mean()),
    }


def _run_policy(
    checkpoint: Path,
    samples: Sequence[dict[str, Any]],
    *,
    method: str,
    device: str,
    seeds: Sequence[int],
    levels: GripperLevels,
    execution_horizon: int,
    guard_steps: int,
    open_wait_steps: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    records: list[dict[str, Any]] = []
    action_joint_names: tuple[str, ...] | None = None
    for seed in seeds:
        set_seed(seed)
        policy = RealRobotPolicy(checkpoint, device=device, seed=seed)
        if action_joint_names is None:
            action_joint_names = policy.action_joint_names
        elif action_joint_names != policy.action_joint_names:
            raise ValueError("One checkpoint exposed inconsistent action joint names")
        with torch.no_grad():
            for sample in samples:
                prediction = policy.predict_action_chunk(
                    sample["images"], sample["states"]
                )
                metrics = chunk_metrics(
                    prediction,
                    sample["target"],
                    sample["states"][-1],
                    gripper_threshold=levels.threshold,
                    execution_horizon=execution_horizon,
                    guard_steps=guard_steps,
                    open_wait_steps=open_wait_steps,
                )
                records.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "demo_index": sample["demo_index"],
                        "demo_name": sample["demo_name"],
                        "close_index": sample["close_index"],
                        "release_index": sample["release_index"],
                        "start": sample["start"],
                        "offset_sec": sample["offset_sec"],
                        "offset_label": sample["offset_label"],
                        "observation_indices": sample["observation_indices"],
                        "observation_fingerprint": sample["fingerprint"],
                        "observed_gripper_history": sample["states"][:, -1].tolist(),
                        "target_gripper": sample["target"][:, -1].tolist(),
                        "predicted_gripper": prediction[:, -1].tolist(),
                        **metrics,
                    }
                )
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert action_joint_names is not None
    return records, action_joint_names


def summarize_records(
    records: Sequence[dict[str, Any]], offsets_sec: Sequence[float]
) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    boolean_metrics = (
        "predicted_open_in_executed_prefix",
        "predicted_open_after_guard",
        "false_open_in_executed_prefix",
        "false_open_after_guard",
        "false_open_after_switch_wait",
    )
    scalar_metrics = (
        "false_open_step_count",
        "arm_mean_second_difference_l2",
        "arm_first_action_state_jump_l2",
        "arm_mae",
    )
    for method in METHOD_SPECS:
        summary[method] = {}
        for offset_sec in offsets_sec:
            selected = [
                record
                for record in records
                if record["method"] == method
                and math.isclose(record["offset_sec"], offset_sec, abs_tol=1e-12)
            ]
            if not selected:
                raise ValueError(f"No {method} records for {_offset_label(offset_sec)}")
            values: dict[str, Any] = {"chunks": len(selected)}
            for name in boolean_metrics:
                count = sum(bool(record[name]) for record in selected)
                values[f"{name}_count"] = count
                values[f"{name}_rate"] = count / len(selected)
            for name in scalar_metrics:
                metric_values = np.asarray(
                    [record[name] for record in selected], dtype=np.float64
                )
                values[f"mean_{name}"] = float(metric_values.mean())
                values[f"std_{name}"] = float(metric_values.std(ddof=0))
            summary[method][_offset_label(offset_sec)] = values
    return summary


def print_summary(
    summary: dict[str, dict[str, dict[str, Any]]],
    *,
    offset_sec: float,
    open_wait_steps: int,
) -> None:
    label = _offset_label(offset_sec)
    print(f"\nCarry diagnostic at close {label}")
    print(
        f"{'Method':<12} {'false-open/exec':>15} {'false-open/>guard':>18} "
        f"{f'false-open/{open_wait_steps}wait':>18} "
        f"{'mean ||delta2 q||':>20} {'first jump':>14} {'arm MAE':>12}"
    )
    for method in METHOD_SPECS:
        metrics = summary[method][label]
        chunks = metrics["chunks"]
        print(
            f"{method:<12} "
            f"{metrics['false_open_in_executed_prefix_count']:>7}/{chunks:<7} "
            f"{metrics['false_open_after_guard_count']:>9}/{chunks:<8} "
            f"{metrics['false_open_after_switch_wait_count']:>9}/{chunks:<8} "
            f"{metrics['mean_arm_mean_second_difference_l2']:>20.8f} "
            f"{metrics['mean_arm_first_action_state_jump_l2']:>14.8f} "
            f"{metrics['mean_arm_mae']:>12.8f}"
        )


def run_diagnostic(args: argparse.Namespace) -> Path:
    checkpoints, raw_metadata = resolve_checkpoints("pick_and_place", args.training_demos)
    config_paths, configs = resolve_training_configs(
        "pick_and_place", args.training_demos, checkpoints
    )
    effective = validate_artifact_compatibility(
        "pick_and_place", args.training_demos, raw_metadata, configs
    )
    reference = effective["VanillaFM"]
    horizon = int(reference["horizon"])
    if args.execution_horizon > horizon:
        raise ValueError(
            f"execution_horizon={args.execution_horizon} exceeds horizon={horizon}"
        )
    camera_dir, trajectory_dir, dataset_path = resolve_dataset(
        "pick_and_place", effective
    )
    demos = load_aligned_demos(
        "pick_and_place",
        camera_dir=camera_dir,
        trajectory_dir=trajectory_dir,
        use_gripper=True,
        max_alignment_error_s=float(
            configs["VanillaFM"].get("max_alignment_error_s", 0.05)
        ),
    )
    if demos[0].state_joint_names[-1] != PICK_AND_PLACE_GRIPPER_NAME:
        raise ValueError(
            f"Expected final state coordinate {PICK_AND_PLACE_GRIPPER_NAME!r}, "
            f"got {demos[0].state_joint_names[-1]!r}"
        )
    levels = infer_gripper_levels(demos)
    held_out_indices = resolve_held_out_indices(
        demos, raw_metadata, configs["VanillaFM"], args.training_demos
    )
    samples, events = build_grasp_samples(
        demos,
        held_out_indices,
        levels=levels,
        horizon=horizon,
        observation_horizon=int(reference["observation_horizon"]),
        observation_dt_sec=float(reference["observation_dt_sec"]),
        offsets_sec=args.offsets_sec,
        minimum_run=args.minimum_run,
    )

    all_records: list[dict[str, Any]] = []
    joint_names: tuple[str, ...] | None = None
    for method in METHOD_SPECS:
        records, method_joint_names = _run_policy(
            checkpoints[method],
            samples,
            method=method,
            device=args.device,
            seeds=args.seeds,
            levels=levels,
            execution_horizon=args.execution_horizon,
            guard_steps=args.guard_steps,
            open_wait_steps=args.open_wait_steps,
        )
        if joint_names is None:
            joint_names = method_joint_names
        elif joint_names != method_joint_names:
            raise ValueError(
                f"{method} action coordinates {method_joint_names} differ from {joint_names}"
            )
        all_records.extend(records)

    summary = summarize_records(all_records, args.offsets_sec)
    carry_offset = min(args.offsets_sec, key=lambda value: abs(value - 0.8))
    print_summary(
        summary,
        offset_sec=carry_offset,
        open_wait_steps=args.open_wait_steps,
    )

    carry_samples = [
        sample
        for sample in samples
        if math.isclose(sample["offset_sec"], carry_offset, abs_tol=1e-12)
    ]
    observed_carry = [float(sample["states"][-1, -1]) for sample in carry_samples]
    held_steps = [event["held_steps"] for event in events]
    timestamp = datetime.now(timezone.utc)
    result = {
        "timestamp": timestamp.isoformat(),
        "task_name": "pick_and_place",
        "training_demos": int(args.training_demos),
        "device": args.device,
        "sampling_seeds": [int(seed) for seed in args.seeds],
        "offsets_sec": [float(value) for value in args.offsets_sec],
        "prediction_horizon": horizon,
        "execution_horizon": int(args.execution_horizon),
        "guard_steps": int(args.guard_steps),
        "open_wait_steps": int(args.open_wait_steps),
        "sample_selection": "target_gripper_closed_for_full_prediction_horizon",
        "observation_horizon": int(reference["observation_horizon"]),
        "observation_dt_sec": float(reference["observation_dt_sec"]),
        "dataset_path": dataset_path,
        "camera_dir": str(camera_dir),
        "trajectory_dir": str(trajectory_dir),
        "held_out_indices": held_out_indices,
        "held_out_demo_names": [demos[index].name for index in held_out_indices],
        "checkpoints": {method: str(path) for method, path in checkpoints.items()},
        "training_configs": {method: str(path) for method, path in config_paths.items()},
        "state_gripper_source": raw_metadata["VanillaFM"].get("gripper_state_source"),
        "action_gripper_source": raw_metadata["VanillaFM"].get("gripper_action_source"),
        "gripper_levels": {
            "close_command": levels.close,
            "open_command": levels.open,
            "classification_threshold": levels.threshold,
        },
        "carry_observed_gripper_range": [min(observed_carry), max(observed_carry)],
        "demonstrated_hold_steps_range": [min(held_steps), max(held_steps)],
        "action_joint_names": list(joint_names or ()),
        "events": events,
        "sample_observations": [
            {
                key: sample[key]
                for key in (
                    "demo_index",
                    "demo_name",
                    "start",
                    "offset_sec",
                    "observation_indices",
                    "fingerprint",
                )
            }
            for sample in samples
        ],
        "summary": summary,
        "records": all_records,
    }
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"pick_and_place_demos{args.training_demos}_grasp_"
        f"{timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    )
    output_path = output_dir / filename
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"\nGripper commands: close={levels.close:.6f}, open={levels.open:.6f}; "
        f"carry state range={min(observed_carry):.6f}..{max(observed_carry):.6f}"
    )
    print(f"Demonstrated grasp-to-release duration: {min(held_steps)}..{max(held_steps)} steps")
    print(f"Saved results to {output_path}")
    return output_path


def run_self_tests() -> None:
    levels = infer_gripper_levels(
        [
            type(
                "Demo",
                (),
                {"actions": np.asarray([[0.08], [0.08], [0.0], [0.0], [0.0], [0.08]])},
            )()
        ]
    )
    assert levels.close == 0.0
    assert math.isclose(levels.open, 0.08)
    event = find_grasp_event(
        np.asarray([0.08, 0.08, 0.0, 0.0, 0.0, 0.08, 0.08, 0.08]),
        levels,
        minimum_run=3,
    )
    assert event == GraspEvent(close_index=2, release_index=5)
    assert is_closed_target_chunk(np.zeros((4, 3)), levels.threshold)
    opening_target = np.zeros((4, 3))
    opening_target[-1, -1] = 0.08
    assert not is_closed_target_chunk(opening_target, levels.threshold)

    target = np.zeros((5, 3), dtype=np.float64)
    prediction = target.copy()
    prediction[:2, -1] = 0.08
    metrics = chunk_metrics(
        prediction,
        target,
        np.zeros(3),
        gripper_threshold=levels.threshold,
        execution_horizon=4,
        guard_steps=2,
        open_wait_steps=3,
    )
    assert metrics["predicted_open_in_executed_prefix"]
    assert metrics["false_open_in_executed_prefix"]
    assert not metrics["predicted_open_after_guard"]
    assert not metrics["false_open_after_switch_wait"]
    assert metrics["false_open_step_count"] == 2
    assert metrics["arm_mean_second_difference_l2"] == 0.0
    assert metrics["arm_first_action_state_jump_l2"] == 0.0

    persistent_prediction = target.copy()
    persistent_prediction[:3, -1] = 0.08
    persistent_metrics = chunk_metrics(
        persistent_prediction,
        target,
        np.zeros(3),
        gripper_threshold=levels.threshold,
        execution_horizon=4,
        guard_steps=2,
        open_wait_steps=3,
    )
    assert persistent_metrics["false_open_after_switch_wait"]

    suffix_only_jitter = np.zeros((32, 3), dtype=np.float64)
    suffix_only_jitter[16:, 0] = np.tile([0.0, 1.0], 8)
    prefix_metrics = chunk_metrics(
        suffix_only_jitter,
        np.zeros_like(suffix_only_jitter),
        np.zeros(3),
        gripper_threshold=levels.threshold,
        execution_horizon=16,
        guard_steps=2,
        open_wait_steps=3,
    )
    assert prefix_metrics["arm_mean_second_difference_l2"] == 0.0
    assert prefix_metrics["arm_mae"] == 0.0

    mocked = []
    for method in METHOD_SPECS:
        for offset in DEFAULT_OFFSETS_SEC:
            mocked.append(
                {
                    "method": method,
                    "offset_sec": offset,
                    "predicted_open_in_executed_prefix": False,
                    "predicted_open_after_guard": False,
                    "false_open_in_executed_prefix": False,
                    "false_open_after_guard": False,
                    "false_open_after_switch_wait": False,
                    "false_open_step_count": 0,
                    "arm_mean_second_difference_l2": 0.0,
                    "arm_first_action_state_jump_l2": 0.0,
                    "arm_mae": 0.0,
                }
            )
    summary = summarize_records(mocked, DEFAULT_OFFSETS_SEC)
    assert summary["NGFM"]["+0.8s"]["chunks"] == 1
    print("Grasping-task self-tests passed (offline only; no robot initialized).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-demos", type=int, default=25)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--guard-steps", type=int, default=2)
    parser.add_argument("--open-wait-steps", type=int, default=3)
    parser.add_argument("--minimum-run", type=int, default=3)
    parser.add_argument("--offsets-sec", type=float, nargs="+", default=DEFAULT_OFFSETS_SEC)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check event detection and metrics without loading data or checkpoints.",
    )
    args = parser.parse_args()
    if args.execution_horizon <= 0:
        parser.error("--execution-horizon must be positive")
    if args.guard_steps < 0 or args.guard_steps >= args.execution_horizon:
        parser.error("--guard-steps must be in [0, execution-horizon)")
    if args.open_wait_steps <= 0 or args.open_wait_steps > args.execution_horizon:
        parser.error("--open-wait-steps must be in [1, execution-horizon]")
    if args.minimum_run <= 0:
        parser.error("--minimum-run must be positive")
    if not args.offsets_sec:
        parser.error("--offsets-sec must contain at least one offset")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
    else:
        warnings.filterwarnings("default", category=RuntimeWarning)
        run_diagnostic(args)


if __name__ == "__main__":
    main()
