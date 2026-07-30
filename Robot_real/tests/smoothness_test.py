"""Offline action-chunk smoothness comparison for real-robot policies.

This module only reads recorded data and checkpoints.  It never imports a
robot driver, creates a robot environment, or executes a predicted action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Robot_real.dataset_utils import (  # noqa: E402
    TASK_NAMES,
    ActionChunkDataset,
    default_dataset_paths,
    denormalize_joint_angles,
    load_aligned_demos,
)
from Robot_real.rollout_model import RealRobotPolicy  # noqa: E402
from Robot_real.train_model import set_seed  # noqa: E402


ROBOT_REAL_ROOT = REPOSITORY_ROOT / "Robot_real"
CHECKPOINT_ROOT = ROBOT_REAL_ROOT / "checkpoints"
CONFIG_ROOT = ROBOT_REAL_ROOT / "real_config"
DEFAULT_OUTPUT_DIR = ROBOT_REAL_ROOT / "tests" / "smoothness_test_results"

METHOD_SPECS = {
    "VanillaFM": {"uniformfm", "vanillafm", "fm"},
    "DP": {"diffusionpolicy", "diffusion", "dp"},
    "NGFM": {"dgfmv2", "ngfm"},
}
COMPARISON_METADATA_KEYS = (
    "horizon",
    "observation_horizon",
    "observation_dt_sec",
    "image_size",
    "use_gripper",
    "camera_names",
    "state_joint_names",
    "action_joint_names",
    "state_dof",
    "action_dof",
    "normalization_type",
    "execution_horizon",
)
CONFIG_VALIDATION_KEYS = (
    "task",
    "dataset_size",
    "horizon",
    "observation_horizon",
    "observation_dt_sec",
    "image_size",
    "use_gripper",
    "sampler_steps",
    "diffusion_steps",
    "diffusion_schedule",
    "diffusion_pred_type",
    "diffusion_eta",
    "clip_sample",
    "clip_sample_range",
)


class ArtifactResolutionError(RuntimeError):
    """Raised when an experimental artifact cannot be resolved uniquely."""


def _method_for_model_type(model_type: object) -> str | None:
    normalized = str(model_type).replace("_", "").replace("-", "").lower()
    for method, aliases in METHOD_SPECS.items():
        if normalized in aliases:
            return method
    return None


def _absolute_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate).resolve()


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ArtifactResolutionError(f"Could not read checkpoint {path}: {error}") from error
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ArtifactResolutionError(f"Checkpoint has no metadata dictionary: {path}")
    return metadata


def resolve_checkpoints(
    task_name: str,
    training_demos: int,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    pattern = str(CHECKPOINT_ROOT / "*.pt")
    paths = sorted(CHECKPOINT_ROOT.glob("*.pt"))
    by_method: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        method: [] for method in METHOD_SPECS
    }
    for path in paths:
        metadata = _load_checkpoint_metadata(path)
        method = _method_for_model_type(metadata.get("model_type"))
        if (
            method is not None
            and metadata.get("task") == task_name
            and metadata.get("dataset_size") == training_demos
        ):
            by_method[method].append((path.resolve(), metadata))

    resolved_paths: dict[str, Path] = {}
    resolved_metadata: dict[str, dict[str, Any]] = {}
    errors = []
    for method, candidates in by_method.items():
        if len(candidates) != 1:
            candidate_lines = (
                "\n".join(f"    - {path}" for path, _ in candidates)
                if candidates
                else "    (none)"
            )
            errors.append(
                f"{method}: expected exactly one checkpoint for "
                f"task={task_name!r}, training_demos={training_demos}; "
                f"searched {pattern}; candidates:\n{candidate_lines}"
            )
            continue
        resolved_paths[method], resolved_metadata[method] = candidates[0]
    if errors:
        raise ArtifactResolutionError("Checkpoint resolution failed:\n" + "\n".join(errors))
    return resolved_paths, resolved_metadata


def resolve_training_configs(
    task_name: str,
    training_demos: int,
    checkpoints: dict[str, Path],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    configs_by_method: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        method: [] for method in METHOD_SPECS
    }
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        try:
            config = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactResolutionError(f"Could not read training config {path}: {error}") from error
        method = _method_for_model_type(config.get("model_type"))
        if (
            method is not None
            and config.get("task") == task_name
            and config.get("dataset_size") == training_demos
            and config.get("checkpoint_path") is not None
            and _absolute_path(config["checkpoint_path"]) == checkpoints[method]
        ):
            configs_by_method[method].append((path.resolve(), config))

    resolved_paths: dict[str, Path] = {}
    resolved_configs: dict[str, dict[str, Any]] = {}
    errors = []
    pattern = str(CONFIG_ROOT / "*.json")
    for method, candidates in configs_by_method.items():
        if len(candidates) != 1:
            candidate_lines = (
                "\n".join(f"    - {path}" for path, _ in candidates)
                if candidates
                else "    (none)"
            )
            errors.append(
                f"{method}: expected exactly one training config for "
                f"task={task_name!r}, training_demos={training_demos}, "
                f"checkpoint={checkpoints[method]}; searched {pattern}; candidates:\n"
                f"{candidate_lines}"
            )
            continue
        resolved_paths[method], resolved_configs[method] = candidates[0]
    if errors:
        raise ArtifactResolutionError("Training-config resolution failed:\n" + "\n".join(errors))
    return resolved_paths, resolved_configs


def _effective_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Apply the same legacy defaults used by ``RealRobotPolicy``."""
    effective = dict(metadata)
    effective["observation_horizon"] = int(metadata.get("observation_horizon") or 1)
    effective["observation_dt_sec"] = float(metadata.get("observation_dt_sec") or 0.1)
    effective["state_joint_names"] = list(
        metadata.get("state_joint_names") or metadata.get("joint_names") or ()
    )
    effective["action_joint_names"] = list(
        metadata.get("action_joint_names") or metadata.get("joint_names") or ()
    )
    effective["state_dof"] = int(
        metadata.get("state_dof") or len(effective["state_joint_names"]) or metadata["dof"]
    )
    effective["action_dof"] = int(
        metadata.get("action_dof") or len(effective["action_joint_names"]) or metadata["dof"]
    )
    effective["camera_names"] = list(metadata.get("camera_names") or ("frontview", "wristview"))
    effective["normalization_type"] = metadata.get(
        "normalization_type",
        "min_max_-1_1" if "min" in metadata.get("normalization", {}) else "z_score",
    )
    effective["execution_horizon"] = int(
        metadata.get("execution_horizon") or metadata["horizon"]
    )
    return effective


def _values_equal(left: Any, right: Any) -> bool:
    left_is_sequence = isinstance(left, (list, tuple))
    right_is_sequence = isinstance(right, (list, tuple))
    if left_is_sequence != right_is_sequence:
        return False
    if left_is_sequence:
        return list(left) == list(right)
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def validate_artifact_compatibility(
    task_name: str,
    training_demos: int,
    metadata_by_method: dict[str, dict[str, Any]],
    configs_by_method: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    effective = {
        method: _effective_metadata(metadata)
        for method, metadata in metadata_by_method.items()
    }
    errors = []
    for method in METHOD_SPECS:
        metadata = effective[method]
        config = configs_by_method[method]
        if metadata.get("task") != task_name:
            errors.append(f"{method}: checkpoint task={metadata.get('task')!r}, requested {task_name!r}")
        if metadata.get("dataset_size") != training_demos:
            errors.append(
                f"{method}: checkpoint dataset_size={metadata.get('dataset_size')!r}, "
                f"requested {training_demos}"
            )
        if _method_for_model_type(metadata.get("model_type")) != method:
            errors.append(f"{method}: unexpected checkpoint model_type={metadata.get('model_type')!r}")
        for key in CONFIG_VALIDATION_KEYS:
            if key not in config or config[key] is None:
                continue
            checkpoint_value = metadata.get(key)
            if not _values_equal(checkpoint_value, config[key]):
                errors.append(
                    f"{method}: training config {key}={config[key]!r} does not match "
                    f"checkpoint/runtime {key}={checkpoint_value!r}"
                )

    reference_method = next(iter(METHOD_SPECS))
    reference = effective[reference_method]
    for method in list(METHOD_SPECS)[1:]:
        metadata = effective[method]
        for key in COMPARISON_METADATA_KEYS:
            if not _values_equal(metadata.get(key), reference.get(key)):
                errors.append(
                    f"{method} and {reference_method} differ in {key}: "
                    f"{metadata.get(key)!r} != {reference.get(key)!r}"
                )
        for normalization_key in ("state_normalization", "action_normalization"):
            left = metadata_by_method[method].get(
                normalization_key, metadata_by_method[method].get("normalization")
            )
            right = metadata_by_method[reference_method].get(
                normalization_key, metadata_by_method[reference_method].get("normalization")
            )
            if json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
                errors.append(
                    f"{method} and {reference_method} differ in {normalization_key}"
                )
    if errors:
        raise ValueError(
            "Checkpoint/config incompatibility prevents a fair comparison:\n  - "
            + "\n  - ".join(errors)
        )
    if reference["execution_horizon"] > reference["horizon"]:
        raise ValueError(
            f"execution_horizon={reference['execution_horizon']} exceeds "
            f"prediction horizon={reference['horizon']}"
        )
    return effective


def resolve_dataset(
    task_name: str,
    effective_metadata: dict[str, dict[str, Any]],
) -> tuple[Path, Path, str]:
    resolved = {}
    for method, metadata in effective_metadata.items():
        dataset_root = _absolute_path(
            metadata.get("dataset_root") or "Robot_real/real_dataset"
        )
        defaults = default_dataset_paths(task_name, dataset_root)
        camera_dir = _absolute_path(metadata.get("camera_dir") or defaults.camera_dir)
        trajectory_dir = _absolute_path(metadata.get("trajectory_dir") or defaults.trajectory_dir)
        resolved[method] = (camera_dir, trajectory_dir)
    unique = set(resolved.values())
    if len(unique) != 1:
        details = "\n".join(
            f"  - {method}: camera={paths[0]}, trajectory={paths[1]}"
            for method, paths in resolved.items()
        )
        raise ValueError(f"Checkpoints refer to different datasets:\n{details}")
    camera_dir, trajectory_dir = next(iter(unique))
    dataset_label = str(camera_dir.parent)
    return camera_dir, trajectory_dir, dataset_label


def evenly_spaced_indices(
    candidate_indices: Sequence[int],
    requested: int,
) -> tuple[list[int], str | None]:
    candidates = [int(index) for index in candidate_indices]
    if requested <= 0:
        raise ValueError("test_chunks must be positive")
    if not candidates:
        raise ValueError("No valid action-chunk start indices are available")
    actual = min(requested, len(candidates))
    positions = np.rint(np.linspace(0, len(candidates) - 1, num=actual)).astype(np.int64)
    unique_positions = np.unique(positions)
    if len(unique_positions) != actual:
        raise AssertionError("Even spacing unexpectedly produced duplicate positions")
    selected = [candidates[int(position)] for position in unique_positions]
    warning = None
    if actual < requested:
        warning = (
            f"Requested {requested} test chunks, but only {actual} valid points exist; "
            "using all valid points."
        )
    return selected, warning


def _observation_fingerprint(
    image_history: Sequence[Sequence[str | Path]],
    state_history: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for frame in image_history:
        for path in frame:
            digest.update(str(Path(path).resolve()).encode())
            digest.update(b"\0")
    digest.update(np.asarray(state_history, dtype=np.float32).tobytes())
    return digest.hexdigest()


def build_observation_samples(
    demo: Any,
    *,
    horizon: int,
    observation_horizon: int,
    image_size: int,
    requested: int,
) -> tuple[list[dict[str, Any]], str | None]:
    dataset = ActionChunkDataset(
        [demo],
        horizon=horizon,
        observation_horizon=observation_horizon,
        image_size=image_size,
    )
    candidates = [
        start
        for demo_index, start in dataset.samples
        if demo_index == 0 and start >= observation_horizon - 1
    ]
    selected, warning = evenly_spaced_indices(candidates, requested)
    samples = []
    for start in selected:
        observation_indices = dataset.observation_indices(start)
        image_history = [
            [str(path) for path in demo.image_paths[index]]
            for index in observation_indices
        ]
        state_history = demo.states[observation_indices].copy()
        samples.append(
            {
                "start": start,
                "observation_indices": observation_indices.tolist(),
                "images": image_history,
                "states": state_history,
                "fingerprint": _observation_fingerprint(image_history, state_history),
            }
        )
    return samples, warning


def compute_smoothness_metrics(
    chunks: np.ndarray,
    *,
    start_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    values = np.asarray(chunks, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected chunks shaped (N, H, J), got {values.shape}")
    if values.shape[0] == 0 or values.shape[1] < 3 or values.shape[2] == 0:
        raise ValueError("At least one chunk with horizon >= 3 and one joint is required")
    if start_indices is None:
        start_indices = list(range(values.shape[0]))
    if len(start_indices) != values.shape[0]:
        raise ValueError("start_indices length does not match chunk count")

    first = np.diff(values, axis=1)
    second = np.diff(values, n=2, axis=1)
    first_l2 = np.linalg.norm(first, axis=2)
    second_l2 = np.linalg.norm(second, axis=2)
    chunk_first_l2 = first_l2.mean(axis=1)
    chunk_second_l2 = second_l2.mean(axis=1)
    per_chunk = []
    for index, start in enumerate(start_indices):
        per_chunk.append(
            {
                "chunk_index": int(index),
                "start_index": int(start),
                "per_joint_mean_abs_first_difference": np.abs(first[index]).mean(axis=0).tolist(),
                "per_joint_mean_abs_second_difference": np.abs(second[index]).mean(axis=0).tolist(),
                "mean_first_difference_l2": float(chunk_first_l2[index]),
                "mean_second_difference_l2": float(chunk_second_l2[index]),
            }
        )
    return {
        "per_joint_mean_abs_first_difference": np.abs(first).mean(axis=(0, 1)).tolist(),
        "per_joint_mean_abs_second_difference": np.abs(second).mean(axis=(0, 1)).tolist(),
        "mean_first_difference_l2": float(first_l2.mean()),
        "std_chunk_first_difference_l2": float(chunk_first_l2.std(ddof=0)),
        "mean_second_difference_l2": float(second_l2.mean()),
        "std_chunk_second_difference_l2": float(chunk_second_l2.std(ddof=0)),
        "per_chunk": per_chunk,
    }


def _policy_chunks(
    checkpoint: Path,
    samples: Sequence[dict[str, Any]],
    *,
    device: str,
    seed: int,
) -> tuple[np.ndarray, RealRobotPolicy]:
    set_seed(seed)
    policy = RealRobotPolicy(checkpoint, device=device, seed=seed)
    policy.generator.manual_seed(seed)
    chunks = []
    with torch.no_grad():
        for sample in samples:
            chunks.append(
                policy.predict_action_chunk(sample["images"], sample["states"])
            )
    return np.stack(chunks, axis=0), policy


def _serializable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "task",
        "model_type",
        "dataset_size",
        "horizon",
        "observation_horizon",
        "observation_dt_sec",
        "image_size",
        "use_gripper",
        "camera_names",
        "state_joint_names",
        "action_joint_names",
        "joint_names",
        "state_dof",
        "action_dof",
        "dof",
        "normalization_type",
        "train_indices",
        "sampler_steps",
        "diffusion_steps",
        "diffusion_schedule",
        "diffusion_pred_type",
        "diffusion_eta",
        "clip_sample",
        "clip_sample_range",
        "execution_horizon",
    )
    return {key: metadata.get(key) for key in keys}


def print_summary(
    methods: dict[str, dict[str, Any]],
    joint_names: Sequence[str],
    section: str,
) -> None:
    title = section.replace("_", " ").title()
    print(f"\n{title}")
    print(f"{'Method':<12} {'Mean ||Δq||₂':>16} {'Mean ||Δ²q||₂':>18}")
    for method in METHOD_SPECS:
        metrics = methods[method][section]
        print(
            f"{method:<12} {metrics['mean_first_difference_l2']:>16.8f} "
            f"{metrics['mean_second_difference_l2']:>18.8f}"
        )
    for method in METHOD_SPECS:
        metrics = methods[method][section]
        print(f"\nMethod: {method}")
        print(f"{'Joint':<28} {'mean |Δq|':>16} {'mean |Δ²q|':>16}")
        for name, first, second in zip(
            joint_names,
            metrics["per_joint_mean_abs_first_difference"],
            metrics["per_joint_mean_abs_second_difference"],
        ):
            print(f"{name:<28} {first:>16.8f} {second:>16.8f}")


def run_comparison(args: argparse.Namespace) -> Path:
    checkpoints, raw_metadata = resolve_checkpoints(args.task_name, args.training_demos)
    config_paths, configs = resolve_training_configs(
        args.task_name, args.training_demos, checkpoints
    )
    effective = validate_artifact_compatibility(
        args.task_name, args.training_demos, raw_metadata, configs
    )
    camera_dir, trajectory_dir, dataset_path = resolve_dataset(args.task_name, effective)
    reference = effective["VanillaFM"]
    demos = load_aligned_demos(
        args.task_name,
        camera_dir=camera_dir,
        trajectory_dir=trajectory_dir,
        use_gripper=bool(reference["use_gripper"]),
        max_alignment_error_s=float(configs["VanillaFM"].get("max_alignment_error_s", 0.05)),
    )
    first_demo = demos[0]
    samples, selection_warning = build_observation_samples(
        first_demo,
        horizon=int(reference["horizon"]),
        observation_horizon=int(reference["observation_horizon"]),
        image_size=int(reference["image_size"]),
        requested=args.test_chunks,
    )
    warnings_record = []
    if selection_warning:
        warnings.warn(selection_warning, RuntimeWarning, stacklevel=2)
        warnings_record.append(selection_warning)

    methods = {}
    action_joint_names: list[str] | None = None
    for method in METHOD_SPECS:
        chunks, policy = _policy_chunks(
            checkpoints[method], samples, device=args.device, seed=args.seed
        )
        if action_joint_names is None:
            action_joint_names = list(policy.action_joint_names)
        elif list(policy.action_joint_names) != action_joint_names:
            raise ValueError(
                f"{method} exposes action joints {policy.action_joint_names}, "
                f"expected {action_joint_names}"
            )
        starts = [sample["start"] for sample in samples]
        execution_horizon = int(effective[method]["execution_horizon"])
        methods[method] = {
            "full_prediction": compute_smoothness_metrics(chunks, start_indices=starts),
            "executed_prefix": compute_smoothness_metrics(
                chunks[:, :execution_horizon], start_indices=starts
            ),
        }
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert action_joint_names is not None
    print_summary(methods, action_joint_names, "full_prediction")
    if reference["execution_horizon"] != reference["horizon"]:
        print_summary(methods, action_joint_names, "executed_prefix")

    timestamp = datetime.now(timezone.utc)
    result = {
        "timestamp": timestamp.isoformat(),
        "task_name": args.task_name,
        "training_demos": int(args.training_demos),
        "requested_test_chunks": int(args.test_chunks),
        "actual_test_chunks": len(samples),
        "sampling_seed": int(args.seed),
        "device": args.device,
        "dataset_path": dataset_path,
        "camera_dir": str(camera_dir),
        "trajectory_dir": str(trajectory_dir),
        "episode_key": first_demo.name,
        "selected_start_indices": [sample["start"] for sample in samples],
        "selected_observation_indices": [sample["observation_indices"] for sample in samples],
        "observation_fingerprints": [sample["fingerprint"] for sample in samples],
        "control_frequency_hz": 1.0 / float(reference["observation_dt_sec"]),
        "observation_horizon": int(reference["observation_horizon"]),
        "prediction_horizon": int(reference["horizon"]),
        "execution_horizon": int(reference["execution_horizon"]),
        "execution_horizon_source": (
            "checkpoint"
            if raw_metadata["VanillaFM"].get("execution_horizon") is not None
            else "no controller prefix configured; full prediction horizon used"
        ),
        "action_representation": "joint_position",
        "action_dimension": len(action_joint_names),
        "joint_names": action_joint_names,
        "checkpoints": {method: str(path) for method, path in checkpoints.items()},
        "training_configs": {method: str(path) for method, path in config_paths.items()},
        "checkpoint_metadata": {
            method: _serializable_metadata(metadata)
            for method, metadata in raw_metadata.items()
        },
        "warnings": warnings_record,
        "methods": methods,
    }
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_task = "".join(character if character.isalnum() else "_" for character in args.task_name)
    filename = (
        f"{safe_task}_demos{args.training_demos}_chunks{args.test_chunks}_"
        f"seed{args.seed}_{timestamp.strftime('%Y%m%dT%H%M%S')}.json"
    )
    output_path = output_dir / filename
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nSelected start indices: {result['selected_start_indices']}")
    print(f"Saved results to {output_path}")
    return output_path


def run_self_tests() -> None:
    constant = np.full((2, 5, 2), 3.0)
    constant_metrics = compute_smoothness_metrics(constant)
    assert constant_metrics["mean_first_difference_l2"] == 0.0
    assert constant_metrics["mean_second_difference_l2"] == 0.0

    time = np.arange(6, dtype=np.float64)
    linear = np.stack((time, 2.0 * time), axis=1)[None, ...]
    linear_metrics = compute_smoothness_metrics(linear)
    assert linear_metrics["mean_first_difference_l2"] > 0.0
    assert abs(linear_metrics["mean_second_difference_l2"]) < 1e-12

    alternating = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])[:, None][None, ...]
    alternating_metrics = compute_smoothness_metrics(alternating)
    assert alternating_metrics["mean_second_difference_l2"] > 0.0

    separated = np.asarray(
        [[[0.0], [1.0], [2.0]], [[100.0], [101.0], [102.0]]]
    )
    boundary_metrics = compute_smoothness_metrics(separated)
    assert math.isclose(boundary_metrics["mean_first_difference_l2"], 1.0)
    assert boundary_metrics["mean_second_difference_l2"] == 0.0

    selected, warning = evenly_spaced_indices(list(range(11)), 4)
    assert selected == [0, 3, 7, 10] and warning is None
    assert evenly_spaced_indices(list(range(11)), 4)[0] == selected

    stats = {
        "min": np.asarray([-2.0, 10.0], dtype=np.float32),
        "range": np.asarray([4.0, 20.0], dtype=np.float32),
    }
    normalized = np.asarray([[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]])
    physical = denormalize_joint_angles(normalized, stats)
    assert not math.isclose(
        compute_smoothness_metrics(normalized)["mean_first_difference_l2"],
        compute_smoothness_metrics(physical)["mean_first_difference_l2"],
    )

    def mocked_model(seed: int) -> np.ndarray:
        generator = np.random.default_rng(seed)
        return generator.normal(size=(3, 5, 2))

    first_output = mocked_model(17)
    second_output = mocked_model(17)
    assert np.array_equal(first_output, second_output)
    assert compute_smoothness_metrics(first_output) == compute_smoothness_metrics(second_output)

    shared_observations = [{"value": index} for index in range(3)]
    received = []
    for _method in METHOD_SPECS:
        received.append([id(observation) for observation in shared_observations])
    assert received[0] == received[1] == received[2]
    print("Smoothness self-tests passed (9 offline invariants; no robot initialized).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", "--task-name", choices=TASK_NAMES)
    parser.add_argument("--training_demos", "--training-demos", type=int)
    parser.add_argument("--test_chunks", "--test-chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline metric/selection/reproducibility checks without loading artifacts.",
    )
    args = parser.parse_args()
    if not args.self_test:
        if args.task_name is None:
            parser.error("--task_name is required")
        if args.training_demos is None:
            parser.error("--training_demos is required")
        if args.training_demos <= 0:
            parser.error("--training_demos must be positive")
        if args.test_chunks <= 0:
            parser.error("--test_chunks must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return
    try:
        run_comparison(args)
    except (ArtifactResolutionError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
