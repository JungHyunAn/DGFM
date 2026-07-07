"""Real-robot dataset discovery, timestamp alignment, and action chunking."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

TASK_NAMES = ("peg_in_hole", "sweep", "pick_and_place")
CAMERA_NAMES = ("frontview", "wristview")
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "real_dataset"


@dataclass(frozen=True)
class DatasetPaths:
    camera_dir: Path
    trajectory_dir: Path


@dataclass
class AlignedDemo:
    """One 10 Hz rollout aligned to the nearest recorded joint samples."""

    name: str
    camera_rollout: Path
    trajectory_rollout: Path
    timestamps_s: np.ndarray
    joint_names: tuple[str, ...]
    joints: np.ndarray
    image_paths: np.ndarray
    alignment_error_s: np.ndarray
    image_cache: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.timestamps_s)


def default_dataset_paths(
    task: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    camera_dir: str | Path | None = None,
    trajectory_dir: str | Path | None = None,
) -> DatasetPaths:
    """Map a supported task to its default camera and trajectory folders."""
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASK_NAMES}")
    root = Path(dataset_root).expanduser().resolve()
    camera = Path(camera_dir).expanduser().resolve() if camera_dir else root / f"{task}_camera"
    trajectory = (
        Path(trajectory_dir).expanduser().resolve()
        if trajectory_dir
        else root / f"{task}_trajectory"
    )
    return DatasetPaths(camera, trajectory)


def _rollout_directories(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    directories = sorted(item for item in path.iterdir() if item.is_dir())
    if not directories:
        raise ValueError(f"No timestamped rollout folders found in {path}")
    return directories


def match_rollout_directories(paths: DatasetPaths) -> list[tuple[Path, Path]]:
    """Check rollout counts and pair camera/trajectory runs in timestamp order."""
    camera_rollouts = _rollout_directories(paths.camera_dir)
    trajectory_rollouts = _rollout_directories(paths.trajectory_dir)
    if len(camera_rollouts) != len(trajectory_rollouts):
        raise ValueError(
            "Camera and trajectory rollout counts differ: "
            f"{len(camera_rollouts)} in {paths.camera_dir}, "
            f"{len(trajectory_rollouts)} in {paths.trajectory_dir}"
        )
    return list(zip(camera_rollouts, trajectory_rollouts))


def _read_camera_frames(camera_rollout: Path) -> tuple[np.ndarray, np.ndarray]:
    frames_csv = camera_rollout / "frames.csv"
    if not frames_csv.is_file():
        raise FileNotFoundError(f"Missing camera metadata: {frames_csv}")

    records: list[tuple[int, str, str]] = []
    with frames_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            timestamp_ns = int(row["timestamp_ns"])
            paths = [item.strip() for item in row["realsense_rgb"].split(";") if item.strip()]
            if len(paths) != 2:
                raise ValueError(
                    f"Expected front/wrist paths in {frames_csv}, got {row['realsense_rgb']!r}"
                )
            absolute_paths = [(camera_rollout / item).resolve() for item in paths]
            missing = [str(item) for item in absolute_paths if not item.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing camera image(s): {missing}")
            records.append((timestamp_ns, str(absolute_paths[0]), str(absolute_paths[1])))

    if not records:
        raise ValueError(f"No camera frames found in {frames_csv}")
    records.sort(key=lambda item: item[0])
    timestamps_s = np.asarray([item[0] for item in records], dtype=np.float64) / 1e9
    image_paths = np.asarray([[item[1], item[2]] for item in records], dtype=object)
    return timestamps_s, image_paths


def _read_joint_trajectory(
    trajectory_rollout: Path,
    use_gripper: bool,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    trajectory_candidates = (
        ("teleop_action_joint.csv", "joint_positions"),
        ("right_arm_joints.csv", "positions"),
    )
    trajectory_csv: Path | None = None
    position_column: str | None = None
    for filename, column in trajectory_candidates:
        candidate = trajectory_rollout / filename
        if candidate.is_file():
            trajectory_csv = candidate
            position_column = column
            break
    if trajectory_csv is None or position_column is None:
        expected = ", ".join(filename for filename, _ in trajectory_candidates)
        raise FileNotFoundError(
            f"Missing joint trajectory in {trajectory_rollout}; expected one of: {expected}"
        )

    timestamps: list[float] = []
    positions: list[list[float]] = []
    recorded_names: tuple[str, ...] | None = None
    with trajectory_csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            names = tuple(item.strip() for item in row["joint_names"].split(","))
            values = [float(item) for item in row[position_column].split(",")]
            if len(names) != len(values):
                raise ValueError(f"Joint name/value count differs in {trajectory_csv}")
            if recorded_names is None:
                recorded_names = names
            elif names != recorded_names:
                raise ValueError(f"Joint order changes within {trajectory_csv}")
            timestamps.append(float(row["time"]))
            positions.append(values)

    if not timestamps or recorded_names is None:
        raise ValueError(f"No joint samples found in {trajectory_csv}")
    keep = [i for i, name in enumerate(recorded_names) if use_gripper or "finger" not in name]
    if not keep:
        raise ValueError(f"No joints selected from {trajectory_csv}")
    selected_names = tuple(recorded_names[i] for i in keep)
    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    positions_array = np.asarray(positions, dtype=np.float32)[:, keep]
    order = np.argsort(timestamps_array)
    return timestamps_array[order], selected_names, positions_array[order]


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(reference, query, side="left")
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    choose_left = np.abs(reference[left] - query) <= np.abs(reference[right] - query)
    return np.where(choose_left, left, right)


def load_aligned_demo(
    camera_rollout: Path,
    trajectory_rollout: Path,
    *,
    use_gripper: bool = False,
    max_alignment_error_s: float = 0.05,
) -> AlignedDemo:
    """Trim to temporal overlap and align 10 Hz images to nearest joint samples."""
    camera_times, image_paths = _read_camera_frames(camera_rollout)
    joint_times, joint_names, joint_positions = _read_joint_trajectory(
        trajectory_rollout, use_gripper
    )

    overlap = (camera_times >= joint_times[0]) & (camera_times <= joint_times[-1])
    camera_times = camera_times[overlap]
    image_paths = image_paths[overlap]
    if len(camera_times) == 0:
        raise ValueError(
            f"No temporal overlap between {camera_rollout.name} and {trajectory_rollout.name}"
        )

    nearest = _nearest_indices(joint_times, camera_times)
    errors = np.abs(joint_times[nearest] - camera_times)
    if float(errors.max()) > max_alignment_error_s:
        raise ValueError(
            f"Timestamp mismatch in {camera_rollout.name}/{trajectory_rollout.name}: "
            f"maximum nearest-sample error is {errors.max():.6f}s "
            f"(limit {max_alignment_error_s:.6f}s)"
        )

    return AlignedDemo(
        name=camera_rollout.name,
        camera_rollout=camera_rollout,
        trajectory_rollout=trajectory_rollout,
        timestamps_s=camera_times,
        joint_names=joint_names,
        joints=joint_positions[nearest],
        image_paths=image_paths,
        alignment_error_s=errors,
    )


def load_aligned_demos(
    task: str,
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    camera_dir: str | Path | None = None,
    trajectory_dir: str | Path | None = None,
    use_gripper: bool = False,
    max_alignment_error_s: float = 0.05,
) -> list[AlignedDemo]:
    paths = default_dataset_paths(task, dataset_root, camera_dir, trajectory_dir)
    pairs = match_rollout_directories(paths)
    demos = [
        load_aligned_demo(
            camera,
            trajectory,
            use_gripper=use_gripper,
            max_alignment_error_s=max_alignment_error_s,
        )
        for camera, trajectory in pairs
    ]
    joint_orders = {demo.joint_names for demo in demos}
    if len(joint_orders) != 1:
        raise ValueError("Joint names/order differ across demonstrations")
    return demos


def uniform_demo_indices(total: int, requested: int | None) -> list[int]:
    """Uniform sparse selection; 50 total and 25 requested gives stride two."""
    if total <= 0:
        raise ValueError("total must be positive")
    if requested is None:
        return list(range(total))
    if requested <= 0 or requested > total:
        raise ValueError(f"dataset_size must be in [1, {total}], got {requested}")
    return [int(i * total / requested) for i in range(requested)]


def select_train_validation_demos(
    demos: Sequence[AlignedDemo],
    dataset_size: int | None,
    *,
    validation: bool = True,
    val_samples: int = 5,
) -> tuple[list[AlignedDemo], list[AlignedDemo], list[int], list[int]]:
    """Choose deterministic uniform train/validation indices from dataset sizes."""
    train_indices = uniform_demo_indices(len(demos), dataset_size)
    if not validation or val_samples <= 0:
        return [demos[i] for i in train_indices], [], train_indices, []

    train_index_set = set(train_indices)
    unused = [i for i in range(len(demos)) if i not in train_index_set]
    unused_count = min(val_samples, len(unused))
    unused_positions = (
        uniform_demo_indices(len(unused), unused_count) if unused_count else []
    )
    val_indices = [unused[position] for position in unused_positions]
    needed = val_samples - len(val_indices)
    if needed:
        if needed > len(train_indices):
            raise ValueError(
                f"val_samples={val_samples} cannot be filled from {len(unused)} unused "
                f"and {len(train_indices)} training demonstrations without replacement"
            )
        train_positions = uniform_demo_indices(len(train_indices), needed)
        val_indices.extend(train_indices[position] for position in train_positions)
    return (
        [demos[i] for i in train_indices],
        [demos[i] for i in val_indices],
        train_indices,
        val_indices,
    )


def compute_joint_stats(demos: Sequence[AlignedDemo]) -> dict[str, np.ndarray]:
    """Fit the same per-joint [-1, 1] statistics used by simulation training."""
    if not demos:
        raise ValueError("At least one demonstration is required for normalization")
    joints = np.concatenate([demo.joints for demo in demos], axis=0).astype(np.float32)
    joint_min = joints.min(axis=0).astype(np.float32)
    joint_max = joints.max(axis=0).astype(np.float32)
    joint_range = (joint_max - joint_min).astype(np.float32)
    joint_range = np.where(joint_range < 1e-8, 1.0, joint_range).astype(np.float32)
    return {"min": joint_min, "max": joint_max, "range": joint_range}


def normalize_joint_angles(
    joints: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    """Normalize joint angles to [-1, 1], with legacy z-score support."""
    joints = np.asarray(joints, dtype=np.float32)
    if "min" in stats:
        normalized = (2.0 / stats["range"]) * (joints - stats["min"]) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    return ((joints - stats["mean"]) / stats["std"]).astype(np.float32)


def denormalize_joint_angles(
    joints: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    """Restore normalized model outputs to physical joint angles."""
    joints = np.asarray(joints, dtype=np.float32)
    if "min" in stats:
        return ((joints + 1.0) * 0.5 * stats["range"] + stats["min"]).astype(np.float32)
    return (joints * stats["std"] + stats["mean"]).astype(np.float32)


def _load_resized_image_pair(paths: np.ndarray, image_size: int) -> np.ndarray:
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        images.append(image)
    return np.stack(images, axis=0).copy()


def cache_demo_images(
    demos: Sequence[AlignedDemo],
    image_size: int = 224,
    *,
    workers: int = 8,
) -> None:
    """Decode and resize every unique observation image once into CPU RAM."""
    if workers <= 0:
        raise ValueError("cache_workers must be positive")
    uncached = [demo for demo in demos if image_size not in demo.image_cache]
    if not uncached:
        return

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for demo in tqdm(uncached, desc=f"Caching {image_size}x{image_size} RGB", unit="demo"):
            pairs = list(
                executor.map(
                    lambda paths: _load_resized_image_pair(paths, image_size),
                    demo.image_paths,
                )
            )
            cached = torch.from_numpy(np.stack(pairs, axis=0))
            demo.image_cache[image_size] = cached
            total_bytes += cached.numel() * cached.element_size()
    tqdm.write(
        f"Cached {sum(len(demo) for demo in uncached)} frames "
        f"({total_bytes / 1024**3:.2f} GiB uint8) in CPU RAM."
    )


class ActionChunkDataset(Dataset):
    """Current observation paired with the next fixed-horizon joint targets."""

    def __init__(
        self,
        demos: Sequence[AlignedDemo],
        *,
        horizon: int = 16,
        image_size: int = 224,
        normalization_stats: dict[str, np.ndarray] | None = None,
        cache_images: bool = False,
        cache_workers: int = 8,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.demos = list(demos)
        self.horizon = int(horizon)
        self.image_size = int(image_size)
        self.normalization_stats = normalization_stats
        if cache_images:
            cache_demo_images(self.demos, self.image_size, workers=cache_workers)
        self.samples: list[tuple[int, int]] = []
        for demo_index, demo in enumerate(self.demos):
            self.samples.extend(
                (demo_index, start) for start in range(max(0, len(demo) - self.horizon))
            )
        if not self.samples:
            raise ValueError(f"No demonstrations contain more than {horizon} aligned frames")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        demo_index, start = self.samples[index]
        demo = self.demos[demo_index]
        state = demo.joints[start].copy()
        actions = demo.joints[start + 1 : start + 1 + self.horizon].copy()
        if self.normalization_stats is not None:
            state = normalize_joint_angles(state, self.normalization_stats)
            actions = normalize_joint_angles(actions, self.normalization_stats)
        cached_images = demo.image_cache.get(self.image_size)
        images = (
            cached_images[start]
            if cached_images is not None
            else torch.from_numpy(_load_resized_image_pair(demo.image_paths[start], self.image_size))
        )
        return {
            "sample_index": index,
            "images": images,
            "joint_state": torch.from_numpy(state.astype(np.float32)),
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "demo_index": demo_index,
            "demo_name": demo.name,
            "start": start,
        }


def demo_subset_dataset(
    demo: AlignedDemo,
    *,
    horizon: int,
    image_size: int,
    normalization_stats: dict[str, np.ndarray],
    max_steps: int | None = None,
    cache_images: bool = False,
    cache_workers: int = 8,
) -> ActionChunkDataset:
    dataset = ActionChunkDataset(
        [demo],
        horizon=horizon,
        image_size=image_size,
        normalization_stats=normalization_stats,
        cache_images=cache_images,
        cache_workers=cache_workers,
    )
    if max_steps is not None and len(dataset.samples) > max_steps:
        indices = uniform_demo_indices(len(dataset.samples), max_steps)
        dataset.samples = [dataset.samples[i] for i in indices]
    return dataset
