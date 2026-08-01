"""Train a vision-conditioned flow-matching or diffusion policy on real data."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Robot_real.dataset_utils import (
    CAMERA_NAMES,
    DEFAULT_DATASET_ROOT,
    TELEOP_ACTION_GRIPPER_COLUMN,
    TELEOP_STATE_GRIPPER_COLUMN,
    ActionChunkDataset,
    compute_joint_stats,
    demo_subset_dataset,
    load_aligned_demos,
    normalize_joint_angles,
    select_train_validation_demos,
)
from Robot_simulation.models.DP_class import DiffusionSchedule, run_diffusion
from Robot_simulation.models.DGFM_class import cluster_points_x
from Robot_simulation.models.DGFMv2_class import (
    DGFMv2,
    MixtureSamplerV2,
    compute_cluster_pca_fast_x_only,
)
from Robot_simulation.models.VanillaFM_class import EMAModel, VectorField
from Robot_simulation.models.vision_encoder import FrozenResNet18Encoder

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "real_config" / "peg_in_hole_uniformfm_50.json"
)
MODEL_TYPES = ("UniformFM", "DGFMv2", "DiffusionPolicy")
MODEL_TYPE_ALIASES = {"fm": "UniformFM", "diffusion": "DiffusionPolicy", "DP": "DiffusionPolicy"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open() as stream:
        config = json.load(stream)
    config["config_path"] = str(config_path)
    return config


def build_models(
    *,
    horizon: int,
    dof: int,
    state_dof: int | None = None,
    observation_horizon: int = 1,
    feature_proj_dim: int,
    condition_embed_dim: int,
    num_convs_per_block: int,
    pretrained_vision: bool,
    vision_finetune_mode: str,
    device: torch.device,
    vision_train_bn: bool = False,
    vision_pool: str = "avg",
    vision_spatial_softmax_temperature: float = 1.0,
    vision_feature_norm: str = "none",
    vision_augmentation: bool = False,
    vision_random_shift: int = 4,
    vision_color_jitter: float = 0.1,
) -> tuple[VectorField, FrozenResNet18Encoder]:
    if observation_horizon <= 0:
        raise ValueError("observation_horizon must be positive")
    encoder = FrozenResNet18Encoder(
        camera_names=CAMERA_NAMES,
        pretrained=pretrained_vision,
        finetune_mode=vision_finetune_mode,
        train_bn=vision_train_bn,
        pool=vision_pool,
        spatial_softmax_temperature=vision_spatial_softmax_temperature,
        feature_proj_dim=feature_proj_dim,
        feature_norm=vision_feature_norm,
        augmentation=vision_augmentation,
        random_shift=vision_random_shift,
        color_jitter=vision_color_jitter,
    ).to(device)
    condition_dim = observation_horizon * (
        (dof if state_dof is None else state_dof) + encoder.output_dim
    )
    model = VectorField(
        seq_len=horizon,
        dof=dof,
        param_len=condition_dim,
        gripper_idx=list(range(7, dof)) if dof > 7 else [],
        observation_type="vision",
        condition_embed_dim=condition_embed_dim,
        num_convs_per_block=num_convs_per_block,
    ).to(device)
    return model, encoder


def make_condition(
    encoder: FrozenResNet18Encoder,
    images: torch.Tensor,
    joint_state: torch.Tensor,
    device: torch.device,
    vision_batch_size: int | None = None,
) -> torch.Tensor:
    if images.ndim == 5:
        images = images.unsqueeze(1)
    if joint_state.ndim == 2:
        joint_state = joint_state.unsqueeze(1)
    if images.ndim != 6 or joint_state.ndim != 3:
        raise ValueError(
            "Expected images (B, O, V, H, W, C) and states (B, O, D), "
            f"got {tuple(images.shape)} and {tuple(joint_state.shape)}"
        )
    if images.shape[:2] != joint_state.shape[:2]:
        raise ValueError(
            f"Image/state history shape mismatch: {images.shape[:2]} "
            f"vs {joint_state.shape[:2]}"
        )
    batch_size, observation_horizon = images.shape[:2]
    flat_images = images.reshape(
        batch_size * observation_horizon, *images.shape[2:]
    )
    flat_images = flat_images.to(device, non_blocking=True)
    if vision_batch_size and flat_images.shape[0] > vision_batch_size:
        image_features = torch.cat(
            [encoder(chunk) for chunk in flat_images.split(vision_batch_size)], dim=0
        )
    else:
        image_features = encoder(flat_images)
    state_history = joint_state.to(device, non_blocking=True).reshape(batch_size, -1)
    image_history = image_features.reshape(batch_size, -1)
    return torch.cat([state_history, image_history], dim=-1)


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_scale: float = 0.05,
):
    """Match the linear-warmup cosine scheduler used by run_eval.py."""
    def lr_scale(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


@torch.no_grad()
def sample_action_chunks(
    model: VectorField,
    conditions: torch.Tensor,
    *,
    model_type: str,
    horizon: int,
    dof: int,
    sampler_steps: int = 100,
    diffusion_steps: int = 100,
    diffusion_schedule: str = "cosine",
    diffusion_pred_type: str = "x0",
    diffusion_eta: float = 0.0,
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate chunks with UniformFM/DGFMv2 flow or DiffusionPolicy sampling."""
    model_type = MODEL_TYPE_ALIASES.get(model_type, model_type)
    if sampler_steps <= 0:
        raise ValueError("sampler_steps must be positive")
    if model_type == "DiffusionPolicy" and sampler_steps < 2:
        raise ValueError("DiffusionPolicy requires sampler_steps >= 2")
    batch = conditions.shape[0]
    noise = torch.randn(
        batch,
        horizon,
        dof,
        device=conditions.device,
        dtype=conditions.dtype,
        generator=generator,
    )
    if model_type in ("UniformFM", "DGFMv2"):
        sample = noise
        dt = 1.0 / sampler_steps
        time = torch.empty((batch, 1), device=conditions.device, dtype=conditions.dtype)
        for step in range(sampler_steps):
            time.fill_(step * dt)
            sample.add_(model(sample, time, conditions), alpha=dt)
        return sample
    if model_type == "DiffusionPolicy":
        return run_diffusion(
            model,
            noise,
            conditions,
            str(conditions.device),
            T_diff=diffusion_steps,
            schedule_type=diffusion_schedule,
            ddim_steps=min(sampler_steps, diffusion_steps),
            eta=diffusion_eta,
            pred_type=diffusion_pred_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")


def _training_loss(
    model: VectorField,
    actions: torch.Tensor,
    conditions: torch.Tensor,
    *,
    model_type: str,
    n_t: int,
    diffusion_schedule: DiffusionSchedule | None,
    diffusion_steps: int,
    diffusion_pred_type: str,
) -> torch.Tensor:
    batch, horizon, dof = actions.shape
    expanded_actions = (
        actions.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, horizon, dof)
    )
    expanded_conditions = (
        conditions.unsqueeze(1)
        .expand(-1, n_t, -1)
        .reshape(-1, conditions.shape[-1])
    )

    if model_type == "UniformFM":
        source = torch.randn_like(actions)
        source = source.unsqueeze(1).expand(-1, n_t, -1, -1).reshape_as(expanded_actions)
        time = torch.rand(batch * n_t, 1, device=actions.device)
        interpolated = (1.0 - time[:, None]) * source + time[:, None] * expanded_actions
        prediction = model(interpolated, time, expanded_conditions)
        target = expanded_actions - source
    elif model_type == "DiffusionPolicy":
        if diffusion_schedule is None:
            raise RuntimeError("Diffusion schedule was not initialized")
        indices = torch.randint(0, diffusion_steps, (batch * n_t,), device=actions.device)
        noise = torch.randn_like(expanded_actions)
        alpha_bar = diffusion_schedule._gather(
            diffusion_schedule.alphas_bar, indices, expanded_actions.shape
        )
        noisy_actions = (
            torch.sqrt(alpha_bar) * expanded_actions
            + torch.sqrt(1.0 - alpha_bar) * noise
        )
        time = ((indices.float() + 1.0) / diffusion_steps).unsqueeze(-1)
        prediction = model(noisy_actions, time, expanded_conditions)
        target = expanded_actions if diffusion_pred_type == "x0" else noise
    else:
        raise ValueError(f"_training_loss does not handle {model_type!r}")
    return torch.mean((prediction - target) ** 2)


def _collect_dgfmv2_arrays(
    dataset: ActionChunkDataset,
    state_stats: dict[str, np.ndarray],
    action_stats: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect normalized chunks/state conditions without decoding images."""
    actions = []
    states = []
    for demo_index, start in dataset.samples:
        demo = dataset.demos[demo_index]
        state_history = demo.states[dataset.observation_indices(start)]
        states.append(
            normalize_joint_angles(state_history, state_stats).reshape(-1)
        )
        chunk = demo.actions[start : start + dataset.horizon]
        actions.append(normalize_joint_angles(chunk, action_stats))
    return (
        torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        torch.from_numpy(np.asarray(states, dtype=np.float32)),
    )


def _fit_dgfmv2(
    dataset: ActionChunkDataset,
    state_stats: dict[str, np.ndarray],
    action_stats: dict[str, np.ndarray],
    model: VectorField,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict[str, Any],
    device: torch.device,
):
    """Fit the simulation DGFMv2 X-only cluster/PCA intermediate distribution."""
    actions, states = _collect_dgfmv2_arrays(dataset, state_stats, action_stats)
    flat_actions = actions.numpy().reshape(len(actions), -1)
    state_conditions = states.numpy()
    cluster_size = config.get("cluster_size")
    if cluster_size is None:
        cluster_size = max(2, len(actions) // int(config["cluster_partition"]))
    if cluster_size < 2:
        raise ValueError("DGFMv2 cluster_size must be at least two")

    print(
        f"Fitting DGFMv2 X-only mixture: windows={len(actions)} "
        f"cluster_size={cluster_size}"
    )
    clusters, inverse_clusters = cluster_points_x(
        flat_actions,
        m=cluster_size,
        jaccard_thresh=float(config["cluster_jaccard_thresh"]),
        merge_k=int(config["cluster_merge_k"]),
        standardize=bool(config["cluster_standardize"]),
        scale_x=float(config["cluster_scale_x"]),
    )
    cluster_sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.float64)
    mu_x, basis, covariance, weights, c_mean, c_std = compute_cluster_pca_fast_x_only(
        flat_actions,
        state_conditions,
        clusters,
        eps=float(config["cluster_eps"]),
        outlier_q=float(config["cluster_outlier_q"]),
        max_pca_samples=int(config["max_pca_samples"]),
        n_jobs=int(config["pca_n_jobs"]),
    )
    mixture = MixtureSamplerV2(
        mu_x,
        basis,
        covariance,
        weights,
        c_mean,
        c_std,
        device=str(device),
        reg=float(config["mixture_reg"]),
        orth_sigma=float(config["mixture_orth_sigma"]),
    )
    policy = DGFMv2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        task_name=config["task"],
        horizon=config["horizon"],
        dof=actions.shape[-1],
        condition_dim=model.param_len,
        gripper_idx=list(range(7, actions.shape[-1])) if actions.shape[-1] > 7 else [],
        device=str(device),
        use_ema=False,
    )
    return policy, mixture, inverse_clusters, cluster_sizes, len(clusters), cluster_size


def _dgfmv2_training_loss(
    model: VectorField,
    policy: DGFMv2,
    mixture: MixtureSamplerV2,
    inverse_clusters,
    cluster_sizes: np.ndarray,
    sample_indices: torch.Tensor,
    actions: torch.Tensor,
    conditions: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    """Build noise→DGFMv2 mixture→data interpolants for one mini-batch."""
    batch, horizon, dof = actions.shape
    n_t = int(config["n_t"])
    cluster_ids = policy._sample_covering_clusters(
        sample_indices.to(actions.device), inverse_clusters, cluster_sizes
    )
    intermediate = mixture.sample_cond(
        conditions,
        truncated=bool(config["dgfm_truncated"]),
        trunc=(float(config["dgfm_trunc_low"]), float(config["dgfm_trunc_high"])),
        pis=cluster_ids,
    ).reshape(batch, horizon, dof)
    source = torch.randn_like(actions)
    time = policy.sample_t(batch * n_t)
    actions_r = actions.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, horizon, dof)
    intermediate_r = (
        intermediate.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, horizon, dof)
    )
    source_r = source.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, horizon, dof)
    conditions_r = (
        conditions.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, conditions.shape[-1])
    )
    a, b, c, a_dot, b_dot, c_dot = policy._path_weights(
        time, config["interpolation_path"]
    )
    interpolated = (
        a[:, None] * source_r + b[:, None] * intermediate_r + c[:, None] * actions_r
    )
    target = (
        a_dot[:, None] * source_r
        + b_dot[:, None] * intermediate_r
        + c_dot[:, None] * actions_r
    )
    prediction = model(interpolated, time, conditions_r)
    return torch.mean((prediction - target) ** 2)


def _denormalize_joint_tensor(
    joints: torch.Tensor,
    stats: dict[str, np.ndarray],
) -> torch.Tensor:
    """Denormalize on-device, retaining compatibility with older checkpoints."""
    if "min" in stats:
        joint_min = torch.as_tensor(stats["min"], device=joints.device).view(1, 1, -1)
        joint_range = torch.as_tensor(stats["range"], device=joints.device).view(1, 1, -1)
        return (joints + 1.0) * 0.5 * joint_range + joint_min
    mean = torch.as_tensor(stats["mean"], device=joints.device).view(1, 1, -1)
    std = torch.as_tensor(stats["std"], device=joints.device).view(1, 1, -1)
    return joints * std + mean


@torch.no_grad()
def validate_offline(
    model: VectorField,
    encoder: FrozenResNet18Encoder,
    demos,
    state_stats: dict[str, np.ndarray],
    action_stats: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Compare generated chunks with held-out demonstration chunks in joint units."""
    model.eval()
    encoder.eval()
    action_dim = len(
        action_stats["min"] if "min" in action_stats else action_stats["mean"]
    )
    per_demo: dict[str, float] = {}

    for demo_index, demo in enumerate(demos):
        dataset = demo_subset_dataset(
            demo,
            horizon=config["horizon"],
            image_size=config["image_size"],
            state_normalization_stats=state_stats,
            action_normalization_stats=action_stats,
            observation_horizon=int(config.get("observation_horizon", 1)),
            max_steps=config.get("validation_steps_per_demo", 100),
            cache_images=config.get("cache_images", False),
            cache_workers=config.get("cache_workers", 8),
        )
        loader = DataLoader(
            dataset,
            batch_size=config.get("validation_batch_size", config["batch_size"]),
            shuffle=False,
            num_workers=config.get("num_workers", 0),
            pin_memory=device.type == "cuda",
        )
        squared_error = 0.0
        element_count = 0
        generator = torch.Generator(device=device)
        generator.manual_seed(config["seed"] + demo_index)
        for batch in loader:
            conditions = make_condition(
                encoder,
                batch["images"],
                batch["joint_state"],
                device,
                config.get("vision_batch_size"),
            )
            predictions = sample_action_chunks(
                model,
                conditions,
                model_type=config["model_type"],
                horizon=config["horizon"],
                dof=action_dim,
                sampler_steps=config["sampler_steps"],
                diffusion_steps=config["diffusion_steps"],
                diffusion_schedule=config["diffusion_schedule"],
                diffusion_pred_type=config["diffusion_pred_type"],
                diffusion_eta=config["diffusion_eta"],
                clip_sample=config["clip_sample"],
                clip_sample_range=config["clip_sample_range"],
                generator=generator,
            )
            targets = batch["actions"].to(device)
            error = (
                _denormalize_joint_tensor(predictions, action_stats)
                - _denormalize_joint_tensor(targets, action_stats)
            ) ** 2
            squared_error += float(error.sum())
            element_count += error.numel()
        per_demo[demo.name] = squared_error / max(1, element_count)
    return float(np.mean(list(per_demo.values()))), per_demo


def _checkpoint_payload(
    model_state: dict[str, torch.Tensor],
    encoder_state: dict[str, torch.Tensor],
    config: dict[str, Any],
    state_stats: dict[str, np.ndarray],
    action_stats: dict[str, np.ndarray],
    train_indices: list[int],
    val_indices: list[int],
    state_joint_names: tuple[str, ...],
    action_joint_names: tuple[str, ...],
    condition_dim: int,
    epoch: int,
    validation_mse: float | None,
) -> dict[str, Any]:
    metadata_keys = (
        "task",
        "model_type",
        "horizon",
        "observation_horizon",
        "observation_dt_sec",
        "image_size",
        "use_gripper",
        "feature_proj_dim",
        "condition_embed_dim",
        "num_convs_per_block",
        "vision_finetune_mode",
        "vision_train_bn",
        "vision_pool",
        "vision_spatial_softmax_temperature",
        "vision_feature_norm",
        "vision_augmentation",
        "vision_random_shift",
        "vision_color_jitter",
        "vision_batch_size",
        "vision_encoder_lr_scale",
        "vision_projection_lr_scale",
        "warmup_epochs",
        "sampler_steps",
        "diffusion_steps",
        "diffusion_schedule",
        "diffusion_pred_type",
        "diffusion_eta",
        "clip_sample",
        "clip_sample_range",
        "seed",
        "dataset_root",
        "camera_dir",
        "trajectory_dir",
        "dataset_size",
        "val_samples",
        "interpolation_path",
        "cluster_partition",
        "cluster_size",
        "cluster_jaccard_thresh",
        "cluster_merge_k",
        "cluster_standardize",
        "cluster_scale_x",
        "cluster_eps",
        "cluster_outlier_q",
        "max_pca_samples",
        "pca_n_jobs",
        "mixture_reg",
        "mixture_orth_sigma",
        "dgfm_truncated",
        "dgfm_trunc_low",
        "dgfm_trunc_high",
        "resolved_cluster_count",
        "resolved_cluster_size",
    )
    metadata = {key: config.get(key) for key in metadata_keys}
    metadata.update(
        use_ema=bool(config.get("use_ema", False)),
        ema_includes_vision_encoder=bool(config.get("use_ema", False)),
        dof=len(action_joint_names),
        state_dof=len(state_joint_names),
        action_dof=len(action_joint_names),
        condition_dim=condition_dim,
        camera_names=list(CAMERA_NAMES),
        joint_names=list(action_joint_names),
        state_joint_names=list(state_joint_names),
        action_joint_names=list(action_joint_names),
        train_indices=train_indices,
        val_indices=val_indices,
        normalization={key: value.tolist() for key, value in action_stats.items()},
        state_normalization={
            key: value.tolist() for key, value in state_stats.items()
        },
        action_normalization={
            key: value.tolist() for key, value in action_stats.items()
        },
        normalization_type=(
            "min_max_-1_1" if "min" in action_stats else "z_score"
        ),
        gripper_state_source=(
            TELEOP_STATE_GRIPPER_COLUMN if config.get("use_gripper") else None
        ),
        gripper_action_source=(
            TELEOP_ACTION_GRIPPER_COLUMN if config.get("use_gripper") else None
        ),
    )
    return {
        "model_state_dict": model_state,
        "vision_encoder_state_dict": encoder_state,
        "metadata": metadata,
        "epoch": epoch,
        "validation_mse": validation_mse,
    }


def train(config: dict[str, Any]) -> Path:
    config["model_type"] = MODEL_TYPE_ALIASES.get(config["model_type"], config["model_type"])
    if config["model_type"] not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {config['model_type']!r}")
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    demos = load_aligned_demos(
        config["task"],
        dataset_root=config["dataset_root"],
        camera_dir=config.get("camera_dir"),
        trajectory_dir=config.get("trajectory_dir"),
        use_gripper=config["use_gripper"],
        max_alignment_error_s=config["max_alignment_error_s"],
    )
    train_demos, val_demos, train_indices, val_indices = select_train_validation_demos(
        demos,
        config.get("dataset_size"),
        validation=config["validation"],
        val_samples=config["val_samples"],
    )
    print(
        f"Dataset split: train_indices={train_indices} val_indices={val_indices} "
        f"val_demos={[demo.name for demo in val_demos]}"
    )
    state_stats = compute_joint_stats(train_demos, source="state")
    action_stats = compute_joint_stats(train_demos, source="action")
    train_dataset = ActionChunkDataset(
        train_demos,
        horizon=config["horizon"],
        image_size=config["image_size"],
        state_normalization_stats=state_stats,
        action_normalization_stats=action_stats,
        observation_horizon=int(config.get("observation_horizon", 1)),
        cache_images=config.get("cache_images", False),
        cache_workers=config.get("cache_workers", 8),
    )
    loader_generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=loader_generator,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=(
            config.get("prefetch_factor", 2)
            if config["num_workers"] > 0
            else None
        ),
    )

    state_dof = len(train_demos[0].state_joint_names)
    action_dof = len(train_demos[0].action_joint_names)
    model, encoder = build_models(
        horizon=config["horizon"],
        dof=action_dof,
        state_dof=state_dof,
        observation_horizon=int(config.get("observation_horizon", 1)),
        feature_proj_dim=config["feature_proj_dim"],
        condition_embed_dim=config["condition_embed_dim"],
        num_convs_per_block=config["num_convs_per_block"],
        pretrained_vision=config["pretrained_vision"],
        vision_finetune_mode=config["vision_finetune_mode"],
        device=device,
        vision_train_bn=config["vision_train_bn"],
        vision_pool=config["vision_pool"],
        vision_spatial_softmax_temperature=config["vision_spatial_softmax_temperature"],
        vision_feature_norm=config["vision_feature_norm"],
        vision_augmentation=config["vision_augmentation"],
        vision_random_shift=config["vision_random_shift"],
        vision_color_jitter=config["vision_color_jitter"],
    )
    policy_parameters = list(model.parameters())
    projection_parameters = list(encoder.projection_parameters())
    backbone_parameters = list(encoder.backbone_parameters())
    parameter_groups = [{"params": policy_parameters, "lr": config["learning_rate"]}]
    if projection_parameters:
        parameter_groups.append({
            "params": projection_parameters,
            "lr": config["learning_rate"] * config["vision_projection_lr_scale"],
        })
    if backbone_parameters:
        parameter_groups.append({
            "params": backbone_parameters,
            "lr": config["learning_rate"] * config["vision_encoder_lr_scale"],
        })
    parameters = policy_parameters + projection_parameters + backbone_parameters
    optimizer = torch.optim.Adam(
        parameter_groups, weight_decay=config["weight_decay"]
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=config["warmup_epochs"],
        total_epochs=config["epochs"],
    )
    diffusion_schedule = None
    if config["model_type"] == "DiffusionPolicy":
        diffusion_schedule = DiffusionSchedule(
            config["diffusion_steps"], device, config["diffusion_schedule"]
        )

    dgfm_policy = None
    dgfm_mixture = None
    inverse_clusters = None
    cluster_sizes = None
    dgfm_cluster_count = None
    dgfm_cluster_size = None
    if config["model_type"] == "DGFMv2":
        (
            dgfm_policy,
            dgfm_mixture,
            inverse_clusters,
            cluster_sizes,
            dgfm_cluster_count,
            dgfm_cluster_size,
        ) = _fit_dgfmv2(
            train_dataset,
            state_stats,
            action_stats,
            model,
            optimizer,
            scheduler,
            config,
            device,
        )

    ema = EMAModel(model) if config["use_ema"] else None
    encoder_ema = EMAModel(encoder) if config["use_ema"] else None
    last_validation_mse: float | None = None
    history: list[dict[str, Any]] = []

    print(
        f"Task={config['task']} model={config['model_type']} device={device} "
        f"demos={len(train_demos)}/{len(demos)} val_demos={len(val_demos)} "
        f"training_action_chunks={len(train_dataset)} "
        f"state_history_shape=({train_dataset.observation_horizon}, {state_dof}) "
        f"action_chunk_shape=({config['horizon']}, {action_dof})"
    )
    epoch_progress = tqdm(
        range(1, config["epochs"] + 1),
        desc=f"{config['model_type']} Training",
        unit="epoch",
    )
    for epoch in epoch_progress:
        model.train()
        encoder.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            actions = batch["actions"].to(device, non_blocking=True)
            conditions = make_condition(
                encoder,
                batch["images"],
                batch["joint_state"],
                device,
                config.get("vision_batch_size"),
            )
            if config["model_type"] == "DGFMv2":
                loss = _dgfmv2_training_loss(
                    model,
                    dgfm_policy,
                    dgfm_mixture,
                    inverse_clusters,
                    cluster_sizes,
                    batch["sample_index"],
                    actions,
                    conditions,
                    config,
                )
            else:
                loss = _training_loss(
                    model,
                    actions,
                    conditions,
                    model_type=config["model_type"],
                    n_t=config["n_t"],
                    diffusion_schedule=diffusion_schedule,
                    diffusion_steps=config["diffusion_steps"],
                    diffusion_pred_type=config["diffusion_pred_type"],
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config["gradient_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip_norm"])
            optimizer.step()
            if ema is not None:
                ema.step(model)
                encoder_ema.step(encoder)
            total_loss += float(loss)
            batches += 1
        scheduler.step()

        train_loss = total_loss / max(1, batches)
        record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
        should_validate = bool(val_demos) and (
            epoch % config["validation_period"] == 0 or epoch == config["epochs"]
        )
        if should_validate:
            evaluation_model = ema.averaged_model if ema is not None else model
            evaluation_encoder = (
                encoder_ema.averaged_model if encoder_ema is not None else encoder
            )
            val_mse, per_demo = validate_offline(
                evaluation_model,
                evaluation_encoder,
                val_demos,
                state_stats,
                action_stats,
                config,
                device,
            )
            record.update(validation_mse=val_mse, validation_by_demo=per_demo)
            last_validation_mse = val_mse
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.6f}",
                val_mse=f"{val_mse:.8f}",
            )
            epoch_progress.write(
                f"Epoch {epoch}/{config['epochs']} validation: "
                f"MSE={val_mse:.8f} | "
                + ", ".join(
                    f"{demo_name}={demo_mse:.8f}"
                    for demo_name, demo_mse in per_demo.items()
                )
            )
        else:
            epoch_progress.set_postfix(train_loss=f"{train_loss:.6f}")
        history.append(record)

    final_model = ema.averaged_model if ema is not None else model
    final_encoder = encoder_ema.averaged_model if encoder_ema is not None else encoder
    final_model_state = copy.deepcopy(final_model.state_dict())
    final_encoder_state = copy.deepcopy(final_encoder.state_dict())
    checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if config["model_type"] == "DGFMv2":
        config["resolved_cluster_count"] = dgfm_cluster_count
        config["resolved_cluster_size"] = dgfm_cluster_size
    payload = _checkpoint_payload(
        final_model_state,
        final_encoder_state,
        config,
        state_stats,
        action_stats,
        train_indices,
        val_indices,
        train_demos[0].state_joint_names,
        train_demos[0].action_joint_names,
        model.param_len,
        config["epochs"],
        last_validation_mse,
    )
    torch.save(payload, checkpoint_path)
    history_path = checkpoint_path.with_suffix(".history.json")
    with history_path.open("w") as stream:
        json.dump(history, stream, indent=2)
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", choices=("peg_in_hole", "sweep", "pick_and_place"))
    parser.add_argument("--camera-dir", type=Path)
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--model-type", choices=MODEL_TYPES)
    parser.add_argument("--use-gripper", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for argument, key in (
        (args.task, "task"),
        (args.camera_dir, "camera_dir"),
        (args.trajectory_dir, "trajectory_dir"),
        (args.dataset_size, "dataset_size"),
        (args.model_type, "model_type"),
        (args.use_gripper, "use_gripper"),
        (args.validation, "validation"),
        (args.val_samples, "val_samples"),
        (args.device, "device"),
        (args.epochs, "epochs"),
        (args.checkpoint_path, "checkpoint_path"),
    ):
        if argument is not None:
            config[key] = str(argument) if isinstance(argument, Path) else argument
    train(config)


if __name__ == "__main__":
    main()
