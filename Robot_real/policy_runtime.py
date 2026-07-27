"""Lightweight model construction and sampling helpers for real-robot inference.

This module intentionally avoids importing the training and simulation stacks so
rollout workers only need PyTorch, NumPy, and OpenCV/image-model dependencies.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from Robot_simulation.models.VanillaFM_class import VectorField
from Robot_simulation.models.vision_encoder import FrozenResNet18Encoder


MODEL_TYPES = ("UniformFM", "DGFMv2", "DiffusionPolicy")
MODEL_TYPE_ALIASES = {
    "fm": "UniformFM",
    "diffusion": "DiffusionPolicy",
    "DP": "DiffusionPolicy",
}


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
        camera_names=("frontview", "wristview"),
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
    image_features = encoder(flat_images.to(device, non_blocking=True))
    state_history = joint_state.to(device, non_blocking=True).reshape(batch_size, -1)
    return torch.cat(
        [state_history, image_features.reshape(batch_size, -1)], dim=-1
    )


class _DiffusionSchedule:
    def __init__(self, steps: int, device: torch.device, schedule: str):
        self.steps = int(steps)
        if schedule == "cosine":
            x = torch.linspace(0, self.steps, self.steps + 1, dtype=torch.float64)
            alphas_bar = torch.cos(
                ((x / self.steps) + 0.008) / 1.008 * math.pi * 0.5
            ) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
            betas = torch.clamp(betas, 0.0, 0.999).float()
        elif schedule == "linear":
            betas = torch.linspace(1.0e-4, 2.0e-2, self.steps)
        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule}")
        alphas = 1.0 - betas.to(device)
        self.alphas_bar = torch.cumprod(alphas, dim=0)

    def gather(self, indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        values = self.alphas_bar.gather(0, indices)
        return values.view(-1, *([1] * (len(shape) - 1)))


@torch.no_grad()
def _run_diffusion(
    model: VectorField,
    noise: torch.Tensor,
    conditions: torch.Tensor,
    *,
    diffusion_steps: int,
    schedule_type: str,
    sampler_steps: int,
    eta: float,
    pred_type: str,
    clip_sample: bool,
    clip_sample_range: float,
) -> torch.Tensor:
    schedule = _DiffusionSchedule(diffusion_steps, noise.device, schedule_type)
    if sampler_steps >= diffusion_steps:
        indices = list(range(diffusion_steps - 1, -1, -1))
    else:
        stride = (diffusion_steps - 1) / float(sampler_steps - 1)
        indices = [
            int(round((sampler_steps - 1 - i) * stride))
            for i in range(sampler_steps)
        ]
        indices = sorted(set(indices), reverse=True)
        if indices[-1] != 0:
            indices.append(0)

    sample = noise
    batch = sample.shape[0]
    for iteration, index in enumerate(indices):
        k = torch.full((batch,), index, device=sample.device, dtype=torch.long)
        time = ((k.float() + 1.0) / float(diffusion_steps)).unsqueeze(-1)
        alpha_bar = schedule.gather(k, sample.shape)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)
        output = model(sample, time, conditions)
        if pred_type == "x0":
            x0 = output
            epsilon = (sample - sqrt_alpha_bar * x0) / (sqrt_one_minus + 1.0e-8)
        elif pred_type == "epsilon":
            epsilon = output
            x0 = (sample - sqrt_one_minus * epsilon) / (sqrt_alpha_bar + 1.0e-8)
        else:
            raise ValueError(f"Unknown diffusion prediction type: {pred_type}")
        if clip_sample:
            x0 = torch.clamp(x0, -clip_sample_range, clip_sample_range)
        if iteration == len(indices) - 1:
            return x0

        next_index = indices[iteration + 1]
        next_k = torch.full(
            (batch,), next_index, device=sample.device, dtype=torch.long
        )
        next_alpha_bar = schedule.gather(next_k, sample.shape)
        sigma = eta * torch.sqrt(
            torch.clamp(
                (1.0 - next_alpha_bar)
                / (1.0 - alpha_bar + 1.0e-8)
                * (1.0 - alpha_bar / (next_alpha_bar + 1.0e-8)),
                min=0.0,
            )
        )
        direction = torch.sqrt(
            torch.clamp(1.0 - next_alpha_bar - sigma**2, min=0.0)
        ) * epsilon
        random_noise = torch.randn_like(sample) if eta > 0.0 else 0.0
        sample = torch.sqrt(next_alpha_bar) * x0 + direction + sigma * random_noise
    raise RuntimeError("Diffusion sampler produced no result")


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
    model_type = MODEL_TYPE_ALIASES.get(model_type, model_type)
    if sampler_steps <= 0:
        raise ValueError("sampler_steps must be positive")
    if model_type == "DiffusionPolicy" and sampler_steps < 2:
        raise ValueError("DiffusionPolicy requires sampler_steps >= 2")
    noise = torch.randn(
        conditions.shape[0],
        horizon,
        dof,
        device=conditions.device,
        dtype=conditions.dtype,
        generator=generator,
    )
    if model_type in ("UniformFM", "DGFMv2"):
        sample = noise
        dt = 1.0 / sampler_steps
        time = torch.empty(
            (conditions.shape[0], 1),
            device=conditions.device,
            dtype=conditions.dtype,
        )
        for step in range(sampler_steps):
            time.fill_(step * dt)
            sample.add_(model(sample, time, conditions), alpha=dt)
        return sample
    if model_type == "DiffusionPolicy":
        return _run_diffusion(
            model,
            noise,
            conditions,
            diffusion_steps=diffusion_steps,
            schedule_type=diffusion_schedule,
            sampler_steps=min(sampler_steps, diffusion_steps),
            eta=diffusion_eta,
            pred_type=diffusion_pred_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
    raise ValueError(f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")
