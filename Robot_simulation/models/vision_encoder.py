"""Shared frozen ResNet-18 observation encoding for dataset and rollout images."""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


_IMAGE_CACHE: dict[str, np.ndarray] = {}


def _resolve_image_path(path: str, dataset_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(dataset_dir, path)


def _read_rgb_image(path: str, dataset_dir: str, cache_images: bool = False) -> np.ndarray:
    full_path = _resolve_image_path(path, dataset_dir)
    if cache_images and full_path in _IMAGE_CACHE:
        return _IMAGE_CACHE[full_path]
    image = np.asarray(imageio.imread(full_path)[..., :3], dtype=np.uint8).copy()
    if cache_images:
        _IMAGE_CACHE[full_path] = image
    return image


def image_cache_size() -> int:
    return len(_IMAGE_CACHE)



class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class _ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(_BasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


_RESNET18_WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def _make_resnet18(pretrained: bool) -> nn.Module:
    model = _ResNet18()
    if pretrained:
        state = torch.hub.load_state_dict_from_url(
            _RESNET18_WEIGHTS_URL, map_location="cpu", check_hash=True, progress=True
        )
        model.load_state_dict(state)
    model.fc = nn.Identity()
    return model

OBSERVATION_TYPES = ("state", "vision")
VISION_FINETUNE_MODES = ("frozen", "layer4", "layer3_layer4", "all")
VISION_POOL_TYPES = ("avg", "spatial_softmax")
VISION_FEATURE_NORMS = ("none", "layernorm")


def validate_observation_type(observation_type: str) -> str:
    if observation_type not in OBSERVATION_TYPES:
        raise ValueError(
            f"Unsupported observation_type={observation_type!r}; expected one of {OBSERVATION_TYPES}"
        )
    return observation_type


class FrozenResNet18Encoder(nn.Module):
    """Encode RGB views with a frozen or partially fine-tuned ResNet-18."""

    backbone_feature_dim = 512

    def __init__(
        self,
        camera_names: Sequence[str],
        pretrained: bool = True,
        finetune: bool = False,
        finetune_mode: str | None = None,
        train_bn: bool = False,
        pool: str = "avg",
        spatial_softmax_temperature: float = 1.0,
        feature_proj_dim: int = 0,
        feature_norm: str = "none",
        augmentation: bool = False,
        random_shift: int = 4,
        color_jitter: float = 0.1,
    ):
        super().__init__()
        self.camera_names = tuple(camera_names)
        if finetune_mode is None:
            finetune_mode = "layer4" if finetune else "frozen"
        if finetune_mode not in VISION_FINETUNE_MODES:
            raise ValueError(
                f"Unsupported finetune_mode={finetune_mode!r}; expected one of {VISION_FINETUNE_MODES}"
            )
        if pool not in VISION_POOL_TYPES:
            raise ValueError(f"Unsupported pool={pool!r}; expected one of {VISION_POOL_TYPES}")
        if feature_norm not in VISION_FEATURE_NORMS:
            raise ValueError(
                f"Unsupported feature_norm={feature_norm!r}; expected one of {VISION_FEATURE_NORMS}"
            )
        if spatial_softmax_temperature <= 0:
            raise ValueError("spatial_softmax_temperature must be positive")
        if feature_proj_dim < 0:
            raise ValueError("feature_proj_dim must be non-negative")
        if random_shift < 0:
            raise ValueError("random_shift must be non-negative")
        if color_jitter < 0:
            raise ValueError("color_jitter must be non-negative")

        self.backbone = _make_resnet18(pretrained)
        self.finetune_mode = finetune_mode
        self.finetune = finetune_mode != "frozen"
        self.train_bn = bool(train_bn)
        self.pool = pool
        self.spatial_softmax_temperature = float(spatial_softmax_temperature)
        self.feature_proj_dim = int(feature_proj_dim)
        self.feature_norm = feature_norm
        self.augmentation = bool(augmentation)
        self.random_shift = int(random_shift)
        self.color_jitter = float(color_jitter)
        self.raw_feature_dim = 1024 if self.pool == "spatial_softmax" else self.backbone_feature_dim
        if self.feature_proj_dim > 0:
            self.feature_dim = self.feature_proj_dim
            self.projection_head = nn.Sequential(
                nn.Linear(self.raw_feature_dim, self.feature_proj_dim),
                nn.LayerNorm(self.feature_proj_dim),
                nn.Mish(),
                nn.Linear(self.feature_proj_dim, self.feature_proj_dim),
            )
            self.feature_norm_layer = nn.Identity()
        else:
            self.feature_dim = self.raw_feature_dim
            self.projection_head = nn.Identity()
            self.feature_norm_layer = (
                nn.LayerNorm(self.feature_dim) if self.feature_norm == "layernorm" else nn.Identity()
            )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.set_backbone_trainable(self.finetune_mode)
        self.eval()

    @property
    def output_dim(self) -> int:
        return self.feature_dim * len(self.camera_names)

    def _selected_backbone_modules(self, mode: str) -> tuple[nn.Module, ...]:
        if mode == "frozen":
            return ()
        if mode == "layer4":
            return (self.backbone.layer4,)
        if mode == "layer3_layer4":
            return (self.backbone.layer3, self.backbone.layer4)
        if mode == "all":
            return (self.backbone,)
        raise ValueError(f"Unsupported finetune mode: {mode}")

    def _freeze_batchnorm(self) -> None:
        if self.train_bn:
            return
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(False)
                module.eval()

    def set_backbone_trainable(self, mode: str | None = None) -> None:
        if mode is None:
            mode = self.finetune_mode
        self.backbone.requires_grad_(False)
        for module in self._selected_backbone_modules(mode):
            module.requires_grad_(True)
        self._freeze_batchnorm()

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.augmentation):
            return x
        if self.random_shift > 0:
            pad = self.random_shift
            x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
            max_offset = 2 * pad
            offsets_y = torch.randint(0, max_offset + 1, (x.shape[0],), device=x.device)
            offsets_x = torch.randint(0, max_offset + 1, (x.shape[0],), device=x.device)
            x = torch.cat([
                x[i:i + 1, :, offsets_y[i]:offsets_y[i] + 224, offsets_x[i]:offsets_x[i] + 224]
                for i in range(x.shape[0])
            ], dim=0)
        if self.color_jitter > 0:
            jitter = self.color_jitter
            brightness = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
                1.0 - jitter, 1.0 + jitter
            )
            contrast = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
                1.0 - jitter, 1.0 + jitter
            )
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * contrast + mean
            x = torch.clamp(x * brightness, 0.0, 1.0)
        return x

    def _pool_features(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.pool == "avg":
            return torch.flatten(self.backbone.avgpool(feature_map), 1)
        batch, channels, height, width = feature_map.shape
        logits = feature_map.reshape(batch, channels, height * width) / self.spatial_softmax_temperature
        weights = torch.softmax(logits, dim=-1)
        ys = torch.linspace(-1.0, 1.0, height, device=feature_map.device, dtype=feature_map.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=feature_map.device, dtype=feature_map.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        expected_x = torch.sum(weights * grid_x.reshape(1, 1, height * width), dim=-1)
        expected_y = torch.sum(weights * grid_y.reshape(1, 1, height * width), dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(batch, channels * 2)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if self.finetune and mode:
            for module in self._selected_backbone_modules(self.finetune_mode):
                module.train(True)
        self._freeze_batchnorm()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode uint8/float images shaped (B, V, H, W, 3) or (B, V, 3, H, W)."""
        if images.ndim != 5:
            raise ValueError(f"Expected 5D image batch, got {tuple(images.shape)}")
        if images.shape[-1] in (3, 4):
            images = images[..., :3].permute(0, 1, 4, 2, 3)
        elif images.shape[2] != 3:
            raise ValueError(f"Expected RGB channels, got {tuple(images.shape)}")
        batch_size, num_views, channels, height, width = images.shape
        if num_views != len(self.camera_names):
            raise ValueError(f"Expected {len(self.camera_names)} views, got {num_views}")
        x = images.reshape(batch_size * num_views, channels, height, width).float()
        if float(x.max()) > 1.0:
            x = x / 255.0
        if (height, width) != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
        x = self._apply_augmentation(x)
        x = (x - self.image_mean) / self.image_std
        features = self._pool_features(self.backbone.forward_features(x))
        features = self.projection_head(features)
        features = self.feature_norm_layer(features)
        return features.reshape(batch_size, num_views * self.feature_dim)

    def backbone_parameters(self):
        return (
            parameter
            for name, parameter in self.named_parameters()
            if name.startswith("backbone.") and parameter.requires_grad
        )

    def projection_parameters(self):
        return (
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("backbone.") and parameter.requires_grad
        )

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

def load_camera_path_matrix(
    image_path_group,
    camera_names: Sequence[str],
) -> np.ndarray:
    """Read an HDF5 image_paths group into a decoded (T, V) string matrix."""
    columns = []
    for camera_name in camera_names:
        if camera_name not in image_path_group:
            raise ValueError(f"Dataset is missing image paths for camera {camera_name!r}")
        values = image_path_group[camera_name][:]
        columns.append([
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ])
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError(f"Camera path lengths disagree: {sorted(lengths)}")
    return np.asarray(columns, dtype=object).T


def encode_image_path_windows(
    path_windows: np.ndarray,
    dataset_dir: str,
    encoder: FrozenResNet18Encoder,
    device: torch.device | str,
    cache_images: bool = True,
    encoder_batch_size: int | None = None,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """Differentiably encode image windows shaped (B, observation_horizon, views)."""
    paths = np.asarray(path_windows, dtype=object)
    if paths.ndim != 3:
        raise ValueError(f"Expected path windows shaped (B, O, V), got {paths.shape}")
    policy_batch_size, observation_horizon, num_views = paths.shape
    if num_views != len(encoder.camera_names):
        raise ValueError(f"Expected {len(encoder.camera_names)} views, got {num_views}")
    flat_paths = paths.reshape(-1, num_views)
    encoder_batch_size = (
        len(flat_paths) if encoder_batch_size is None else int(encoder_batch_size)
    )
    if encoder_batch_size <= 0:
        raise ValueError(
            f"encoder_batch_size must be positive, got {encoder_batch_size}"
        )

    feature_batches = []
    for start in range(0, len(flat_paths), encoder_batch_size):
        path_rows = flat_paths[start:start + encoder_batch_size]
        images = np.stack([
            np.stack([
                _read_rgb_image(path, dataset_dir, cache_images)
                for path in row
            ], axis=0)
            for row in path_rows
        ], axis=0)
        image_tensor = torch.from_numpy(images).to(device)
        if gradient_checkpointing and torch.is_grad_enabled():
            encoded = checkpoint(encoder, image_tensor, use_reentrant=False)
        else:
            encoded = encoder(image_tensor)
        feature_batches.append(encoded)

    features = torch.cat(feature_batches, dim=0)
    return features.reshape(policy_batch_size, observation_horizon * encoder.output_dim)


@torch.no_grad()
def encode_image_path_episodes(
    path_episodes: Sequence[np.ndarray],
    dataset_dir: str,
    encoder: FrozenResNet18Encoder,
    device: torch.device | str,
    batch_size: int = 128,
    verbose: bool = False,
    cache_images: bool = False,
) -> list[np.ndarray]:
    """Load external images and return one (T, V*512) feature array per episode."""
    encoder = encoder.to(device).eval()
    total_frames = sum(len(paths) for paths in path_episodes)
    if verbose:
        print(
            "[vision-encode] "
            f"episodes={len(path_episodes)} | frames={total_frames} | "
            f"cameras={list(encoder.camera_names)} | batch_size={batch_size} | "
            f"device={device} | feature_dim_per_frame={encoder.output_dim}"
        )
    outputs = []
    log_period = max(1, len(path_episodes) // 10)
    for episode_idx, path_matrix in enumerate(path_episodes):
        if verbose and (episode_idx == 0 or (episode_idx + 1) % log_period == 0):
            print(
                f"[vision-encode] episode={episode_idx + 1}/{len(path_episodes)} "
                f"path_matrix_shape={tuple(path_matrix.shape)}"
            )
        episode_features = []
        for start in range(0, len(path_matrix), batch_size):
            rows = path_matrix[start:start + batch_size]
            images = np.stack([
                np.stack([
                    _read_rgb_image(path, dataset_dir, cache_images)
                    for path in row
                ], axis=0)
                for row in rows
            ], axis=0)
            encoded = encoder(torch.from_numpy(images).to(device))
            if verbose and episode_idx == 0 and start == 0:
                print(
                    f"[vision-encode] first_image_batch={tuple(images.shape)} -> "
                    f"first_feature_batch={tuple(encoded.shape)}"
                )
            episode_features.append(encoded.cpu().numpy().astype(np.float32))
        outputs.append(np.concatenate(episode_features, axis=0))
    if verbose:
        feature_shapes = sorted({tuple(features.shape[1:]) for features in outputs})
        print(
            f"[vision-encode] complete | episode_feature_shapes={feature_shapes} | "
            f"encoded_frames={sum(len(features) for features in outputs)} | image_cache_size={image_cache_size()}"
        )
    return outputs


@torch.no_grad()
def encode_camera_history(
    history: Sequence[Mapping[str, np.ndarray]],
    encoder: FrozenResNet18Encoder,
    device: torch.device | str,
) -> np.ndarray:
    """Encode an ordered rollout image history to a flat condition vector."""
    images = np.stack([
        np.stack([frame[name] for name in encoder.camera_names], axis=0)
        for frame in history
    ], axis=0)
    features = encoder(torch.from_numpy(images).to(device))
    return features.cpu().numpy().reshape(-1).astype(np.float32)
