import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Robot_simulation.models import vision_encoder as vision_module


class _TrackingEncoder(nn.Module):
    camera_names = ("frontview", "eye_in_hand")
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(6, self.output_dim)
        self.seen_batch_sizes: list[int] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.seen_batch_sizes.append(images.shape[0])
        features = images.float().mean(dim=(2, 3)).reshape(images.shape[0], -1)
        return self.projection(features)


class _AugmentationHarness:
    training = True
    augmentation = True
    random_shift = 2
    color_jitter = 0.0


def _reference_random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    max_offset = 2 * pad
    offsets_y = torch.randint(0, max_offset + 1, (x.shape[0],), device=x.device)
    offsets_x = torch.randint(0, max_offset + 1, (x.shape[0],), device=x.device)
    output_height = x.shape[-2] - 2 * pad
    output_width = x.shape[-1] - 2 * pad
    return torch.cat([
        x[
            i:i + 1,
            :,
            offsets_y[i]:offsets_y[i] + output_height,
            offsets_x[i]:offsets_x[i] + output_width,
        ]
        for i in range(x.shape[0])
    ], dim=0)


def test_vectorized_random_shift_matches_reference_and_preserves_gradients() -> None:
    images = torch.arange(3 * 2 * 7 * 9, dtype=torch.float32).reshape(3, 2, 7, 9)
    images.requires_grad_(True)

    torch.manual_seed(17)
    actual = vision_module.FrozenResNet18Encoder._apply_augmentation(
        _AugmentationHarness(), images
    )
    torch.manual_seed(17)
    expected = _reference_random_shift(images, _AugmentationHarness.random_shift)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, images)
    actual.sum().backward()
    assert images.grad is not None
    assert torch.count_nonzero(images.grad) > 0


def test_augmentation_is_disabled_outside_training_mode() -> None:
    harness = _AugmentationHarness()
    harness.training = False
    images = torch.rand(2, 3, 8, 8)

    output = vision_module.FrozenResNet18Encoder._apply_augmentation(harness, images)

    assert output is images


def test_online_window_encoding_is_batched_and_differentiable(monkeypatch) -> None:
    def fake_read(path: str, dataset_dir: str, cache_images: bool = False) -> np.ndarray:
        del dataset_dir, cache_images
        value = int(path.split("_")[-1])
        return np.full((8, 8, 3), value, dtype=np.uint8)

    monkeypatch.setattr(vision_module, "_read_rgb_image", fake_read)
    paths = np.asarray(
        [[[f"image_{2 * obs + view}" for view in range(2)] for obs in range(2)] for _ in range(3)],
        dtype=object,
    )
    encoder = _TrackingEncoder()

    features = vision_module.encode_image_path_windows(
        paths,
        dataset_dir="unused",
        encoder=encoder,
        device="cpu",
        cache_images=False,
        encoder_batch_size=2,
        gradient_checkpointing=True,
    )
    features.square().mean().backward()

    assert features.shape == (3, 2 * encoder.output_dim)
    assert max(encoder.seen_batch_sizes) == 2
    assert encoder.projection.weight.grad is not None
