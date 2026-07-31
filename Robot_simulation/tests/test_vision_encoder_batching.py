import numpy as np
import torch
from torch import nn

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
