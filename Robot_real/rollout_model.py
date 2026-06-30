"""Generate one action chunk from current camera images and joint angles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from Robot_real.train_model import (
    build_models,
    make_condition,
    resolve_device,
    sample_action_chunks,
    set_seed,
)

ImageInput = str | Path | np.ndarray | torch.Tensor


def _prepare_rgb_image(image: ImageInput, image_size: int) -> np.ndarray:
    """Load/resize one RGB image to the representation used during training."""
    if isinstance(image, (str, Path)):
        image_path = Path(image).expanduser().resolve()
        array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if array is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif torch.is_tensor(image):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(f"Expected one image with 3 dimensions, got {array.shape}")
    if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA channels, got {array.shape}")
    array = array[..., :3]
    if array.shape[:2] != (image_size, image_size):
        array = cv2.resize(
            array,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(array)


class RealRobotPolicy:
    """Checkpoint-backed policy for repeated real-robot inference.

    The checkpoint is loaded once. ``predict_action_chunk`` expects the current
    images in ``(frontview, wristview)`` order and the current joint vector in
    the same order recorded during training.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        sampler_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.metadata = checkpoint["metadata"]
        self.dof = int(self.metadata["dof"])
        self.horizon = int(self.metadata["horizon"])
        self.image_size = int(self.metadata["image_size"])
        self.sampler_steps = int(
            sampler_steps
            if sampler_steps is not None
            else self.metadata.get("sampler_steps", 100)
        )

        policy_seed = int(self.metadata.get("seed", 0) if seed is None else seed)
        set_seed(policy_seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(policy_seed)

        self.model, self.encoder = build_models(
            horizon=self.horizon,
            dof=self.dof,
            feature_proj_dim=int(self.metadata["feature_proj_dim"]),
            condition_embed_dim=int(self.metadata["condition_embed_dim"]),
            num_convs_per_block=int(self.metadata["num_convs_per_block"]),
            pretrained_vision=False,
            vision_finetune_mode=str(self.metadata["vision_finetune_mode"]),
            device=self.device,
            vision_train_bn=bool(self.metadata.get("vision_train_bn", False)),
            vision_pool=str(self.metadata.get("vision_pool", "avg")),
            vision_spatial_softmax_temperature=float(
                self.metadata.get("vision_spatial_softmax_temperature", 1.0)
            ),
            vision_feature_norm=str(self.metadata.get("vision_feature_norm", "none")),
            vision_augmentation=False,
            vision_random_shift=int(self.metadata.get("vision_random_shift", 4)),
            vision_color_jitter=float(self.metadata.get("vision_color_jitter", 0.1)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
        self.model.eval()
        self.encoder.eval()

        normalization = self.metadata["normalization"]
        self.normalization = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in normalization.items()
        }

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["joint_names"])

    @torch.no_grad()
    def predict_action_chunk(
        self,
        images: Sequence[ImageInput],
        joint_angles: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Return a denormalized action chunk shaped ``(horizon, dof)``."""
        if len(images) != 2:
            raise ValueError(
                "Expected two current images in (frontview, wristview) order, "
                f"got {len(images)}"
            )
        image_array = np.stack(
            [_prepare_rgb_image(image, self.image_size) for image in images],
            axis=0,
        )
        image_batch = torch.from_numpy(image_array).unsqueeze(0)

        joints = torch.as_tensor(
            joint_angles,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        if joints.numel() != self.dof:
            raise ValueError(
                f"Checkpoint expects {self.dof} joints {self.joint_names}, "
                f"but received {joints.numel()} values"
            )
        if "min" in self.normalization:
            normalized_joints = torch.clamp(
                (2.0 / self.normalization["range"])
                * (joints - self.normalization["min"])
                - 1.0,
                -1.0,
                1.0,
            ).unsqueeze(0)
        else:
            normalized_joints = (
                (joints - self.normalization["mean"]) / self.normalization["std"]
            ).unsqueeze(0)
        condition = make_condition(
            self.encoder,
            image_batch,
            normalized_joints,
            self.device,
        )
        normalized_chunk = sample_action_chunks(
            self.model,
            condition,
            model_type=str(self.metadata["model_type"]),
            horizon=self.horizon,
            dof=self.dof,
            sampler_steps=self.sampler_steps,
            diffusion_steps=int(self.metadata["diffusion_steps"]),
            diffusion_schedule=str(self.metadata["diffusion_schedule"]),
            diffusion_pred_type=str(self.metadata["diffusion_pred_type"]),
            diffusion_eta=float(self.metadata["diffusion_eta"]),
            clip_sample=bool(self.metadata["clip_sample"]),
            clip_sample_range=float(self.metadata["clip_sample_range"]),
            generator=self.generator,
        )[0]
        if "min" in self.normalization:
            action_chunk = (
                (normalized_chunk + 1.0)
                * 0.5
                * self.normalization["range"][None, :]
                + self.normalization["min"][None, :]
            )
        else:
            action_chunk = (
                normalized_chunk * self.normalization["std"][None, :]
                + self.normalization["mean"][None, :]
            )
        return action_chunk.cpu().numpy()

    def __call__(
        self,
        images: Sequence[ImageInput],
        joint_angles: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        return self.predict_action_chunk(images, joint_angles)


def predict_action_chunk(
    checkpoint_path: str | Path,
    images: Sequence[ImageInput],
    joint_angles: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    device: str = "cuda",
    sampler_steps: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """One-shot convenience interface; reuse ``RealRobotPolicy`` in control loops."""
    policy = RealRobotPolicy(
        checkpoint_path,
        device=device,
        sampler_steps=sampler_steps,
        seed=seed,
    )
    return policy.predict_action_chunk(images, joint_angles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--front-image", type=Path, required=True)
    parser.add_argument("--wrist-image", type=Path, required=True)
    parser.add_argument("--joint-angles", type=float, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampler-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = RealRobotPolicy(
        args.checkpoint,
        device=args.device,
        sampler_steps=args.sampler_steps,
        seed=args.seed,
    )
    action_chunk = policy.predict_action_chunk(
        (args.front_image, args.wrist_image),
        args.joint_angles,
    )
    result = {
        "joint_names": list(policy.joint_names),
        "shape": list(action_chunk.shape),
        "action_chunk": action_chunk.tolist(),
    }
    output = json.dumps(result, indent=2)
    if args.output is None:
        print(output)
    else:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n")
        print(f"Saved action chunk to {output_path}")


if __name__ == "__main__":
    main()
