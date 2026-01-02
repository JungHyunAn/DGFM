# train_visual_encoder.py
"""
Train a vision encoder that maps an environment image/vision observation -> environment parameter vector.

Outputs:
- Saves encoder state_dict to `--out_path`
- Saves a JSON with normalization stats (optional but strongly recommended)

Usage example:
python -m Robot_simulation.train_vision_encoder \
  --task_name nut --param_len 3 --epochs 50 --n_samples 200 \
  --out_path ./vision_encoder/vision_encoder_nut_simple.pt \
  --stats_path ./vision_encoder/vision_encoder_nut_simple_stats.json
"""

import os
import json
import math
import argparse
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from Robot_simulation.run_eval import _spawn_env_once


# ----------------------------
# Model: simple CNN encoder
# ----------------------------
class SimpleVisionEncoder(nn.Module):
    """
    A minimal CNN that outputs a latent vector, followed by a regressor to param_len.
    """
    def __init__(self, in_channels: int, param_len: int, latent_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(latent_dim, param_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        z = self.proj(x)
        y = self.head(z)
        return y


# ----------------------------
# Dataset
# ----------------------------
class VisionParamDataset(Dataset):
    def __init__(
        self,
        visions: np.ndarray,     # (N,H,W,C) or (N,C,H,W)
        params: np.ndarray,      # (N,param_len)
        normalize_params: bool = True,
        param_mean: Optional[np.ndarray] = None,
        param_std: Optional[np.ndarray] = None,
        normalize_images: bool = True,
    ):
        assert visions.shape[0] == params.shape[0], "visions and params must have same N"
        self.normalize_images = normalize_images
        self.normalize_params = normalize_params

        # Convert to float32
        visions = visions.astype(np.float32)
        params = params.astype(np.float32)

        # If images look like 0..255, scale to 0..1 (common)
        if normalize_images:
            vmax = float(np.nanmax(visions))
            if vmax > 1.5:
                visions = visions / 255.0

        # Ensure NCHW
        if visions.ndim != 4:
            raise ValueError(f"Expected visions to be 4D, got {visions.shape}")
        if visions.shape[-1] in (1, 3, 4):  # NHWC
            visions = np.transpose(visions, (0, 3, 1, 2))  # -> NCHW

        self.visions = visions
        self.params_raw = params

        # Param normalization
        if normalize_params:
            if param_mean is None or param_std is None:
                param_mean = params.mean(axis=0)
                param_std = params.std(axis=0)
            # guard against zero std
            param_std = np.where(param_std < 1e-8, 1.0, param_std)

            self.param_mean = param_mean.astype(np.float32)
            self.param_std = param_std.astype(np.float32)
            self.params = ((params - self.param_mean) / self.param_std).astype(np.float32)
        else:
            self.param_mean = None
            self.param_std = None
            self.params = params.astype(np.float32)

    def __len__(self) -> int:
        return self.visions.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.visions[idx])     # (C,H,W)
        y = torch.from_numpy(self.params[idx])      # (param_len,)
        return x, y


# ----------------------------
# Data generation (parallel)
# ----------------------------
def generate_dataset_parallel(
    task_name: str,
    n_samples: int,
    base_seed: int,
    workers: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Returns:
        visions: (N, H, W, C) or (N, C, H, W) as float32
        params:  (N, param_len) float32
        settings: list of per-sample env settings dicts
    """
    env_params_list: List[np.ndarray] = [None] * n_samples
    env_settings_all: List[Dict[str, Any]] = [None] * n_samples
    env_vision_list: List[np.ndarray] = [None] * n_samples

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [
            ex.submit(_spawn_env_once, task_name, base_seed + i, i, True, "frontview")
            for i in range(n_samples)
        ]

        # tqdm wrapped over as_completed
        for fut in tqdm(
            as_completed(futs),
            total=n_samples,
            desc=f"Generating {task_name} dataset",
            dynamic_ncols=True,
        ):
            idx, setting, params, vision = fut.result()
            env_settings_all[idx] = setting
            env_params_list[idx] = params
            env_vision_list[idx] = vision

    params = np.asarray(env_params_list, dtype=np.float32)
    visions = np.asarray(env_vision_list, dtype=np.float32)

    return visions, params, env_settings_all


# ----------------------------
# Training utilities
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    mse_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        mse = torch.mean((pred - y) ** 2, dim=1)  # per-sample
        mse_sum += float(mse.sum().item())
        n += x.shape[0]
    return mse_sum / max(1, n)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, choices=["door", "wipe", "two_arm", "nut"])
    parser.add_argument("--param_len", type=int, required=True)

    # Data generation
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--env_workers", type=int, default=10)

    # Train config
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=128)

    # Normalization
    parser.add_argument("--normalize_params", action="store_true", default=True)
    parser.add_argument("--no_normalize_params", action="store_false", dest="normalize_params")
    parser.add_argument("--normalize_images", action="store_true", default=True)
    parser.add_argument("--no_normalize_images", action="store_false", dest="normalize_images")

    # IO
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--stats_path", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    if args.stats_path is not None:
        os.makedirs(os.path.dirname(args.stats_path) or ".", exist_ok=True)

    set_seed(args.base_seed + 12345)

    # 1) Generate dataset from env
    workers = min(args.env_workers, max(1, (os.cpu_count() or 4) - 2), args.n_samples)
    print(f"[Data] Generating {args.n_samples} samples for task={args.task_name} with workers={workers} ...")
    visions, params, _settings = generate_dataset_parallel(
        task_name=args.task_name,
        n_samples=args.n_samples,
        base_seed=args.base_seed,
        workers=workers,
    )
    print(f"[Data] visions shape={visions.shape}, params shape={params.shape}")

    # Infer input channels
    if visions.ndim != 4:
        raise ValueError(f"visions must be 4D, got {visions.shape}")
    if visions.shape[-1] in (1, 3, 4):
        in_channels = int(visions.shape[-1])  # NHWC
    else:
        in_channels = int(visions.shape[1])   # NCHW

    # 2) Build dataset with param normalization (fit on full set, then split)
    #    If you prefer: fit on train only; below fits on full set for simplicity.
    full_ds = VisionParamDataset(
        visions=visions,
        params=params,
        normalize_params=args.normalize_params,
        normalize_images=args.normalize_images,
    )

    # If you want normalization based on train split only, do this instead:
    # - split indices
    # - compute mean/std on train params
    # - rebuild datasets with those stats
    # For most cases, fitting on full set is acceptable for encoder pretraining.

    # 3) Train/val split
    n_val = int(math.floor(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(args.base_seed + 999),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 4) Model
    model = SimpleVisionEncoder(
        in_channels=in_channels,
        param_len=args.param_len,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # 5) Train loop
    best_val = float("inf")
    print(f"[Train] epochs={args.epochs}, batch_size={args.batch_size}, device={device}")

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(1, n_batches)
        val_mse = evaluate(model, val_loader, device=device)

        print(f"[Epoch {ep:03d}] train_mse={train_loss:.6f} | val_mse={val_mse:.6f}")

        # Save best
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), args.out_path)

            # Save stats (param normalization) for correct de-normalization later
            if args.stats_path is not None:
                stats = {
                    "task_name": args.task_name,
                    "param_len": args.param_len,
                    "normalize_params": bool(args.normalize_params),
                    "normalize_images": bool(args.normalize_images),
                    "param_mean": full_ds.param_mean.tolist() if full_ds.param_mean is not None else None,
                    "param_std": full_ds.param_std.tolist() if full_ds.param_std is not None else None,
                    "in_channels": in_channels,
                }
                with open(args.stats_path, "w") as f:
                    json.dump(stats, f, indent=2)

            print(f"  [Saved] best model -> {args.out_path} (val_mse={best_val:.6f})")

    print(f"[Done] best_val_mse={best_val:.6f}")
    print(f"[Done] model saved at: {args.out_path}")
    if args.stats_path is not None:
        print(f"[Done] stats saved at: {args.stats_path}")


if __name__ == "__main__":
    main()
