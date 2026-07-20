"""Shared utilities for synthetic-data flow matching models."""

from __future__ import annotations

import numpy as np
import ot
import torch


def split_train_validation(
    target: torch.Tensor,
    *,
    validation_fraction: float = 0.1,
    max_validation_size: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly split a target sample while keeping both partitions non-empty."""
    if target.ndim != 2:
        raise ValueError(f"Expected a rank-2 target tensor, got shape {tuple(target.shape)}")
    if target.shape[0] < 2:
        raise ValueError("Flow-matching validation requires at least two target samples")

    validation_size = min(
        max(1, int(validation_fraction * target.shape[0])),
        max_validation_size,
        target.shape[0] - 1,
    )
    permutation = torch.randperm(target.shape[0], device=target.device)
    return target[permutation[:-validation_size]], target[permutation[-validation_size:]]


def empirical_wasserstein2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the empirical W2 distance used for model selection."""
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    x_weights = np.full(len(x_np), 1.0 / len(x_np), dtype=np.float64)
    y_weights = np.full(len(y_np), 1.0 / len(y_np), dtype=np.float64)
    cost = ot.dist(x_np, y_np) ** 2
    return float(np.sqrt(ot.emd2(x_weights, y_weights, cost)))


@torch.no_grad()
def run_flow(model, x0, device, n_steps: int = 100) -> torch.Tensor:
    """Euler-integrate a time-conditioned vector field from t=0 to t=1."""
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    x = torch.as_tensor(x0, dtype=torch.float32, device=device).clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((x.shape[0],), step * dt, device=device, dtype=x.dtype)
        x.add_(model(x, t), alpha=dt)
    return x
