"""Minibatch optimal-transport conditional Flow Matching."""

from __future__ import annotations

import numpy as np
import ot
import torch

from Synthetic_data.models.VanillaFM_class import VanillaFM


def _sample_ot_pairs(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    size = len(x0)
    weights = np.full(size, 1.0 / size, dtype=np.float64)
    cost = ot.dist(x0.detach().cpu().numpy(), x1.detach().cpu().numpy()) ** 2
    plan = ot.emd(weights, weights, cost).reshape(-1)
    if plan.sum() <= 0:
        return x0, x1
    pair_indices = np.random.choice(size * size, size=size, replace=True, p=plan / plan.sum())
    source_indices = torch.as_tensor(pair_indices // size, device=x0.device)
    target_indices = torch.as_tensor(pair_indices % size, device=x1.device)
    return x0[source_indices], x1[target_indices]


class OT_CFM(VanillaFM):
    """Vanilla FM whose Gaussian/data endpoints are paired by minibatch OT."""

    def __init__(self, *args, ot_max_batch_size: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        if ot_max_batch_size <= 0:
            raise ValueError("ot_max_batch_size must be positive")
        self.ot_max_batch_size = int(ot_max_batch_size)

    def _sample_endpoints(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = torch.randn_like(x1)
        if len(x1) <= self.ot_max_batch_size:
            return _sample_ot_pairs(x0, x1)

        source_perm = torch.randperm(len(x1), device=x1.device)
        target_perm = torch.randperm(len(x1), device=x1.device)
        source_chunks = []
        target_chunks = []
        for start in range(0, len(x1), self.ot_max_batch_size):
            source = x0[source_perm[start : start + self.ot_max_batch_size]]
            target = x1[target_perm[start : start + self.ot_max_batch_size]]
            source, target = _sample_ot_pairs(source, target)
            source_chunks.append(source)
            target_chunks.append(target)
        return torch.cat(source_chunks), torch.cat(target_chunks)
