"""Condition-free vector field and base trainer for synthetic data."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.distributions import Beta
from tqdm import tqdm

from Synthetic_data.models.FM_util import empirical_wasserstein2, run_flow, split_train_validation


class VectorField(nn.Module):
    """MLP vector field conditioned only on continuous time."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1, 1)
        if t.shape[0] == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0], 1)
        return self.net(torch.cat((x, t), dim=1))


class VanillaFM:
    """Base condition-free Flow Matching trainer."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        *,
        device="cpu",
        time_sampling: str = "uniform",
        beta_a: float = 1.5,
        beta_b: float = 1.0,
    ):
        if time_sampling not in {"uniform", "shifted"}:
            raise ValueError("time_sampling must be 'uniform' or 'shifted'")
        if beta_a <= 0 or beta_b <= 0:
            raise ValueError("beta_a and beta_b must be positive")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.time_sampling = time_sampling
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)

    def sample_t(self, n: int) -> torch.Tensor:
        if self.time_sampling == "shifted":
            samples = Beta(self.beta_a, self.beta_b).sample((n,)).to(self.device)
            return 1.0 - samples
        return torch.rand(n, device=self.device)

    def _sample_endpoints(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.randn_like(x1), x1

    def _build_interpolants(
        self,
        x1: torch.Tensor,
        n_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0, x1 = self._sample_endpoints(x1)
        dim = x1.shape[1]
        t = self.sample_t(len(x1) * n_t)
        x0r = x0.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, dim)
        x1r = x1.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, dim)
        xt = (1.0 - t[:, None]) * x0r + t[:, None] * x1r
        return xt, t, x1r - x0r

    def train(
        self,
        target_points: torch.Tensor,
        *,
        n_t: int,
        max_epochs: int,
        batch_size: int,
        early_stopping: bool = True,
        stop_criteria: int = 3,
        tolerance: float = 1e-3,
        validation_fraction: float = 0.1,
        max_validation_size: int = 2000,
        progress: bool = True,
    ):
        if n_t <= 0 or max_epochs <= 0 or batch_size <= 0:
            raise ValueError("n_t, max_epochs, and batch_size must all be positive")

        self.model = self.model.to(self.device)
        target_points = target_points.to(self.device)
        train_points, validation_points = split_train_validation(
            target_points,
            validation_fraction=validation_fraction,
            max_validation_size=max_validation_size,
        )

        best_w2 = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        records: list[dict] = []
        stale_epochs = 0

        epochs = tqdm(
            range(1, max_epochs + 1),
            desc=f"{self.time_sampling.title()}FM Training",
            unit="epoch",
            disable=not progress,
        )
        for epoch in epochs:
            self.model.train()
            permutation = torch.randperm(len(train_points), device=self.device)
            loss_sum = 0.0
            batch_count = 0
            for start in range(0, len(train_points), batch_size):
                idx = permutation[start : start + batch_size]
                xt, t, target_velocity = self._build_interpolants(train_points[idx], n_t)
                prediction = self.model(xt, t)
                loss = ((prediction - target_velocity) ** 2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                loss_sum += float(loss.item())
                batch_count += 1

            self.model.eval()
            generated = run_flow(
                self.model,
                torch.randn_like(validation_points),
                self.device,
            )
            validation_w2 = empirical_wasserstein2(generated, validation_points)
            improved = best_w2 - validation_w2 >= tolerance
            if validation_w2 < best_w2:
                best_w2 = validation_w2
                best_state = copy.deepcopy(self.model.state_dict())

            stale_epochs = 0 if improved else stale_epochs + 1
            records.append(
                {
                    "epoch": epoch,
                    "validation_w2": validation_w2,
                    "train_loss": loss_sum / max(1, batch_count),
                    "best_model_save": bool(validation_w2 <= best_w2),
                }
            )
            if early_stopping and stale_epochs >= stop_criteria:
                break

        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(best_state)
        best_model.eval()
        return best_model, self.model, records
