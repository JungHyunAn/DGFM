"""Condition-free DGFMv2 with full-dimensional cluster intermediates."""

from __future__ import annotations

import copy

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import chi2, truncnorm
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from Synthetic_data.models.FM_util import empirical_wasserstein2, run_flow, split_train_validation
from Synthetic_data.models.VanillaFM_class import VanillaFM


def cluster_points_x(
    x: np.ndarray,
    cluster_size: int,
    *,
    jaccard_thresh: float = 0.8,
    merge_k: int = 10,
    standardize: bool = True,
) -> tuple[list[set[int]], dict[int, list[int]]]:
    """Build overlapping nearest-neighbor clusters using data coordinates only."""
    sample_count = x.shape[0]
    if sample_count < 2:
        raise ValueError("DGFMv2 clustering requires at least two training samples")
    if cluster_size < 2:
        raise ValueError("cluster_size must be at least two")

    features = x
    if standardize:
        std = x.std(axis=0, keepdims=True)
        features = (x - x.mean(axis=0, keepdims=True)) / np.where(std < 1e-8, 1.0, std)

    neighbors = NearestNeighbors(
        n_neighbors=min(cluster_size, sample_count), algorithm="auto"
    ).fit(features)
    raw_indices = neighbors.kneighbors(features, return_distance=False)
    raw_clusters = [set(row.tolist()) | {idx} for idx, row in enumerate(raw_indices)]

    covered = np.zeros(sample_count, dtype=bool)
    seeds: list[int] = []
    candidates: list[set[int]] = []
    for idx, candidate in enumerate(raw_clusters):
        if not covered[idx]:
            seeds.append(idx)
            candidates.append(candidate.copy())
            covered[list(candidate)] = True

    if len(candidates) <= 1:
        clusters = [set(range(sample_count))]
    else:
        seed_features = features[seeds]
        seed_model = NearestNeighbors(
            n_neighbors=min(merge_k + 1, len(seeds)), algorithm="auto"
        ).fit(seed_features)
        seed_neighbors = seed_model.kneighbors(seed_features, return_distance=False)
        parent = list(range(len(candidates)))
        cluster_sets = {idx: candidate for idx, candidate in enumerate(candidates)}

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int) -> None:
            left, right = find(left), find(right)
            if left == right:
                return
            if len(cluster_sets[left]) < len(cluster_sets[right]):
                left, right = right, left
            parent[right] = left
            cluster_sets[left] |= cluster_sets.pop(right)

        for idx, adjacent in enumerate(seed_neighbors):
            for other in adjacent:
                left, right = find(idx), find(int(other))
                if left == right:
                    continue
                intersection = len(cluster_sets[left] & cluster_sets[right])
                union_size = len(cluster_sets[left] | cluster_sets[right])
                if union_size and intersection / union_size >= jaccard_thresh:
                    union(left, right)
        clusters = list(cluster_sets.values())

    inverse = {idx: [] for idx in range(sample_count)}
    for cluster_idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            inverse[sample_idx].append(cluster_idx)
    return clusters, inverse


def _pad_basis(basis: np.ndarray, dimension: int) -> np.ndarray:
    columns = [basis[:, idx].astype(np.float32, copy=False) for idx in range(basis.shape[1])]
    for axis in range(dimension):
        if len(columns) == dimension:
            break
        vector = np.zeros(dimension, dtype=np.float32)
        vector[axis] = 1.0
        for column in columns:
            vector -= np.dot(column, vector) * column
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            columns.append(vector / norm)
    return np.stack(columns, axis=1).astype(np.float32, copy=False)


def _fit_cluster(
    x: np.ndarray,
    indices: list[int],
    eps: float,
    outlier_threshold: float,
    max_pca_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    points = x[indices]
    dimension = x.shape[1]
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(points) - 1) + eps * np.eye(dimension)
    try:
        chol = np.linalg.cholesky(covariance)
        whitened = np.linalg.solve(chol, centered.T)
        clean = points[(whitened * whitened).sum(axis=0) <= outlier_threshold]
    except np.linalg.LinAlgError:
        clean = points
    if len(clean) < 2:
        clean = points

    pca_points = clean
    if len(clean) > max_pca_samples:
        selected = np.random.choice(len(clean), max_pca_samples, replace=False)
        pca_points = clean[selected]
    pca_centered = pca_points - pca_points.mean(axis=0, keepdims=True)
    varying = pca_centered.var(axis=0) > 1e-12
    if len(pca_points) < 2 or not varying.any():
        basis = np.eye(dimension, dtype=np.float32)
    else:
        reduced = pca_centered[:, varying]
        component_count = min(reduced.shape[1], len(reduced) - 1)
        pca = IncrementalPCA(
            n_components=component_count,
            batch_size=min(1024, len(reduced)),
        ).fit(reduced)
        basis = np.zeros((dimension, component_count), dtype=np.float32)
        basis[varying] = pca.components_.T.astype(np.float32, copy=False)
        basis = _pad_basis(basis, dimension)

    mean = clean.mean(axis=0)
    latent = (clean - mean) @ basis
    latent_covariance = latent.T @ latent / max(1, len(clean) - 1)
    latent_covariance += eps * np.eye(dimension)
    return mean, basis, latent_covariance, len(clean) / len(x)


def compute_cluster_pca_full(
    x: np.ndarray,
    clusters: list[set[int]],
    *,
    eps: float = 1e-3,
    outlier_q: float = 0.9,
    max_pca_samples: int = 2000,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a full ambient-dimensional PCA Gaussian to every cluster."""
    threshold = chi2.ppf(outlier_q, df=x.shape[1])
    fitted = Parallel(n_jobs=n_jobs)(
        delayed(_fit_cluster)(x, list(cluster), eps, threshold, max_pca_samples)
        for cluster in clusters
    )
    means, bases, covariances, weights = zip(*fitted)
    return np.stack(means), np.stack(bases), np.stack(covariances), np.asarray(weights)


class MixtureSamplerV2:
    """Full-dimensional per-cluster PCA sampler."""

    def __init__(
        self,
        means,
        bases,
        latent_covariances,
        weights,
        clusters,
        inverse_clusters,
        *,
        truncation: float = 1.5,
        device="cpu",
        regularization: float = 1e-6,
    ):
        self.device = torch.device(device)
        self.means = torch.as_tensor(means, dtype=torch.float32, device=self.device)
        self.bases = torch.as_tensor(bases, dtype=torch.float32, device=self.device)
        covariance = torch.as_tensor(latent_covariances, dtype=torch.float32, device=self.device)
        identity = torch.eye(covariance.shape[-1], device=self.device)
        self.cholesky = torch.linalg.cholesky(covariance + regularization * identity)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        self.weights = weights / weights.sum()
        self.clusters = clusters
        self.inverse_clusters = inverse_clusters
        self.truncation = float(truncation)
        self.dimension = self.means.shape[1]

    @torch.no_grad()
    def sample(self, count: int, cluster_ids=None) -> tuple[torch.Tensor, torch.Tensor]:
        if cluster_ids is None:
            cluster_ids = torch.multinomial(self.weights, count, replacement=True)
        cluster_ids = torch.as_tensor(cluster_ids, device=self.device, dtype=torch.long)
        latent = truncnorm.rvs(
            -self.truncation,
            self.truncation,
            size=(count, self.dimension),
        ).astype(np.float32)
        latent = torch.from_numpy(latent).to(self.device)
        latent = torch.bmm(self.cholesky[cluster_ids], latent.unsqueeze(-1)).squeeze(-1)
        points = self.means[cluster_ids] + torch.bmm(
            self.bases[cluster_ids], latent.unsqueeze(-1)
        ).squeeze(-1)
        return points, cluster_ids


class DGFMv2(VanillaFM):
    """Smoothly transport noise through a target-associated cluster sample."""

    path_weight_atol = 1e-5

    def _sample_covering_clusters(
        self,
        indices: torch.Tensor,
        inverse_clusters: dict[int, list[int]],
        cluster_sizes: np.ndarray,
    ) -> torch.Tensor:
        selected = []
        for sample_idx in indices.detach().cpu().tolist():
            covering = inverse_clusters[sample_idx]
            probabilities = cluster_sizes[covering] / cluster_sizes[covering].sum()
            selected.append(np.random.choice(covering, p=probabilities))
        return torch.as_tensor(selected, device=self.device, dtype=torch.long)

    def _path_weights(
        self,
        t,
        interpolation_path,
        residual_lambda=0.2,
    ):
        if interpolation_path != "residual-cosine-midpoint":
            raise ValueError(
                f"Unsupported DGFM interpolation path '{interpolation_path}'."
            )

        tau = torch.as_tensor(0.5, dtype=t.dtype, device=t.device)
        lam = torch.as_tensor(residual_lambda, dtype=t.dtype, device=t.device)
        left = t <= tau
        right = ~left

        pi = torch.as_tensor(torch.pi, device=t.device, dtype=t.dtype)

        r_left = torch.clamp(t / tau, 0.0, 1.0)
        r_right = torch.clamp((t - tau) / (1.0 - tau), 0.0, 1.0)

        s_left = 0.5 * (1.0 - torch.cos(pi * r_left))
        s_right = 0.5 * (1.0 - torch.cos(pi * r_right))

        sdot_left = 0.5 * pi * torch.sin(pi * r_left) / tau
        sdot_right = 0.5 * pi * torch.sin(pi * r_right) / (1.0 - tau)

        a_left_D = 1.0 - s_left
        b_left_D = s_left
        c_left_D = torch.zeros_like(t)

        a_dot_left_D = -sdot_left
        b_dot_left_D = sdot_left
        c_dot_left_D = torch.zeros_like(t)

        a_right_D = torch.zeros_like(t)
        b_right_D = 1.0 - s_right
        c_right_D = s_right

        a_dot_right_D = torch.zeros_like(t)
        b_dot_right_D = -sdot_right
        c_dot_right_D = sdot_right

        a_D = torch.where(left, a_left_D, a_right_D)
        b_D = torch.where(left, b_left_D, b_right_D)
        c_D = torch.where(left, c_left_D, c_right_D)

        a_dot_D = torch.where(left, a_dot_left_D, a_dot_right_D)
        b_dot_D = torch.where(left, b_dot_left_D, b_dot_right_D)
        c_dot_D = torch.where(left, c_dot_left_D, c_dot_right_D)

        a_FM = 1.0 - t
        b_FM = torch.zeros_like(t)
        c_FM = t

        a_dot_FM = -torch.ones_like(t)
        b_dot_FM = torch.zeros_like(t)
        c_dot_FM = torch.ones_like(t)

        a = (1.0 - lam) * a_D + lam * a_FM
        b = (1.0 - lam) * b_D + lam * b_FM
        c = (1.0 - lam) * c_D + lam * c_FM

        a_dot = (1.0 - lam) * a_dot_D + lam * a_dot_FM
        b_dot = (1.0 - lam) * b_dot_D + lam * b_dot_FM
        c_dot = (1.0 - lam) * c_dot_D + lam * c_dot_FM

        self._check_path_partition(a, b, c)
        return a, b, c, a_dot, b_dot, c_dot

    def _check_path_partition(self, a, b, c):
        err = torch.max(torch.abs(a + b + c - 1.0)).item()
        if err > self.path_weight_atol:
            raise ValueError(
                f"DGFM path weights must sum to 1; max |a+b+c-1|={err:.3e}"
            )

    def _build_dgfm_interpolants(
        self,
        train_points: torch.Tensor,
        permutation: torch.Tensor,
        sampler: MixtureSamplerV2,
        cluster_sizes: np.ndarray,
        n_t: int,
        interpolation_path: str,
        residual_lambda: float,
    ):
        target = train_points[permutation]
        noise = torch.randn_like(target)
        cluster_ids = self._sample_covering_clusters(
            permutation, sampler.inverse_clusters, cluster_sizes
        )
        intermediate, _ = sampler.sample(len(target), cluster_ids)
        t = self.sample_t(len(target) * n_t)
        dimension = target.shape[1]
        target = target.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, dimension)
        noise = noise.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, dimension)
        intermediate = intermediate.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, dimension)
        a, b, c, a_dot, b_dot, c_dot = self._path_weights(
            t,
            interpolation_path,
            residual_lambda=residual_lambda,
        )
        xt = a[:, None] * noise + b[:, None] * intermediate + c[:, None] * target
        velocity = (
            a_dot[:, None] * noise
            + b_dot[:, None] * intermediate
            + c_dot[:, None] * target
        )
        shuffle = torch.randperm(len(xt), device=self.device)
        return xt[shuffle], t[shuffle], velocity[shuffle]

    def train(
        self,
        target_points: torch.Tensor,
        *,
        n_t: int,
        cluster_size: int,
        max_epochs: int,
        batch_size: int,
        injection_time: float = 0.5,
        interpolation_path: str = "residual-cosine-midpoint",
        residual_lambda: float = 0.4,
        truncation: float = 1.5,
        early_stopping: bool = True,
        stop_criteria: int = 3,
        tolerance: float = 1e-3,
        cluster_jaccard_thresh: float = 0.8,
        cluster_merge_k: int = 10,
        cluster_standardize: bool = True,
        cluster_eps: float = 1e-3,
        cluster_outlier_q: float = 0.9,
        max_pca_samples: int = 2000,
        pca_n_jobs: int = -1,
        progress: bool = True,
    ):
        if injection_time != 0.5:
            raise ValueError(
                "residual-cosine-midpoint requires injection_time=0.5"
            )
        if not 0.0 <= residual_lambda <= 1.0:
            raise ValueError("residual_lambda must lie between zero and one")
        if truncation <= 0:
            raise ValueError("truncation must be positive")
        if not 0.0 < cluster_outlier_q < 1.0:
            raise ValueError("cluster_outlier_q must lie strictly between zero and one")

        self.model = self.model.to(self.device)
        target_points = target_points.to(self.device)
        train_points, validation_points = split_train_validation(target_points)
        train_np = train_points.detach().cpu().numpy()
        clusters, inverse = cluster_points_x(
            train_np,
            cluster_size,
            jaccard_thresh=cluster_jaccard_thresh,
            merge_k=cluster_merge_k,
            standardize=cluster_standardize,
        )
        means, bases, covariances, weights = compute_cluster_pca_full(
            train_np,
            clusters,
            eps=cluster_eps,
            outlier_q=cluster_outlier_q,
            max_pca_samples=max_pca_samples,
            n_jobs=pca_n_jobs,
        )
        sampler = MixtureSamplerV2(
            means,
            bases,
            covariances,
            weights,
            clusters,
            inverse,
            truncation=truncation,
            device=self.device,
        )
        cluster_sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.float64)

        best_w2 = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        records: list[dict] = []
        stale_epochs = 0
        epochs = tqdm(
            range(1, max_epochs + 1),
            desc="DGFMv2 Training",
            unit="epoch",
            disable=not progress,
        )
        for epoch in epochs:
            self.model.train()
            permutation = torch.randperm(len(train_points), device=self.device)
            xt, t, target_velocity = self._build_dgfm_interpolants(
                train_points,
                permutation,
                sampler,
                cluster_sizes,
                n_t,
                interpolation_path,
                residual_lambda,
            )
            loss_sum = 0.0
            batch_count = 0
            for start in range(0, len(xt), batch_size):
                prediction = self.model(xt[start : start + batch_size], t[start : start + batch_size])
                loss = ((prediction - target_velocity[start : start + batch_size]) ** 2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                loss_sum += float(loss.item())
                batch_count += 1

            self.model.eval()
            generated = run_flow(self.model, torch.randn_like(validation_points), self.device)
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
                    "cluster_count": len(clusters),
                }
            )
            if early_stopping and stale_epochs >= stop_criteria:
                break

        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(best_state)
        best_model.eval()
        return best_model, self.model, records, sampler


DGFM = DGFMv2
