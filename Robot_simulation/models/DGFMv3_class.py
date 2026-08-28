"""DGFMv3 with joint action/proprioception/pixel-PCA condition geometry."""

import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from Robot_simulation.models.DGFM_class import (
    DGFM,
    _standardize_cols,
    cluster_points_joint,
)
from Robot_simulation.env_util import _generate_val_env, eval_model
from Robot_simulation.models.VanillaFM_class import VectorField, normalize_accumulated_gradients
from Robot_simulation.models.vision_encoder import load_image_path_windows
from Robot_simulation.reproducibility import validation_result_is_better


def _incremental_batch_slices(num_samples, batch_size, min_batch_size):
    """Return slices whose final chunk is large enough for IncrementalPCA."""
    if num_samples <= batch_size:
        return [(0, num_samples)]
    starts = list(range(0, num_samples, batch_size))
    if num_samples - starts[-1] < min_batch_size and len(starts) > 1:
        starts.pop()
    return [
        (start, starts[i + 1] if i + 1 < len(starts) else num_samples)
        for i, start in enumerate(starts)
    ]


class PixelPCABank:
    """Camera-wise global pixel PCA and compact per-window coefficients."""

    def __init__(self, means, components, image_size, ranks):
        self.means = torch.as_tensor(means, dtype=torch.float32)
        self.components = torch.as_tensor(components, dtype=torch.float32)
        self.image_size = int(image_size)
        self.ranks = tuple(int(rank) for rank in ranks)
        self.num_views = int(self.means.shape[0])
        self.rank = int(self.components.shape[1])
        self._device_cache = {}

    @classmethod
    def fit_transform(
        cls,
        path_windows,
        dataset_dir,
        *,
        rank,
        image_size,
        batch_size,
        max_fit_images=0,
        seed=0,
    ):
        paths = np.asarray(path_windows, dtype=object)
        if paths.ndim != 3:
            raise ValueError(f"Expected vision paths shaped (N, O, V), got {paths.shape}")
        num_windows, observation_horizon, num_views = paths.shape
        flat_dim = 3 * image_size * image_size
        rank = min(int(rank), flat_dim)
        if rank <= 0:
            raise ValueError(f"Pixel PCA rank must be positive, got {rank}")

        all_means = []
        all_components = []
        all_ranks = []
        window_scores = np.zeros(
            (num_windows, observation_horizon, num_views, rank),
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed)

        for view_idx in range(num_views):
            flat_paths = paths[:, :, view_idx].reshape(-1)
            unique_paths = np.unique(flat_paths)
            fit_paths = unique_paths
            if max_fit_images and unique_paths.shape[0] > max_fit_images:
                selected = np.sort(
                    rng.choice(unique_paths.shape[0], max_fit_images, replace=False)
                )
                fit_paths = unique_paths[selected]
            if fit_paths.shape[0] < 2:
                raise ValueError(
                    "Pixel PCA requires at least two distinct images per camera; "
                    f"view {view_idx} has {fit_paths.shape[0]}"
                )
            effective_rank = min(rank, max(1, fit_paths.shape[0] - 1))
            effective_batch = max(int(batch_size), effective_rank)
            ipca = IncrementalPCA(
                n_components=effective_rank,
                batch_size=effective_batch,
                whiten=False,
            )

            print(
                f"[DGFMv3 pixel-PCA] view={view_idx} unique={len(unique_paths)} "
                f"fit={len(fit_paths)} size={image_size} rank={effective_rank}"
            )
            for start, stop in _incremental_batch_slices(
                len(fit_paths), effective_batch, effective_rank
            ):
                batch_paths = np.asarray(fit_paths[start:stop], dtype=object).reshape(-1, 1, 1)
                images = load_image_path_windows(
                    batch_paths,
                    dataset_dir,
                    cache_images=False,
                    resize_hw=(image_size, image_size),
                    device="cpu",
                )
                matrix = images.reshape(images.shape[0], -1).numpy().astype(np.float32)
                matrix /= 255.0
                ipca.partial_fit(matrix)

            mean = ipca.mean_.astype(np.float32, copy=False)
            components = np.zeros((rank, flat_dim), dtype=np.float32)
            components[:effective_rank] = ipca.components_.astype(np.float32, copy=False)
            all_means.append(mean)
            all_components.append(components)
            all_ranks.append(effective_rank)

            score_lookup = {}
            transform_batch = max(int(batch_size), 1)
            for start in range(0, len(unique_paths), transform_batch):
                stop = min(start + transform_batch, len(unique_paths))
                batch_paths = np.asarray(unique_paths[start:stop], dtype=object).reshape(-1, 1, 1)
                images = load_image_path_windows(
                    batch_paths,
                    dataset_dir,
                    cache_images=False,
                    resize_hw=(image_size, image_size),
                    device="cpu",
                )
                matrix = images.reshape(images.shape[0], -1).numpy().astype(np.float32)
                matrix /= 255.0
                scores = np.zeros((matrix.shape[0], rank), dtype=np.float32)
                scores[:, :effective_rank] = ipca.transform(matrix).astype(
                    np.float32, copy=False
                )
                score_lookup.update(
                    (str(path), score)
                    for path, score in zip(unique_paths[start:stop], scores)
                )
            window_scores[:, :, view_idx, :] = np.stack(
                [score_lookup[str(path)] for path in flat_paths],
                axis=0,
            ).reshape(num_windows, observation_horizon, rank)

        bank = cls(
            np.stack(all_means, axis=0),
            np.stack(all_components, axis=0),
            image_size,
            all_ranks,
        )
        return bank, window_scores.reshape(num_windows, -1)

    def _tensors(self, device):
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = (
                self.means.to(device),
                self.components.to(device),
            )
        return self._device_cache[key]

    def reconstruct(self, coefficients):
        """Reconstruct (B, O, V, H, W, 3) float images from global scores."""
        batch = coefficients.shape[0]
        if coefficients.shape[-1] % (self.num_views * self.rank) != 0:
            raise ValueError(
                f"Coefficient dimension {coefficients.shape[-1]} is not divisible by "
                f"views*rank={self.num_views * self.rank}"
            )
        observation_horizon = coefficients.shape[-1] // (self.num_views * self.rank)
        q = coefficients.reshape(
            batch, observation_horizon, self.num_views, self.rank
        )
        means, components = self._tensors(coefficients.device)
        pixels = means.view(1, 1, self.num_views, -1) + torch.einsum(
            "bovr,vrd->bovd", q, components
        )
        return pixels.reshape(
            batch,
            observation_horizon,
            self.num_views,
            self.image_size,
            self.image_size,
            3,
        )


class JointPixelPCASampler:
    """Cluster mixture over full action PCA, local image PCA, and proprioception."""

    def __init__(
        self,
        mu_x,
        mu_q,
        mu_p,
        basis_x,
        basis_q,
        latent_cholesky,
        weights,
        *,
        device,
        orth_sigma=0.0,
    ):
        self.device = device
        self.mu_x = torch.as_tensor(mu_x, dtype=torch.float32, device=device)
        self.mu_q = torch.as_tensor(mu_q, dtype=torch.float32, device=device)
        self.mu_p = torch.as_tensor(mu_p, dtype=torch.float32, device=device)
        self.basis_x = torch.as_tensor(basis_x, dtype=torch.float32, device=device)
        self.basis_q = torch.as_tensor(basis_q, dtype=torch.float32, device=device)
        self.latent_cholesky = torch.as_tensor(
            latent_cholesky, dtype=torch.float32, device=device
        )
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.weights = weight_tensor / weight_tensor.sum()
        self.orth_sigma = float(orth_sigma)
        self.dx = int(self.mu_x.shape[1])
        self.dq = int(self.mu_q.shape[1])
        self.dp = int(self.mu_p.shape[1])
        self.image_rank = int(self.basis_q.shape[2])
        self.latent_dim = self.dx + self.image_rank + self.dp

    @torch.no_grad()
    def sample_joint(self, count, *, pis, truncated=True, trunc=(-1.5, 1.5)):
        pis = torch.as_tensor(pis, dtype=torch.long, device=self.device)
        x_out = torch.empty(count, self.dx, device=self.device)
        q_out = torch.empty(count, self.dq, device=self.device)
        p_out = torch.empty(count, self.dp, device=self.device)
        for cluster_idx in pis.unique(sorted=True).tolist():
            mask = pis == cluster_idx
            cluster_count = int(mask.sum().item())
            if truncated:
                eps = torch.from_numpy(
                    truncnorm.rvs(
                        trunc[0],
                        trunc[1],
                        size=(cluster_count, self.latent_dim),
                    ).astype(np.float32)
                ).to(self.device)
            else:
                eps = torch.randn(cluster_count, self.latent_dim, device=self.device)
            latent = eps @ self.latent_cholesky[cluster_idx].T
            zx = latent[:, :self.dx]
            zq = latent[:, self.dx:self.dx + self.image_rank]
            zp = latent[:, self.dx + self.image_rank:]
            x = self.mu_x[cluster_idx] + zx @ self.basis_x[cluster_idx].T
            if self.orth_sigma > 0:
                x = x + self.orth_sigma * torch.randn_like(x)
            q = self.mu_q[cluster_idx] + zq @ self.basis_q[cluster_idx].T
            p = self.mu_p[cluster_idx] + zp
            x_out[mask], q_out[mask], p_out[mask] = x, q, p
        return x_out, q_out, p_out


def _fit_cluster_latent_models(X, Q, P, clusters, image_rank, eps):
    """Fit local full action PCA and low-rank global-image-score PCA."""
    _, dx = X.shape
    dq = Q.shape[1]
    dp = P.shape[1]
    image_rank = min(int(image_rank), dq)
    if image_rank <= 0 and dq > 0:
        raise ValueError(f"Local image rank must be positive, got {image_rank}")

    records = []
    for cluster in tqdm(clusters, desc="DGFMv3 local PCA", unit="cluster"):
        idx = np.asarray(sorted(cluster), dtype=np.int64)
        Xi, Qi, Pi = X[idx], Q[idx], P[idx]
        mu_x, mu_q, mu_p = Xi.mean(0), Qi.mean(0), Pi.mean(0)
        Xc, Qc, Pc = Xi - mu_x, Qi - mu_q, Pi - mu_p

        cov_x = (Xc.T @ Xc) / max(1, len(idx) - 1) + eps * np.eye(dx)
        _, eigenvectors = np.linalg.eigh(cov_x)
        basis_x = eigenvectors[:, ::-1].astype(np.float32, copy=True)
        zx = Xc @ basis_x

        if dq > 0:
            _, _, vt = np.linalg.svd(Qc, full_matrices=False)
            available_rank = min(image_rank, vt.shape[0])
            basis_q = np.zeros((dq, image_rank), dtype=np.float32)
            basis_q[:, :available_rank] = vt[:available_rank].T.astype(
                np.float32, copy=False
            )
            zq = Qc @ basis_q
        else:
            basis_q = np.zeros((0, 0), dtype=np.float32)
            zq = np.zeros((len(idx), 0), dtype=np.float32)

        latent = np.concatenate([zx, zq, Pc], axis=1)
        covariance = (latent.T @ latent) / max(1, len(idx) - 1)
        covariance += eps * np.eye(covariance.shape[0])
        covariance = 0.5 * (covariance + covariance.T)
        jitter = eps
        for _ in range(6):
            try:
                latent_cholesky = np.linalg.cholesky(
                    covariance + jitter * np.eye(covariance.shape[0])
                )
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise np.linalg.LinAlgError("Could not regularize DGFMv3 latent covariance")
        records.append(
            (
                mu_x,
                mu_q,
                mu_p,
                basis_x,
                basis_q,
                latent_cholesky,
                len(idx) / len(X),
            )
        )

    mu_x, mu_q, mu_p, bx, bq, chol, weights = zip(*records)
    return (
        np.stack(mu_x),
        np.stack(mu_q),
        np.stack(mu_p),
        np.stack(bx),
        np.stack(bq),
        np.stack(chol),
        np.asarray(weights, dtype=np.float32),
    )


class DGFMv3(DGFM):
    """DGFM trainer with a joint PCA intermediate and a moving condition path."""

    def _condition_path_weights(
        self,
        t,
        condition_interpolation_path="piecewise-linear-half",
    ):
        """Return weights for C_t = w_pca(t) * C_tilde + w_demo(t) * C_demo."""
        if condition_interpolation_path == "piecewise-linear-half":
            w_pca = torch.clamp(1.0 - 2.0 * t, min=0.0)
            w_demo = 1.0 - w_pca
            return w_pca.reshape(-1), w_demo.reshape(-1)
        raise ValueError(
            "Unsupported DGFMv3 condition interpolation path "
            f"{condition_interpolation_path!r}"
        )

    def _build_joint_interpolants(
        self,
        target_trajectories,
        conditions,
        perm_t,
        mixture_sampler,
        inv_cluster,
        cluster_sizes,
        n_t,
        interpolation_path,
        condition_interpolation_path="piecewise-linear-half",
        residual_lambda=0.2,
        dgfm_truncated=True,
        dgfm_trunc_low=-1.5,
        dgfm_trunc_high=1.5,
    ):
        idx = perm_t[:target_trajectories.shape[0]]
        x = target_trajectories[idx]
        demo_p = conditions[idx, :]
        m = x.shape[0]

        z = torch.randn(m, self.horizon, self.dof, device=self.device)
        pis = self._sample_covering_clusters(idx, inv_cluster, cluster_sizes)
        y_flat, tilde_q, tilde_p = mixture_sampler.sample_joint(
            m,
            truncated=dgfm_truncated,
            trunc=(dgfm_trunc_low, dgfm_trunc_high),
            pis=pis,
        )
        y = y_flat.reshape(m, self.horizon, self.dof)

        t = self.sample_t(m * n_t)
        xr = x.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        yr = y.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        zr = z.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        tilde_qr = tilde_q.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, tilde_q.shape[-1])
        tilde_pr = tilde_p.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, tilde_p.shape[-1])
        demo_pr = demo_p.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, demo_p.shape[-1])
        source_idx = idx.unsqueeze(1).expand(-1, n_t).reshape(-1)

        a, b, cc, a_dot, b_dot, c_dot = self._path_weights(
            t, interpolation_path, residual_lambda=residual_lambda
        )
        xt = a.view(-1, 1, 1) * zr + b.view(-1, 1, 1) * yr + cc.view(-1, 1, 1) * xr
        vt = (
            a_dot.view(-1, 1, 1) * zr
            + b_dot.view(-1, 1, 1) * yr
            + c_dot.view(-1, 1, 1) * xr
        )
        w_pca, w_demo = self._condition_path_weights(
            t,
            condition_interpolation_path=condition_interpolation_path,
        )

        perm = torch.randperm(xt.shape[0], device=self.device)
        return (
            xt[perm],
            t[perm],
            vt[perm],
            tilde_qr[perm],
            tilde_pr[perm],
            demo_pr[perm],
            w_pca[perm],
            w_demo[perm],
            source_idx[perm],
        )

    def _resize_reconstruction(self, images, target_size=224):
        if images.shape[-3:-1] == (target_size, target_size):
            return images
        batch, obs, views, height, width, channels = images.shape
        images = images.permute(0, 1, 2, 5, 3, 4).reshape(
            batch * obs * views, channels, height, width
        )
        images = F.interpolate(
            images,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        return images.reshape(
            batch, obs, views, channels, target_size, target_size
        ).permute(0, 1, 2, 4, 5, 3)

    def _condition_batch(
        self,
        source_idx,
        tilde_q,
        tilde_p,
        demo_p,
        w_pca,
        w_demo,
    ):
        # sample_t() returns (B, 1), while condition interpolation weights are
        # per-sample scalars. Keep them one-dimensional before broadcasting
        # over proprioception and pixel tensors.
        w_pca = w_pca.reshape(-1)
        w_demo = w_demo.reshape(-1)
        p_t = w_pca.unsqueeze(1) * tilde_p + w_demo.unsqueeze(1) * demo_p
        if self.pixel_pca_bank is None:
            return p_t

        paths = self.vision_window_paths[source_idx.detach().cpu().numpy()]
        observation_horizon = paths.shape[1]
        policy_chunk = max(
            1,
            int(getattr(self, "vision_encoder_batch_size", len(paths)))
            // observation_horizon,
        )
        condition_chunks = []
        for start in range(0, len(paths), policy_chunk):
            stop = min(start + policy_chunk, len(paths))
            demo_images = load_image_path_windows(
                paths[start:stop],
                self.vision_dataset_dir,
                cache_images=True,
                resize_hw=(224, 224),
                device=self.device,
            ).float()
            demo_images /= 255.0
            tilde_images = self.pixel_pca_bank.reconstruct(
                tilde_q[start:stop]
            ).clamp_(0.0, 1.0)
            tilde_images = self._resize_reconstruction(tilde_images, target_size=224)

            wp = w_pca[start:stop].view(-1, 1, 1, 1, 1, 1)
            wd = w_demo[start:stop].view(-1, 1, 1, 1, 1, 1)
            interpolated = wp * tilde_images + wd * demo_images
            batch = stop - start
            encoded = self.model.vision_encoder(
                interpolated.reshape(
                    batch * observation_horizon,
                    interpolated.shape[2],
                    224,
                    224,
                    3,
                )
            ).reshape(batch, -1)
            condition_chunks.append(
                torch.cat([p_t[start:stop], encoded], dim=1)
            )

            if self.condition_example is None:
                active = torch.nonzero(w_pca[start:stop] > 0, as_tuple=False)
                if active.numel():
                    example_idx = int(active[0].item())
                    example_w_pca = float(w_pca[start + example_idx].detach().cpu())
                    example_t = 0.5 * (1.0 - example_w_pca)
                    self.condition_example = {
                        "tilde": tilde_images[example_idx].detach().cpu(),
                        "interpolated": interpolated[example_idx].detach().cpu(),
                        "demo": demo_images[example_idx].detach().cpu(),
                        "t": example_t,
                    }
        return torch.cat(condition_chunks, dim=0)

    def save_condition_example(self, output_dir):
        """Save one tilde/demo pixel control-path example to the result folder."""
        if self.condition_example is None:
            return None
        import matplotlib.pyplot as plt

        tilde = self.condition_example["tilde"].numpy()
        mixed = self.condition_example["interpolated"].numpy()
        demo = self.condition_example["demo"].numpy()
        observation_horizon, num_views = tilde.shape[:2]
        rows = observation_horizon * num_views
        fig, axes = plt.subplots(
            rows,
            3,
            figsize=(9, max(2.5, 2.5 * rows)),
            squeeze=False,
        )
        camera_names = list(getattr(self.model.vision_encoder, "camera_names", []))
        for obs_idx in range(observation_horizon):
            for view_idx in range(num_views):
                row = obs_idx * num_views + view_idx
                for col, image in enumerate(
                    (tilde[obs_idx, view_idx], mixed[obs_idx, view_idx], demo[obs_idx, view_idx])
                ):
                    axes[row, col].imshow(np.clip(image, 0.0, 1.0))
                    axes[row, col].axis("off")
                view_name = (
                    camera_names[view_idx]
                    if view_idx < len(camera_names)
                    else f"view_{view_idx}"
                )
                axes[row, 0].set_ylabel(f"obs={obs_idx}\n{view_name}")
        axes[0, 0].set_title(r"$\tilde{I}$")
        axes[0, 1].set_title(
            rf"$I_t$ actually used at $t={self.condition_example['t']:.3f}$"
        )
        axes[0, 2].set_title(r"$I_i$")
        fig.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "dgfmv3_condition_example.png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path

    def train(
        self,
        target_trajectories,
        conditions,
        *,
        n_t: int,
        cluster_size: int,
        max_epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        interpolation_path: str = "beizer",
        condition_interpolation_path: str = "piecewise-linear-half",
        residual_lambda: float = 0.2,
        val_period: int = 5,
        early_stopping: bool = True,
        stop_criteria: int = 3,
        scale_x: float = 1.0,
        scale_c: float = 1.0,
        val_trials: int = 25,
        mf: int | None = None,
        n_t_local: int | None = None,
        n_t_global: int | None = None,
        cluster_jaccard_thresh: float = 0.8,
        cluster_merge_k: int = 10,
        cluster_standardize: bool = True,
        cluster_eps: float = 1e-3,
        cluster_outlier_q: float = 0.9,
        max_pca_samples: int = 2000,
        pca_n_jobs: int = -1,
        vision_pca_global_rank: int = 64,
        vision_pca_local_rank: int = 32,
        vision_pca_image_size: int = 64,
        vision_pca_batch_size: int = 128,
        vision_pca_max_images: int = 0,
        mixture_reg: float = 1e-6,
        mixture_orth_sigma: float = 0.0,
        dgfm_truncated: bool = True,
        dgfm_trunc_low: float = -1.5,
        dgfm_trunc_high: float = 1.5,
        max_policy_steps: int = 20,
        executed_horizon: int | None = None,
        observation_horizon: int = 1,
        eval_base_seed: int = 123,
        recorded_control_freq: int | float | None = None,
        trajectory_control_freq: int | float | None = None,
        evaluator=None,
        eval_metadata: dict | None = None,
        **_,
    ):
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive, got "
                f"{gradient_accumulation_steps}"
            )
        if mf is not None or n_t_local is not None or n_t_global is not None:
            print("[DGFMv3] Ignoring deprecated mf/n_t_local/n_t_global; using n_t only.")
        if max_pca_samples <= 0:
            raise ValueError(f"max_pca_samples must be positive, got {max_pca_samples}")
        if vision_pca_global_rank <= 0 or vision_pca_local_rank <= 0:
            raise ValueError("DGFMv3 vision PCA ranks must be positive")
        if vision_pca_image_size <= 0 or vision_pca_batch_size <= 0:
            raise ValueError("DGFMv3 vision PCA image/batch sizes must be positive")
        if vision_pca_max_images < 0:
            raise ValueError("vision_pca_max_images must be non-negative")
        if cluster_merge_k <= 0:
            raise ValueError(f"cluster_merge_k must be positive, got {cluster_merge_k}")
        if not 0.0 < cluster_outlier_q < 1.0:
            raise ValueError(f"cluster_outlier_q must be in (0, 1), got {cluster_outlier_q}")
        if dgfm_trunc_low >= dgfm_trunc_high:
            raise ValueError(
                f"dgfm_trunc_low must be smaller than dgfm_trunc_high, "
                f"got {dgfm_trunc_low} >= {dgfm_trunc_high}"
            )

        print("Preparing pixel-space conditions for DGFMv3 . . .")
        self.model = self.model.to(self.device)
        X_np = target_trajectories.detach().cpu().numpy().reshape(
            target_trajectories.shape[0], -1
        )
        P_np = conditions.detach().cpu().numpy().astype(np.float32, copy=False)

        vision_paths = getattr(self, "vision_window_paths", None)
        if hasattr(self.model, "vision_encoder") and vision_paths is None:
            raise RuntimeError(
                "Vision-conditioned DGFMv3 requires aligned vision_window_paths"
            )
        if vision_paths is not None:
            self.pixel_pca_bank, Q_np = PixelPCABank.fit_transform(
                vision_paths,
                self.vision_dataset_dir,
                rank=vision_pca_global_rank,
                image_size=vision_pca_image_size,
                batch_size=vision_pca_batch_size,
                max_fit_images=vision_pca_max_images,
                seed=eval_base_seed,
            )
        else:
            self.pixel_pca_bank = None
            Q_np = np.zeros((len(X_np), 0), dtype=np.float32)
        self.condition_example = None

        def normalized_block(values):
            if values.shape[1] == 0:
                return values
            standardized = _standardize_cols(values)[0] if cluster_standardize else values
            return standardized / math.sqrt(values.shape[1])

        X_cluster = normalized_block(X_np)
        P_cluster = normalized_block(P_np)
        Q_cluster = normalized_block(Q_np)
        C_cluster = np.concatenate([P_cluster, Q_cluster], axis=1)

        clusters, inv_cluster = cluster_points_joint(
            X_cluster,
            C_cluster,
            m=cluster_size,
            jaccard_thresh=cluster_jaccard_thresh,
            merge_k=cluster_merge_k,
            standardize=False,
            scale_x=scale_x,
            scale_c=scale_c,
        )
        cluster_sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.float64)

        print(
            f"{len(clusters)} clusters made! Applying full X PCA and "
            "cluster-local image-score PCA . . ."
        )
        local_rank = min(vision_pca_local_rank, Q_np.shape[1])
        mu_x, mu_q, mu_p, basis_x, basis_q, latent_cholesky, weights = (
            _fit_cluster_latent_models(
                X_np,
                Q_np,
                P_np,
                clusters,
                image_rank=local_rank,
                eps=max(cluster_eps, mixture_reg),
            )
        )

        mixture_sampler = JointPixelPCASampler(
            mu_x,
            mu_q,
            mu_p,
            basis_x,
            basis_q,
            latent_cholesky,
            weights,
            device=self.device,
            orth_sigma=mixture_orth_sigma,
        )
        self.pixel_pca_metadata = {
            "global_rank": int(vision_pca_global_rank),
            "local_rank": int(local_rank),
            "image_size": int(vision_pca_image_size),
            "batch_size": int(vision_pca_batch_size),
            "max_fit_images": int(vision_pca_max_images),
            "camera_ranks": (
                list(self.pixel_pca_bank.ranks)
                if self.pixel_pca_bank is not None
                else []
            ),
        }

        N = target_trajectories.shape[0]
        joint_N = N * n_t
        best_avg_reward = 0.0
        best_success_rate = 0.0
        best_model = self._copy_eval_model()
        success_rate_recs = {}
        stop_count = 0
        best_validation_rollouts = None
        self.best_validation_rollouts = None

        do_validation = val_period > 0 and val_trials > 0
        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials, eval_base_seed)

        try:
            self.model = self.model.to(self.device)
            self._init_ema()
            if self.use_ema and self.ema is None:
                raise RuntimeError("DGFMv3 EMA initialization failed while use_ema=True")
            target_trajectories = target_trajectories.to(self.device)
            conditions = conditions.to(self.device)

            for epoch in tqdm(range(1, max_epochs + 1), desc="DGFMv3 Training", unit="epoch"):
                self.model.train()
                vision_epoch_hook = getattr(self, "vision_epoch_hook", None)
                if vision_epoch_hook is not None:
                    vision_epoch_hook(epoch)
                perm_t = torch.randperm(N, device=self.device)

                (
                    XT,
                    TIN,
                    VT,
                    TILDE_Q,
                    TILDE_P,
                    DEMO_P,
                    W_PCA,
                    W_DEMO,
                    SOURCE_IDX,
                ) = self._build_joint_interpolants(
                    target_trajectories=target_trajectories,
                    conditions=conditions,
                    perm_t=perm_t,
                    mixture_sampler=mixture_sampler,
                    inv_cluster=inv_cluster,
                    cluster_sizes=cluster_sizes,
                    n_t=n_t,
                    interpolation_path=interpolation_path,
                    condition_interpolation_path=condition_interpolation_path,
                    residual_lambda=residual_lambda,
                    dgfm_truncated=dgfm_truncated,
                    dgfm_trunc_low=dgfm_trunc_low,
                    dgfm_trunc_high=dgfm_trunc_high,
                )

                loss_sum = 0.0
                batch_count = 0
                accumulated_microbatches = 0
                accumulated_samples = 0
                self.optimizer.zero_grad()
                for i in range(0, joint_N, batch_size):
                    xb = XT[i:i + batch_size]
                    tb = TIN[i:i + batch_size]
                    vb = VT[i:i + batch_size]
                    tilde_qb = TILDE_Q[i:i + batch_size]
                    tilde_pb = TILDE_P[i:i + batch_size]
                    demo_pb = DEMO_P[i:i + batch_size]
                    w_pca = W_PCA[i:i + batch_size]
                    w_demo = W_DEMO[i:i + batch_size]
                    source_idx = SOURCE_IDX[i:i + batch_size]

                    with torch.enable_grad():
                        cb = self._condition_batch(
                            source_idx,
                            tilde_qb,
                            tilde_pb,
                            demo_pb,
                            w_pca,
                            w_demo,
                        )
                        pred = self.model(xb, tb, cb)
                        sq_err = (pred - vb) ** 2
                        if hasattr(self.model, "loss_mask") and self.model.loss_mask is not None:
                            sq_err = sq_err * self.model.loss_mask
                        loss = sq_err.mean()

                        microbatch_samples = xb.shape[0]
                        (loss * microbatch_samples).backward()
                        accumulated_microbatches += 1
                        accumulated_samples += microbatch_samples
                        accumulation_boundary = (
                            accumulated_microbatches == gradient_accumulation_steps
                            or i + batch_size >= joint_N
                        )
                        if accumulation_boundary:
                            normalize_accumulated_gradients(
                                self.optimizer, accumulated_samples
                            )
                            vision_after_backward_hook = getattr(
                                self, "vision_after_backward_hook", None
                            )
                            if vision_after_backward_hook is not None:
                                vision_after_backward_hook()
                            self.optimizer.step()
                            self._step_ema()
                            self.optimizer.zero_grad()
                            accumulated_microbatches = 0
                            accumulated_samples = 0

                        loss_sum += float(loss.item())
                        batch_count += 1

                if self.scheduler is not None:
                    self.scheduler.step()

                avg_loss = loss_sum / max(1, batch_count)
                if do_validation and epoch % val_period == 0:
                    eval_model_obj = self._eval_model()
                    eval_model_obj.eval()
                    eval_kwargs = dict(
                        model=eval_model_obj,
                        model_class=VectorField,
                        task_name=self.task_name,
                        seq_len=self.horizon,
                        dof=self.dof,
                        param_len=self.condition_dim,
                        gripper_idx=self.gripper_idx,
                        val_params=val_params,
                        env_settings_all=env_settings_all,
                        device=self.device,
                        trials=val_trials,
                        base_seed=eval_base_seed,
                        max_policy_steps=max_policy_steps,
                        executed_horizon=executed_horizon,
                        observation_horizon=observation_horizon,
                        recorded_control_freq=recorded_control_freq,
                        trajectory_control_freq=trajectory_control_freq,
                        normalization_stats=self.normalization_stats,
                        return_rollouts=True,
                        action_representation=getattr(self, "action_representation", "joint_space"),
                    )
                    if evaluator is None:
                        success_rate, avg_reward, validation_rollouts = eval_model(**eval_kwargs)
                    else:
                        metadata = dict(eval_metadata or {})
                        metadata.update(
                            eval_kwargs={k: v for k, v in eval_kwargs.items() if k != "model"},
                            val_trials=val_trials,
                            eval_base_seed=eval_base_seed,
                            normalization_stats=self.normalization_stats,
                        )
                        response = evaluator.evaluate(eval_model_obj, epoch, metadata)
                        success_rate = response.success_rate
                        avg_reward = response.avg_reward
                        validation_rollouts = response.validation_rollouts

                    if not torch.is_grad_enabled():
                        torch.set_grad_enabled(True)

                    success_rate_recs[epoch] = {
                        "success_rate": success_rate,
                        "avg_reward": avg_reward,
                        "loss": avg_loss,
                    }

                    if success_rate < best_success_rate:
                        tqdm.write(
                            f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                            f"avg reward={avg_reward:.3f}, loss={avg_loss:.3f}"
                        )
                        if early_stopping:
                            if stop_count == stop_criteria:
                                tqdm.write("Early stopping triggered.")
                                break
                            stop_count += 1
                    else:
                        if validation_result_is_better(
                            success_rate, avg_reward, best_success_rate, best_avg_reward,
                            has_best=best_validation_rollouts is not None,
                        ):
                            best_avg_reward = avg_reward
                            best_success_rate = success_rate
                            best_model = self._copy_eval_model()
                            best_validation_rollouts = validation_rollouts
                            self.best_validation_rollouts = best_validation_rollouts
                            stop_count = 0
                            tqdm.write(
                                f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                                f"avg reward={avg_reward:.3f}, loss={avg_loss:.3f} | Best model saved"
                            )
                        else:
                            tqdm.write(
                                f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                                f"avg reward={avg_reward:.3f}, loss={avg_loss:.3f}"
                            )

        except KeyboardInterrupt:
            tqdm.write("Training interrupted by user. Returning best model so far...")

        if not success_rate_recs:
            best_model = self._copy_eval_model()
        return best_model, self.model, success_rate_recs, mixture_sampler
