"""Mixture of Probabilistic PCA trajectory sampler."""

import torch

from Robot_simulation.models.DGFM_class import (
    MixtureSampler,
    cluster_points_joint,
    compute_cluster_pca_fast_joint,
)


class MPPCA:
    """Fit the DGFM low-rank conditional mixture and sample from it directly."""

    def __init__(
        self,
        *,
        task_name: str,
        horizon: int,
        dof: int,
        condition_dim: int,
        device="cpu",
        gripper_idx=None,
    ):
        self.task_name = task_name
        self.horizon = horizon
        self.dof = dof
        self.condition_dim = condition_dim
        self.device = device
        self.gripper_idx = gripper_idx
        self.mixture_sampler = None

    def train(
        self,
        target_trajectories,
        conditions,
        *,
        cluster_size: int,
        cluster_d: int,
        cluster_jaccard_thresh: float = 0.8,
        cluster_merge_k: int = 10,
        cluster_standardize: bool = True,
        scale_x: float = 1.0,
        scale_c: float = 1.0,
        cluster_eps: float = 1e-3,
        cluster_outlier_q: float = 0.9,
        max_pca_samples: int = 2000,
        pca_n_jobs: int = -1,
        mixture_reg: float = 1e-6,
        mixture_orth_sigma: float = 0.0,
        **_,
    ):
        if max_pca_samples <= 0:
            raise ValueError(f"max_pca_samples must be positive, got {max_pca_samples}")
        if cluster_merge_k <= 0:
            raise ValueError(f"cluster_merge_k must be positive, got {cluster_merge_k}")
        if not 0.0 < cluster_outlier_q < 1.0:
            raise ValueError(f"cluster_outlier_q must be in (0, 1), got {cluster_outlier_q}")

        print("Clustering dataset for MPPCA . . .")
        x_np = target_trajectories.detach().cpu().numpy().reshape(target_trajectories.shape[0], -1)
        c_np = conditions.detach().cpu().numpy()

        clusters, _ = cluster_points_joint(
            x_np,
            c_np,
            m=cluster_size,
            jaccard_thresh=cluster_jaccard_thresh,
            merge_k=cluster_merge_k,
            standardize=cluster_standardize,
            scale_x=scale_x,
            scale_c=scale_c,
        )

        print(f"{len(clusters)} clusters made! Applying PCA . . .")
        mu_x, mu_c, B, Szz, Szc, Scc, weights = compute_cluster_pca_fast_joint(
            x_np,
            c_np,
            clusters,
            d_x=cluster_d,
            eps=cluster_eps,
            outlier_q=cluster_outlier_q,
            max_pca_samples=max_pca_samples,
            n_jobs=pca_n_jobs,
        )

        self.mixture_sampler = MixtureSampler(
            mu_x,
            mu_c,
            B,
            Szz,
            Szc,
            Scc,
            weights,
            device=self.device,
            reg=mixture_reg,
            orth_sigma=mixture_orth_sigma,
        )
        return self, self, {}, self.mixture_sampler

    @torch.no_grad()
    def sample(
        self,
        conditions,
        *,
        deterministic_component: bool = False,
        truncated: bool = True,
        trunc=(-1.5, 1.5),
    ):
        if self.mixture_sampler is None:
            raise RuntimeError("MPPCA must be trained before sampling.")

        x_flat, _, _ = self.mixture_sampler.sample_cond(
            conditions,
            deterministic_component=deterministic_component,
            truncated=truncated,
            trunc=trunc,
        )
        return x_flat.reshape(-1, self.horizon, self.dof)

    def state_dict(self):
        if self.mixture_sampler is None:
            raise RuntimeError("MPPCA must be trained before serialization.")
        sampler = self.mixture_sampler
        return {
            "task_name": self.task_name,
            "horizon": self.horizon,
            "dof": self.dof,
            "condition_dim": self.condition_dim,
            "gripper_idx": self.gripper_idx,
            "mu_x": sampler.mu_x.detach().cpu(),
            "mu_c": sampler.mu_c.detach().cpu(),
            "B": sampler.B.detach().cpu(),
            "Sig_zz": sampler.Szz.detach().cpu(),
            "Sig_zc": sampler.Szc.detach().cpu(),
            "Sig_cc": sampler.Scc.detach().cpu(),
            "weights": sampler.weights.detach().cpu(),
            "reg": sampler.reg,
            "orth_sigma": sampler.orth_sigma,
        }

    def save(self, path: str):
        torch.save(self.state_dict(), path)
