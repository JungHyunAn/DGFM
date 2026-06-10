"""DGFMv2 utilities with X-only intermediate trajectory distributions."""

import copy

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import chi2, truncnorm
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from Robot_simulation.models.DGFM_class import (
    DGFM,
    _pad_basis_to_rank,
    _pad_square,
    cluster_points_x,
)
from Robot_simulation.env_util import _generate_val_env, eval_model
from Robot_simulation.models.VanillaFM_class import VectorField


class MixtureSamplerV2:
    """Per-cluster PCA trajectory sampler independent of environment C."""

    def __init__(
        self,
        mu_x,
        B,
        Sig_zz,
        weights,
        c_mean,
        c_std,
        device="cpu",
        reg=1e-6,
        orth_sigma=0.0,
    ):
        self.device = device
        self.K, self.Dx = mu_x.shape
        self.d = B.shape[2]

        self.mu_x = torch.as_tensor(mu_x, dtype=torch.float32, device=device)
        self.B = torch.as_tensor(B, dtype=torch.float32, device=device)
        self.Szz = torch.as_tensor(Sig_zz, dtype=torch.float32, device=device)
        self.c_mean = torch.as_tensor(c_mean, dtype=torch.float32, device=device)
        self.c_std = torch.as_tensor(c_std, dtype=torch.float32, device=device)

        w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.weights = (w / w.sum()).clamp_min(1e-12)

        self.reg = float(reg)
        self.orth_sigma = float(orth_sigma)

        eye_z = torch.eye(self.d, device=device)
        lzz = []
        for k in range(self.K):
            szz_k = 0.5 * (self.Szz[k] + self.Szz[k].transpose(0, 1)) + self.reg * eye_z
            lzz.append(torch.linalg.cholesky(szz_k))
        self.Lzz = torch.stack(lzz, dim=0)

    def _sample_x_fixed_k(self, k, m, truncated=False, trunc=(-1.5, 1.5)):
        if truncated:
            lo, hi = trunc
            z_eps = torch.from_numpy(
                truncnorm.rvs(lo, hi, size=(m, self.d)).astype(np.float32)
            ).to(self.device)
        else:
            z_eps = torch.randn(m, self.d, device=self.device)

        z = z_eps @ self.Lzz[k].T
        x = self.mu_x[k].unsqueeze(0) + z @ self.B[k].T
        if self.orth_sigma > 0.0:
            x = x + torch.randn_like(x) * self.orth_sigma
        return x

    @torch.no_grad()
    def sample_cond(self, c_in, deterministic_component=False, truncated=False, trunc=(-1.5, 1.5), pis=None):
        c = torch.as_tensor(c_in, dtype=torch.float32, device=self.device)
        bsz = c.shape[0]
        if pis is None:
            raise ValueError("MixtureSamplerV2.sample_cond requires cluster ids via pis")
        pis = torch.as_tensor(pis, device=self.device, dtype=torch.long)

        x_out = torch.empty(bsz, self.Dx, device=self.device)
        for k in pis.unique(sorted=True).tolist():
            mask = pis == k
            if mask.any():
                x_out[mask] = self._sample_x_fixed_k(
                    k,
                    int(mask.sum().item()),
                    truncated=truncated,
                    trunc=trunc,
                )
        return x_out

    @torch.no_grad()
    def nearest_env_clusters(self, c_in):
        c = torch.as_tensor(c_in, dtype=torch.float32, device=self.device)
        dists = []
        for k in range(self.K):
            delta = c - self.c_mean[k].unsqueeze(0)
            whitened = torch.linalg.solve_triangular(
                self.c_std[k],
                delta.T,
                upper=False,
            )
            dists.append((whitened * whitened).sum(dim=0).unsqueeze(1))
        return torch.argmin(torch.cat(dists, dim=1), dim=1)


def _process_one_cluster_x_only(X, C, idx, eps, chi2_thresh, max_pca_samples):
    Xi = X[idx]
    Ci = C[idx]
    S, Dx = Xi.shape

    mu_x_all = Xi.mean(axis=0)
    X_centered = Xi - mu_x_all
    if S >= 2:
        cov_x = np.cov(Xi, rowvar=False) + eps * np.eye(Dx)
        try:
            L = np.linalg.cholesky(cov_x)
            Y = np.linalg.solve(L, X_centered.T)
            dists = (Y * Y).sum(axis=0)
            mask = dists <= chi2_thresh
            Xi_c = Xi[mask]
            Ci_c = Ci[mask]
        except np.linalg.LinAlgError:
            Xi_c = Xi
            Ci_c = Ci
    else:
        Xi_c = Xi
        Ci_c = Ci

    if Xi_c.shape[0] < 2:
        Xi_c = Xi
        Ci_c = Ci

    if Xi_c.shape[0] > max_pca_samples:
        sel = np.random.choice(Xi_c.shape[0], max_pca_samples, replace=False)
        Xp = Xi_c[sel]
    else:
        Xp = Xi_c

    Xp0 = Xp - Xp.mean(axis=0, keepdims=True)
    col_var = Xp0.var(axis=0)
    keep = col_var > 1e-12

    if Xp.shape[0] < 2 or int(keep.sum()) == 0:
        Bx = np.eye(Dx, dtype=np.float32)
    else:
        Xp_red = Xp0[:, keep]
        n_comp = int(min(Xp_red.shape[1], max(1, Xp_red.shape[0] - 1)))
        ipca = IncrementalPCA(n_components=n_comp, batch_size=min(1024, Xp_red.shape[0]), whiten=False)
        ipca.fit(Xp_red)
        Bx = np.zeros((Dx, n_comp), dtype=np.float32)
        Bx[keep, :] = ipca.components_.T.astype(np.float32, copy=False)
        Bx = _pad_basis_to_rank(Bx, Dx, Dx)

    mu_x = Xi_c.mean(axis=0)
    Zc = (Xi_c - mu_x) @ Bx
    denom = max(1, Zc.shape[0] - 1)
    Sig_zz = (Zc.T @ Zc) / denom + eps * np.eye(Dx)

    c_mean = Ci_c.mean(axis=0)
    C_centered = Ci_c - c_mean
    Dc = Ci_c.shape[1]
    c_cov = (C_centered.T @ C_centered) / denom + eps * np.eye(Dc)
    c_std = np.linalg.cholesky(c_cov).astype(np.float32, copy=False)
    weight = Xi_c.shape[0] / X.shape[0]

    return mu_x, Bx, Sig_zz, weight, c_mean, c_std


def compute_cluster_pca_fast_x_only(
    X,
    C,
    clusters,
    eps=1e-3,
    outlier_q=0.9,
    max_pca_samples=2000,
    n_jobs=-1,
):
    """Fit full-dimensional per-cluster PCA bases on X only."""
    _, Dx = X.shape
    chi2_thresh = chi2.ppf(outlier_q, df=Dx)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_cluster_x_only)(
            X, C, list(c), eps, chi2_thresh, max_pca_samples
        ) for c in clusters
    )

    mu_x, B, Sig_zz, weights, c_mean, c_std = zip(*results)
    mu_x = np.vstack(mu_x).astype(np.float32, copy=False)
    B = np.stack([_pad_basis_to_rank(b, Dx, Dx) for b in B], axis=0)
    Sig_zz = np.stack([_pad_square(szz, Dx, eps) for szz in Sig_zz], axis=0)
    weights = np.asarray(weights, dtype=np.float32)
    c_mean = np.vstack(c_mean).astype(np.float32, copy=False)
    c_std = np.stack(c_std, axis=0).astype(np.float32, copy=False)

    print(f"[DGFMv2] Packed full-dimensional X PCA rank={Dx} for {len(clusters)} clusters.")
    return mu_x, B, Sig_zz, weights, c_mean, c_std


class DGFMv2(DGFM):
    """DGFM trainer whose intermediate distribution models trajectories X only."""

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
        dgfm_truncated=True,
        dgfm_trunc_low=-1.5,
        dgfm_trunc_high=1.5,
    ):
        idx = perm_t[:target_trajectories.shape[0]]
        x = target_trajectories[idx]
        c = conditions[idx, :]
        m = x.shape[0]

        z = torch.randn(m, self.horizon, self.dof, device=self.device)
        pis = self._sample_covering_clusters(idx, inv_cluster, cluster_sizes)
        y_flat = mixture_sampler.sample_cond(
            c,
            truncated=dgfm_truncated,
            trunc=(dgfm_trunc_low, dgfm_trunc_high),
            pis=pis,
        )
        y = y_flat.reshape(m, self.horizon, self.dof)

        t = self.sample_t(m * n_t)
        xr = x.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        yr = y.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        zr = z.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        cr = c.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, self.condition_dim)

        a, b, cc, a_dot, b_dot, c_dot = self._path_weights(t, interpolation_path)
        xt = a.view(-1, 1, 1) * zr + b.view(-1, 1, 1) * yr + cc.view(-1, 1, 1) * xr
        vt = (
            a_dot.view(-1, 1, 1) * zr
            + b_dot.view(-1, 1, 1) * yr
            + c_dot.view(-1, 1, 1) * xr
        )

        perm = torch.randperm(xt.shape[0], device=self.device)
        return xt[perm], t[perm], vt[perm], cr[perm]

    def train(
        self,
        target_trajectories,
        conditions,
        *,
        n_t: int,
        cluster_size: int,
        max_epochs: int,
        batch_size: int,
        interpolation_path: str = "piecewise-linear-midpoint",
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
        **_,
    ):
        if mf is not None or n_t_local is not None or n_t_global is not None:
            print("[DGFMv2] Ignoring deprecated mf/n_t_local/n_t_global; using n_t only.")
        if max_pca_samples <= 0:
            raise ValueError(f"max_pca_samples must be positive, got {max_pca_samples}")
        if cluster_merge_k <= 0:
            raise ValueError(f"cluster_merge_k must be positive, got {cluster_merge_k}")
        if not 0.0 < cluster_outlier_q < 1.0:
            raise ValueError(f"cluster_outlier_q must be in (0, 1), got {cluster_outlier_q}")
        if dgfm_trunc_low >= dgfm_trunc_high:
            raise ValueError(
                f"dgfm_trunc_low must be smaller than dgfm_trunc_high, "
                f"got {dgfm_trunc_low} >= {dgfm_trunc_high}"
            )

        print("Clustering dataset for DGFMv2 . . .")
        X_np = target_trajectories.detach().cpu().numpy().reshape(target_trajectories.shape[0], -1)
        C_np = conditions.detach().cpu().numpy()

        clusters, inv_cluster = cluster_points_x(
            X_np,
            # C_np, # don't use conditional vector for clustering
            m=cluster_size,
            jaccard_thresh=cluster_jaccard_thresh,
            merge_k=cluster_merge_k,
            standardize=cluster_standardize,
            scale_x=scale_x,
            # scale_c=scale_c,
        )
        cluster_sizes = np.asarray([len(c) for c in clusters], dtype=np.float64)

        print(f"{len(clusters)} clusters made! Applying X-only PCA . . .")
        mu_x, B, Szz, weights, c_mean, c_std = compute_cluster_pca_fast_x_only(
            X_np,
            C_np,
            clusters,
            eps=cluster_eps,
            outlier_q=cluster_outlier_q,
            max_pca_samples=max_pca_samples,
            n_jobs=pca_n_jobs,
        )

        mixture_sampler = MixtureSamplerV2(
            mu_x,
            B,
            Szz,
            weights,
            c_mean,
            c_std,
            device=self.device,
            reg=mixture_reg,
            orth_sigma=mixture_orth_sigma,
        )

        N = target_trajectories.shape[0]
        joint_N = N * n_t
        best_avg_reward = 0.0
        best_success_rate = 0.0
        best_model = copy.deepcopy(self.model)
        success_rate_recs = {}
        stop_count = 0
        best_validation_rollouts = None
        self.best_validation_rollouts = None

        do_validation = val_period > 0 and val_trials > 0
        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials)

        try:
            self.model = self.model.to(self.device)
            target_trajectories = target_trajectories.to(self.device)
            conditions = conditions.to(self.device)

            for epoch in tqdm(range(1, max_epochs + 1), desc="DGFMv2 Training", unit="epoch"):
                self.model.train()
                perm_t = torch.randperm(N, device=self.device)

                XT, TIN, VT, CT = self._build_joint_interpolants(
                    target_trajectories=target_trajectories,
                    conditions=conditions,
                    perm_t=perm_t,
                    mixture_sampler=mixture_sampler,
                    inv_cluster=inv_cluster,
                    cluster_sizes=cluster_sizes,
                    n_t=n_t,
                    interpolation_path=interpolation_path,
                    dgfm_truncated=dgfm_truncated,
                    dgfm_trunc_low=dgfm_trunc_low,
                    dgfm_trunc_high=dgfm_trunc_high,
                )

                loss_sum = 0.0
                batch_count = 0
                for i in range(0, joint_N, batch_size):
                    xb = XT[i:i + batch_size]
                    tb = TIN[i:i + batch_size]
                    vb = VT[i:i + batch_size]
                    cb = CT[i:i + batch_size]

                    with torch.enable_grad():
                        pred = self.model(xb, tb, cb)
                        sq_err = (pred - vb) ** 2
                        if hasattr(self.model, "loss_mask") and self.model.loss_mask is not None:
                            sq_err = sq_err * self.model.loss_mask
                        loss = sq_err.mean()

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        loss_sum += float(loss.item())
                        batch_count += 1

                if self.scheduler is not None:
                    self.scheduler.step()

                avg_loss = loss_sum / max(1, batch_count)
                if do_validation and epoch % val_period == 0:
                    self.model.eval()
                    success_rate, avg_reward, validation_rollouts = eval_model(
                        self.model,
                        VectorField,
                        self.task_name,
                        self.horizon,
                        self.dof,
                        self.condition_dim,
                        self.gripper_idx,
                        val_params,
                        env_settings_all,
                        self.device,
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
                        if best_validation_rollouts is None or (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                            best_avg_reward = avg_reward
                            best_success_rate = success_rate
                            best_model = copy.deepcopy(self.model)
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
            best_model = copy.deepcopy(self.model)
        return best_model, self.model, success_rate_recs, mixture_sampler


class MPPCAv2:
    """Sample X-only PCA clusters selected by nearest standardized environment C."""

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
        self.sample_truncated = True
        self.sample_trunc = (-1.5, 1.5)

    def train(
        self,
        target_trajectories,
        conditions,
        *,
        cluster_size: int,
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
        dgfm_truncated: bool = True,
        dgfm_trunc_low: float = -1.5,
        dgfm_trunc_high: float = 1.5,
        **_,
    ):
        if max_pca_samples <= 0:
            raise ValueError(f"max_pca_samples must be positive, got {max_pca_samples}")
        if cluster_merge_k <= 0:
            raise ValueError(f"cluster_merge_k must be positive, got {cluster_merge_k}")
        if not 0.0 < cluster_outlier_q < 1.0:
            raise ValueError(f"cluster_outlier_q must be in (0, 1), got {cluster_outlier_q}")
        if dgfm_trunc_low >= dgfm_trunc_high:
            raise ValueError(
                f"dgfm_trunc_low must be smaller than dgfm_trunc_high, "
                f"got {dgfm_trunc_low} >= {dgfm_trunc_high}"
            )
        self.sample_truncated = dgfm_truncated
        self.sample_trunc = (dgfm_trunc_low, dgfm_trunc_high)

        print("Clustering dataset for MPPCAv2 . . .")
        x_np = target_trajectories.detach().cpu().numpy().reshape(target_trajectories.shape[0], -1)
        c_np = conditions.detach().cpu().numpy()

        clusters, _ = cluster_points_x(
            X_np,
            # C_np, # don't use conditional vector for clustering
            m=cluster_size,
            jaccard_thresh=cluster_jaccard_thresh,
            merge_k=cluster_merge_k,
            standardize=cluster_standardize,
            scale_x=scale_x,
            # scale_c=scale_c,
        )

        print(f"{len(clusters)} clusters made! Applying X-only PCA . . .")
        mu_x, B, Szz, weights, c_mean, c_std = compute_cluster_pca_fast_x_only(
            x_np,
            c_np,
            clusters,
            eps=cluster_eps,
            outlier_q=cluster_outlier_q,
            max_pca_samples=max_pca_samples,
            n_jobs=pca_n_jobs,
        )

        self.mixture_sampler = MixtureSamplerV2(
            mu_x,
            B,
            Szz,
            weights,
            c_mean,
            c_std,
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
        truncated: bool | None = None,
        trunc=None,
    ):
        if self.mixture_sampler is None:
            raise RuntimeError("MPPCAv2 must be trained before sampling.")

        if truncated is None:
            truncated = self.sample_truncated
        if trunc is None:
            trunc = self.sample_trunc

        pis = self.mixture_sampler.nearest_env_clusters(conditions)
        x_flat = self.mixture_sampler.sample_cond(
            conditions,
            deterministic_component=deterministic_component,
            truncated=truncated,
            trunc=trunc,
            pis=pis,
        )
        return x_flat.reshape(-1, self.horizon, self.dof)

    def state_dict(self):
        if self.mixture_sampler is None:
            raise RuntimeError("MPPCAv2 must be trained before serialization.")
        sampler = self.mixture_sampler
        return {
            "task_name": self.task_name,
            "horizon": self.horizon,
            "dof": self.dof,
            "condition_dim": self.condition_dim,
            "gripper_idx": self.gripper_idx,
            "mu_x": sampler.mu_x.detach().cpu(),
            "B": sampler.B.detach().cpu(),
            "Sig_zz": sampler.Szz.detach().cpu(),
            "weights": sampler.weights.detach().cpu(),
            "c_mean": sampler.c_mean.detach().cpu(),
            "c_std": sampler.c_std.detach().cpu(),
            "reg": sampler.reg,
            "orth_sigma": sampler.orth_sigma,
        }

    def eval(self):
        return self

    def save(self, path: str):
        torch.save(self.state_dict(), path)
