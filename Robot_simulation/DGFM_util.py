"""Dimension-guided FM utilities."""

import copy
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import chi2, truncnorm
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from Robot_simulation.FM_util import VanillaFM, VectorField, _generate_val_env, eval_model


class MixtureSampler:
    """Low-rank conditional Gaussian mixture over flattened trajectories."""

    def __init__(
        self,
        mu_x,
        mu_c,
        B,
        Sig_zz,
        Sig_zc,
        Sig_cc,
        weights,
        device="cpu",
        reg=1e-6,
        orth_sigma=0.0,
    ):
        self.device = device
        self.K, self.Dx = mu_x.shape
        self.Dc = mu_c.shape[1]
        self.d = B.shape[2]

        self.mu_x = torch.as_tensor(mu_x, dtype=torch.float32, device=device)
        self.mu_c = torch.as_tensor(mu_c, dtype=torch.float32, device=device)
        self.B = torch.as_tensor(B, dtype=torch.float32, device=device)
        self.Szz = torch.as_tensor(Sig_zz, dtype=torch.float32, device=device)
        self.Szc = torch.as_tensor(Sig_zc, dtype=torch.float32, device=device)
        self.Scc = torch.as_tensor(Sig_cc, dtype=torch.float32, device=device)

        w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.weights = (w / w.sum()).clamp_min(1e-12)

        self.reg = float(reg)
        self.orth_sigma = float(orth_sigma)

        lcc = []
        invcc = []
        logdet_cc = []
        lzgc = []
        eye_c = torch.eye(self.Dc, device=device)
        for k in range(self.K):
            scc_k = self.Scc[k] + self.reg * eye_c
            lcc_k = torch.linalg.cholesky(scc_k)
            invcc_k = torch.cholesky_inverse(lcc_k)
            szgc_k = self.Szz[k] - self.Szc[k] @ invcc_k @ self.Szc[k].transpose(0, 1)
            szgc_k = 0.5 * (szgc_k + szgc_k.transpose(0, 1)) + self.reg * torch.eye(
                szgc_k.shape[0],
                device=device,
            )

            lcc.append(lcc_k)
            invcc.append(invcc_k)
            logdet_cc.append(2.0 * torch.log(torch.diag(lcc_k)).sum())
            lzgc.append(torch.linalg.cholesky(szgc_k))

        self.Lcc = torch.stack(lcc, dim=0)
        self.invcc = torch.stack(invcc, dim=0)
        self.logdet_cc = torch.stack(logdet_cc, dim=0)
        self.Lzgc = torch.stack(lzgc, dim=0)

    def _x_given_c_fixed_k(self, k, c, truncated=False, trunc=(-1.5, 1.5)):
        delta = c - self.mu_c[k].unsqueeze(0)
        mu_zc = delta @ (self.invcc[k].T @ self.Szc[k].T)
        if truncated:
            lo, hi = trunc
            z_eps = torch.from_numpy(
                truncnorm.rvs(lo, hi, size=(c.shape[0], self.d)).astype(np.float32)
            ).to(self.device)
        else:
            z_eps = torch.randn(c.shape[0], self.d, device=self.device)
        z = mu_zc + z_eps @ self.Lzgc[k].T
        x = self.mu_x[k].unsqueeze(0) + z @ self.B[k].T
        if self.orth_sigma > 0.0:
            x = x + torch.randn_like(x) * self.orth_sigma
        return x

    @torch.no_grad()
    def sample_cond(self, c_in, deterministic_component=False, truncated=False, trunc=(-1.5, 1.5), pis=None):
        c = torch.as_tensor(c_in, dtype=torch.float32, device=self.device)
        bsz = c.shape[0]

        if pis is not None:
            pis = torch.as_tensor(pis, device=self.device, dtype=torch.long)
            x_out = torch.empty(bsz, self.Dx, device=self.device)
            for k in pis.unique(sorted=True).tolist():
                mask = pis == k
                if mask.any():
                    x_out[mask] = self._x_given_c_fixed_k(k, c[mask], truncated=truncated, trunc=trunc)
            return x_out, pis, None

        logp = []
        for k in range(self.K):
            delta = c - self.mu_c[k].unsqueeze(0)
            y = torch.cholesky_solve(delta.T, self.Lcc[k])
            quad = (delta.T * y).sum(dim=0)
            lp = -0.5 * (quad + self.logdet_cc[k] + self.Dc * np.log(2 * np.pi))
            logp.append(lp.unsqueeze(1))
        logp = torch.cat(logp, dim=1)
        logw = logp + torch.log(self.weights).unsqueeze(0)
        logw = logw - logw.logsumexp(dim=1, keepdim=True)
        w = torch.exp(logw)

        pis = torch.argmax(w, dim=1) if deterministic_component else torch.multinomial(w, 1).squeeze(1)
        x_out = torch.empty(bsz, self.Dx, device=self.device)
        for k in pis.unique(sorted=True).tolist():
            mask = pis == k
            if mask.any():
                x_out[mask] = self._x_given_c_fixed_k(k, c[mask], truncated=truncated, trunc=trunc)
        return x_out, pis, w

    @torch.no_grad()
    def sample_joint(self, M, truncated=False, trunc=(-1.5, 1.5), pis=None):
        if pis is None:
            pis = torch.multinomial(self.weights, M, replacement=True).to(self.device)
        else:
            pis = torch.as_tensor(pis, device=self.device, dtype=torch.long)

        c = torch.empty(M, self.Dc, device=self.device)
        for k in pis.unique(sorted=True).tolist():
            mask = pis == k
            if mask.any():
                eps = torch.randn(int(mask.sum().item()), self.Dc, device=self.device)
                c[mask] = self.mu_c[k].unsqueeze(0) + eps @ self.Lcc[k].T

        x = torch.empty(M, self.Dx, device=self.device)
        for k in pis.unique(sorted=True).tolist():
            mask = pis == k
            if mask.any():
                x[mask] = self._x_given_c_fixed_k(k, c[mask], truncated=truncated, trunc=trunc)
        return x, c, pis


def _standardize_cols(A, eps=1e-8):
    """Z-score standardize each column of `A`.

    Args:
        A: (N, D) array.
        eps: Small floor for std to avoid division by zero.

    Returns:
        A_std: Standardized array with zero mean / unit variance per column.
        stats: Tuple (mean[1,D], std[1,D]) used for the transform.
    """
    mu = A.mean(axis=0, keepdims=True) # (D, )
    sd = A.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (A - mu) / sd, (mu, sd)
def cluster_points_joint(X, C, m, jaccard_thresh=0.5, merge_k=10,
                         standardize=True, scale_x=1.0, scale_c=1.0):
    """Greedy set-cover style clustering on joint features [X | C].

    Steps:
      1) Build local m-NN neighborhoods in a standardized / scaled feature space.
      2) Choose uncovered seeds and take their neighborhoods as candidate clusters.
      3) Merge seed clusters with Jaccard overlap ≥ `jaccard_thresh` using seed-to-seed KNN.
      4) Ensure full coverage and build an inverse map from point index to cluster ids.

    Args:
        X: (N, Dx) flattened trajectories.
        C: (N, Dc) environment parameters.
        m: Neighborhood size for initial local clusters (m ≥ 2).
        jaccard_thresh: Merge threshold on set overlap.
        merge_k: Number of nearest seed clusters to consider when merging.
        standardize: If True, z-score features before clustering.
        scale_x: Scale factor applied to standardized X block.
        scale_c: Scale factor applied to standardized C block.

    Returns:
        clusters: List[Set[int]] of merged index sets.
        inv_cluster: Dict[int, List[int]] mapping point → list of cluster ids.
    """
    N, Dx = X.shape
    Dc = C.shape[1]
    assert C.shape[0] == N and m >= 2

    # 0) Feature build
    if standardize:
        Xs, _ = _standardize_cols(X)
        Cs, _ = _standardize_cols(C)
    else:
        Xs, Cs = X, C
    F = np.hstack([scale_x * Xs, scale_c * Cs])

    # 1) local m-NN neighborhoods
    nn = NearestNeighbors(n_neighbors=min(m, N), algorithm='kd_tree')
    nn.fit(F)
    _, indices = nn.kneighbors(F)
    raw = []
    for i, neigh in enumerate(indices):
        s = set(neigh.tolist())
        s.add(i)              # ensure self-inclusion
        raw.append(s)

    # 2) greedy cover -> candidates
    covered = np.zeros(N, dtype=bool)
    seed_indices, candidates = [], []
    for i in range(N):
        if not covered[i]:
            seed_indices.append(i)
            cand = raw[i].copy()
            candidates.append(cand)
            covered[list(cand)] = True

    M = len(candidates)
    if M <= 1:
        # One full cluster; make inv total
        clusters = [set(range(N))]
        inv = {j: [0] for j in range(N)}
        return clusters, inv

    # 3) merge via seed-to-seed KNN + Jaccard
    seed_pts = F[seed_indices]
    seed_nbrs = NearestNeighbors(n_neighbors=min(merge_k+1, M), algorithm='kd_tree').fit(seed_pts)
    _, seed_neighbors = seed_nbrs.kneighbors(seed_pts)

    parent = list(range(M))
    cluster_sets = {i: candidates[i] for i in range(M)}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        # merge smaller into larger
        if len(cluster_sets[ra]) < len(cluster_sets[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        cluster_sets[ra] |= cluster_sets.pop(rb)
        return ra

    for i in range(M):
        for j in seed_neighbors[i][1:]:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            Ci, Cj = cluster_sets[ri], cluster_sets[rj]
            inter = len(Ci & Cj)
            union_sz = len(Ci | Cj)
            if union_sz > 0 and (inter / union_sz) >= jaccard_thresh:
                union(ri, rj)

    merged_clusters = list(cluster_sets.values())

    # 4) assert & repair coverage
    covered_all = set().union(*merged_clusters) if merged_clusters else set()
    if len(covered_all) < N:
        missing = [j for j in range(N) if j not in covered_all]
        print(f"Warning! {len(missing)} points are not covered by clusters, adding them to the first cluster")
        for j in missing:
            merged_clusters[0].add(j)

    # 5) build inverse map
    inv_cluster = {j: [] for j in range(N)}
    for ci, cluster in enumerate(merged_clusters):
        for j in cluster:
            inv_cluster[j].append(ci)

    return merged_clusters, inv_cluster


def _process_one_cluster_joint(X, C, idx, d_x, eps, chi2_thresh, max_pca_samples):
    """Compute per-cluster low-rank stats on X and full stats on C.

    Workflow:
      - Joint outlier filter using empirical covariance of [X|C].
      - PCA basis `B_x` on X only (rank = `d_x`), robust to degeneracy.
      - Covariances: Σ_zz from projected coordinates, Σ_cc full, Σ_zc cross.

    Args:
        X: (N, Dx) flattened trajectories for all points.
        C: (N, Dc) environment parameters for all points.
        idx: Iterable of indices belonging to this cluster.
        d_x: Target rank for the low-rank basis on X.
        eps: Numerical regularization for covariances.
        chi2_thresh: Chi-square quantile for joint outlier removal in [X|C].
        max_pca_samples: Subsample size cap for PCA fit for speed.

    Returns:
        mu_x: (Dx,) mean of X in cluster (after outlier filter).
        mu_c: (Dc,) mean of C in cluster.
        Bx: (Dx, d_x) low-rank basis for X.
        Sig_zz: (d_x, d_x) covariance in latent space.
        Sig_zc: (d_x, Dc) cross-covariance between latent z and C.
        Sig_cc: (Dc, Dc) covariance of C.
        weight: Relative cluster weight (#inliers / N).
    """
    Xi = X[idx]         # (S, Dx)
    Ci = C[idx]         # (S, Dc)
    S, Dx = Xi.shape
    Dc = Ci.shape[1]

    # --- 1) joint outlier filter using empirical cov of [X|C] ---
    J = np.hstack([Xi, Ci])           # (S, Dx+Dc)
    Dj = J.shape[1]
    muJ = J.mean(axis=0)
    covJ = np.cov(J, rowvar=False) + eps * np.eye(Dj)
    L = np.linalg.cholesky(covJ)
    Y = np.linalg.solve(L, (J - muJ).T)    # (Dj, S)
    dists = (Y * Y).sum(axis=0)
    mask = dists <= chi2_thresh
    Xi_c = Xi[mask]
    Ci_c = Ci[mask]
    if Xi_c.shape[0] < max(5, d_x + 1):    # fallback if too few inliers
        Xi_c, Ci_c = Xi, Ci

    # Subsample for PCA if huge
    S2 = Xi_c.shape[0]
    if S2 > max_pca_samples:
        sel = np.random.choice(S2, max_pca_samples, replace=False)
        Xp = Xi_c[sel]
    else:
        Xp = Xi_c

    # --- 2) PCA on X only (robust) ---
    # center and check variance
    Xp0 = Xp - Xp.mean(axis=0, keepdims=True)
    col_var = Xp0.var(axis=0)                  # (Dx,)
    total_var = float(col_var.sum())

    if total_var <= 1e-12 or Xp.shape[0] < 2:
        # Degenerate cluster: no usable variance. Use a fixed fallback basis.
        print("Degenerate cluster!")
        n_comp = min(d_x, Dx)
        Bx = np.zeros((Dx, n_comp), dtype=np.float32)
        for j in range(n_comp):
            Bx[j, j] = 1.0                    # first n_comp standard basis vectors
    else:
        # Keep only nonzero-variance columns for the fit
        keep = col_var > 1e-12
        Xp_red = Xp0[:, keep]
        Dx_red = int(keep.sum())

        if Dx_red == 0:
            # All columns were zero-variance after centering
            n_comp = min(d_x, Dx)
            Bx = np.zeros((Dx, n_comp), dtype=np.float32)
            for j in range(n_comp):
                Bx[j, j] = 1.0
        else:
            # Limit components to numeric rank to avoid over-asking
            n_comp = int(min(d_x, Dx_red, max(1, Xp_red.shape[0] - 1)))

            # For small / ill-conditioned batches, SVD is very stable:
            # U, S, Vt = np.linalg.svd(Xp_red, full_matrices=False)
            # B_red = Vt[:n_comp].T

            ipca = IncrementalPCA(n_components=n_comp, batch_size=min(1024, Xp_red.shape[0]), whiten=False)
            ipca.fit(Xp_red)                   # denominator > 0 now -> no warning
            B_red = ipca.components_.T         # (Dx_red, n_comp)

            # Lift back to full Dx by inserting zeros at dropped columns
            Bx = np.zeros((Dx, n_comp), dtype=np.float32)
            Bx[keep, :] = B_red

    # pad with orthonormal columns in the complement subspace:
    if Bx.shape[1] < d_x:
        print(f"Missing {d_x - Bx.shape[1]} axes")
        k = d_x - Bx.shape[1]
        pad = np.zeros((Dx, k), dtype=np.float32)
        for j in range(k):
            # simple, deterministic padding with standard basis not already used
            col = (Bx.shape[1] + j) % Dx
            pad[col, j] = 1.0
        Bx = np.concatenate([Bx, pad], axis=1)

    # --- 3) reduced z and C stats (means, covs, cross-covs) ---
    mu_x = Xi_c.mean(axis=0)           # (Dx,)
    mu_c = Ci_c.mean(axis=0)           # (Dc,)
    Zc   = (Xi_c - mu_x) @ Bx          # (S_in, d_x)
    Cc   = (Ci_c - mu_c)               # (S_in, Dc)

    denom = max(1, Zc.shape[0] - 1)
    # Σ_zz from the projected cloud; more robust than just diag(var)
    Sig_zz = (Zc.T @ Zc) / denom                       # (d_x, d_x)
    # ensure SPD
    Sig_zz = Sig_zz + eps * np.eye(Sig_zz.shape[0])

    Sig_cc = (Cc.T @ Cc) / denom + eps * np.eye(Dc)    # (Dc, Dc)
    Sig_zc = (Zc.T @ Cc) / denom                       # (d_x, Dc)

    # cluster weight
    weight = Xi_c.shape[0] / X.shape[0]

    return mu_x, mu_c, Bx, Sig_zz, Sig_zc, Sig_cc, weight
def compute_cluster_pca_fast_joint(X, C, clusters, d_x,
                                   eps=1e-3, outlier_q=0.9,
                                   max_pca_samples=2000, n_jobs=-1):
    """Parallel per-cluster statistics for DGFM.

    Args:
        X: (N, Dx) flattened trajectories.
        C: (N, Dc) environment parameters.
        clusters: Iterable of sets/iterables with point indices per cluster.
        d_x: Target rank for the X basis per cluster.
        eps: Numerical regularization for covariances.
        outlier_q: Quantile (0..1) for joint [X|C] chi-square outlier cutoff.
        max_pca_samples: Cap on samples used to fit PCA for speed.
        n_jobs: Joblib parallel workers (-1 uses all cores).

    Returns:
        mu_x: (K, Dx)
        mu_c: (K, Dc)
        B: (K, Dx, d_x)
        Sig_zz: (K, d_x, d_x)
        Sig_zc: (K, d_x, Dc)
        Sig_cc: (K, Dc, Dc)
        weights: (K,) mixture weights
    """
    N, Dx = X.shape
    Dc = C.shape[1]
    Dj = Dx + Dc
    chi2_thresh = chi2.ppf(outlier_q, df=Dj)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_cluster_joint)(
            X, C, list(c), d_x, eps, chi2_thresh, max_pca_samples
        ) for c in clusters
    )

    mu_x, mu_c, B_list, Sig_zz, Sig_zc, Sig_cc, weights = zip(*results)
    # Stack
    mu_x  = np.vstack(mu_x)                       # (K, Dx)
    mu_c  = np.vstack(mu_c)                       # (K, Dc)
    B     = np.stack(B_list, axis=0)              # (K, Dx, d_x)
    Sig_zz = np.stack(Sig_zz, axis=0)             # (K, d_x, d_x)
    Sig_zc = np.stack(Sig_zc, axis=0)             # (K, d_x, Dc)
    Sig_cc = np.stack(Sig_cc, axis=0)             # (K, Dc, Dc)
    weights = np.array(weights, dtype=np.float32) # (K,)
    return mu_x, mu_c, B, Sig_zz, Sig_zc, Sig_cc, weights


class DGFM(VanillaFM):
    """Dimension-guided FM trainer."""

    path_weight_atol = 1e-5

    def _path_weights(self, t, interpolation_path):
        if interpolation_path != "piecewise-linear-midpoint":
            raise ValueError(
                f"Unsupported DGFM interpolation path '{interpolation_path}'. "
                "Expected 'piecewise-linear-midpoint'."
            )

        midpoint = torch.as_tensor(0.5, dtype=t.dtype, device=t.device)
        left = t < midpoint
        right = t > midpoint
        mid = ~(left | right)

        a = torch.where(left, 1.0 - 2.0 * t, torch.zeros_like(t))
        b_left = 2.0 * t
        b_right = 2.0 - 2.0 * t
        b = torch.where(left | mid, b_left, b_right)
        c = torch.where(right, 2.0 * t - 1.0, torch.zeros_like(t))

        a_dot = torch.where(left, -2.0 * torch.ones_like(t), torch.zeros_like(t))
        b_dot = torch.where(left, 2.0 * torch.ones_like(t), -2.0 * torch.ones_like(t))
        c_dot = torch.where(right, 2.0 * torch.ones_like(t), torch.zeros_like(t))

        a_dot = torch.where(mid, -1.0 * torch.ones_like(t), a_dot)
        b_dot = torch.where(mid, torch.zeros_like(t), b_dot)
        c_dot = torch.where(mid, torch.ones_like(t), c_dot)
        self._check_path_partition(a, b, c)
        return a, b, c, a_dot, b_dot, c_dot

    def _check_path_partition(self, a, b, c):
        err = torch.max(torch.abs(a + b + c - 1.0)).item()
        if err > self.path_weight_atol:
            raise ValueError(
                f"DGFM path weights must sum to 1; max |a+b+c-1|={err:.3e}"
            )

    def _sample_covering_clusters(self, idx, inv_cluster, cluster_sizes):
        """Choose one covering cluster per sample, weighted by raw cluster size."""
        choices = []
        idx_cpu = idx.detach().cpu().tolist()
        for i in idx_cpu:
            covering = inv_cluster[i]
            if not covering:
                raise ValueError(f"Training sample {i} is not covered by any DGFM cluster")
            weights = np.asarray([cluster_sizes[k] for k in covering], dtype=np.float64)
            weights = weights / weights.sum()
            choices.append(np.random.choice(covering, p=weights))
        return torch.tensor(choices, dtype=torch.long, device=self.device)

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
    ):
        """Build DGFM interpolants along x_t = a(t)z + b(t)y + c(t)x."""
        idx = perm_t[:target_trajectories.shape[0]]
        x = target_trajectories[idx]
        c = conditions[idx, :]
        m = x.shape[0]

        z = torch.randn(m, self.horizon, self.dof, device=self.device)
        pis = self._sample_covering_clusters(idx, inv_cluster, cluster_sizes)
        y_flat, _, _ = mixture_sampler.sample_cond(c, truncated=True, pis=pis)
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
        cluster_d: int,
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
    ):
        if mf is not None or n_t_local is not None or n_t_global is not None:
            print("[DGFM] Ignoring deprecated mf/n_t_local/n_t_global; using n_t only.")

        print("Clustering dataset . . .")
        X_np = target_trajectories.detach().cpu().numpy().reshape(target_trajectories.shape[0], -1)
        C_np = conditions.detach().cpu().numpy()

        clusters, inv_cluster = cluster_points_joint(
            X_np, C_np, m=cluster_size, jaccard_thresh=0.8, merge_k=10,
            standardize=True, scale_x=scale_x, scale_c=scale_c
        )
        cluster_sizes = np.asarray([len(c) for c in clusters], dtype=np.float64)

        print(f"{len(clusters)} clusters made! Applying PCA . . .")
        mu_x, mu_c, B, Szz, Szc, Scc, weights = compute_cluster_pca_fast_joint(
            X_np, C_np, clusters, d_x=cluster_d, eps=1e-3, outlier_q=0.9,
            max_pca_samples=2000, n_jobs=-1
        )

        mixture_sampler = MixtureSampler(
            mu_x, mu_c, B, Szz, Szc, Scc, weights, device=self.device, reg=1e-6, orth_sigma=0.0
        )

        N = target_trajectories.shape[0]
        joint_N = N * n_t
        best_avg_reward = 0.0
        best_success_rate = 0.0
        best_model = copy.deepcopy(self.model)
        success_rate_recs = {}
        stop_count = 0

        do_validation = val_period > 0 and val_trials > 0
        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials)

        try:
            self.model = self.model.to(self.device)
            target_trajectories = target_trajectories.to(self.device)
            conditions = conditions.to(self.device)

            for epoch in tqdm(range(1, max_epochs + 1), desc="DGFM Training", unit="epoch"):
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
                    success_rate, avg_reward = eval_model(
                        self.model, VectorField, self.task_name, self.horizon, self.dof,
                        self.condition_dim, self.gripper_idx, val_params,
                        env_settings_all, self.device, trials=val_trials
                    )

                    if not torch.is_grad_enabled():
                        torch.set_grad_enabled(True)

                    success_rate_recs[epoch] = {
                        "success_rate": success_rate,
                        "avg_reward": avg_reward,
                        "loss": avg_loss,
                    }

                    if success_rate < best_success_rate:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                                   f"avg reward={avg_reward:.3f}, loss={avg_loss:.3f}")
                        if early_stopping:
                            if stop_count == stop_criteria:
                                tqdm.write("Early stopping triggered.")
                                break
                            stop_count += 1
                    else:
                        if (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                            best_avg_reward = avg_reward
                            best_success_rate = success_rate
                            best_model = copy.deepcopy(self.model)
                            stop_count = 0
                            tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                                       f"avg reward={avg_reward:.3f}, loss={avg_loss:.3f} | Best model saved")
                        else:
                            tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                                       f"avg reward={avg_reward:.3f}, loss={avg_loss:.3f}")

        except KeyboardInterrupt:
            tqdm.write("Training interrupted by user. Returning best model so far...")

        if not success_rate_recs:
            best_model = copy.deepcopy(self.model)
        return best_model, self.model, success_rate_recs, mixture_sampler

