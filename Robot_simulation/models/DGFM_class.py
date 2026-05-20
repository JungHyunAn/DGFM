"""Dimension-guided FM utilities."""

import copy
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import chi2, truncnorm
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from Robot_simulation.models.VanillaFM_class import VanillaFM, VectorField
from Robot_simulation.models.FM_util import _generate_val_env, eval_model


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


def _pad_basis_to_rank(Bx, target_rank, Dx):
    """Pad a basis with deterministic orthonormal-ish fallback axes."""
    if Bx.shape[1] >= target_rank:
        return Bx[:, :target_rank].astype(np.float32, copy=False)

    cols = [Bx[:, j].astype(np.float32, copy=False) for j in range(Bx.shape[1])]
    for j in range(Dx):
        if len(cols) >= target_rank:
            break
        v = np.zeros(Dx, dtype=np.float32)
        v[j] = 1.0
        for q in cols:
            v = v - np.dot(q, v) * q
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            cols.append(v / norm)

    while len(cols) < target_rank:
        v = np.zeros(Dx, dtype=np.float32)
        v[len(cols) % Dx] = 1.0
        cols.append(v)

    return np.stack(cols, axis=1).astype(np.float32, copy=False)


def _pad_square(mat, target_rank, eps):
    out = np.zeros((target_rank, target_rank), dtype=np.float32)
    r = mat.shape[0]
    out[:r, :r] = mat
    if r < target_rank:
        out[r:, r:] = eps * np.eye(target_rank - r, dtype=np.float32)
    return out


def _pad_rows(mat, target_rank):
    out = np.zeros((target_rank, mat.shape[1]), dtype=np.float32)
    out[:mat.shape[0], :] = mat
    return out


def _process_one_cluster_joint(X, C, idx, min_d_x, eps, chi2_thresh, max_pca_samples):
    """Compute per-cluster full PCA stats on X and full stats on C.

    The fitted PCA basis keeps every available principal axis for the cluster.
    `min_d_x` is only used as a minimum fallback rank when the cluster is
    degenerate or numerically low-rank; final rank selection happens globally in
    `compute_cluster_pca_fast_joint`.
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
    if Xi_c.shape[0] < max(5, min_d_x + 1):    # fallback if too few inliers
        Xi_c, Ci_c = Xi, Ci

    # Subsample for PCA if huge
    S2 = Xi_c.shape[0]
    if S2 > max_pca_samples:
        sel = np.random.choice(S2, max_pca_samples, replace=False)
        Xp = Xi_c[sel]
    else:
        Xp = Xi_c

    # --- 2) PCA on X only (robust), keeping all available components ---
    Xp0 = Xp - Xp.mean(axis=0, keepdims=True)
    col_var = Xp0.var(axis=0)                  # (Dx,)
    total_var = float(col_var.sum())
    min_rank = min(min_d_x, Dx)

    if total_var <= 1e-12 or Xp.shape[0] < 2:
        print("Degenerate cluster!")
        Bx = np.zeros((Dx, min_rank), dtype=np.float32)
        for j in range(min_rank):
            Bx[j, j] = 1.0
        eigvals = np.zeros(min_rank, dtype=np.float32)
    else:
        keep = col_var > 1e-12
        Xp_red = Xp0[:, keep]
        Dx_red = int(keep.sum())

        if Dx_red == 0:
            Bx = np.zeros((Dx, min_rank), dtype=np.float32)
            for j in range(min_rank):
                Bx[j, j] = 1.0
            eigvals = np.zeros(min_rank, dtype=np.float32)
        else:
            n_comp = int(min(Dx_red, max(1, Xp_red.shape[0] - 1)))
            ipca = IncrementalPCA(n_components=n_comp, batch_size=min(1024, Xp_red.shape[0]), whiten=False)
            ipca.fit(Xp_red)
            B_red = ipca.components_.T         # (Dx_red, n_comp)

            Bx = np.zeros((Dx, n_comp), dtype=np.float32)
            Bx[keep, :] = B_red
            eigvals = ipca.explained_variance_.astype(np.float32, copy=False)

            if Bx.shape[1] < min_rank:
                print(f"Missing {min_rank - Bx.shape[1]} minimum axes")
                Bx = _pad_basis_to_rank(Bx, min_rank, Dx)
                eigvals = np.pad(eigvals, (0, min_rank - eigvals.shape[0]))

    # --- 3) full reduced z and C stats (means, covs, cross-covs) ---
    mu_x = Xi_c.mean(axis=0)           # (Dx,)
    mu_c = Ci_c.mean(axis=0)           # (Dc,)
    Zc   = (Xi_c - mu_x) @ Bx          # (S_in, full_d_x)
    Cc   = (Ci_c - mu_c)               # (S_in, Dc)

    denom = max(1, Zc.shape[0] - 1)
    Sig_zz = (Zc.T @ Zc) / denom + eps * np.eye(Zc.shape[1])
    Sig_cc = (Cc.T @ Cc) / denom + eps * np.eye(Dc)
    Sig_zc = (Zc.T @ Cc) / denom

    weight = Xi_c.shape[0] / X.shape[0]

    return mu_x, mu_c, Bx, eigvals, Sig_zz, Sig_zc, Sig_cc, weight


def compute_cluster_pca_fast_joint(X, C, clusters, d_x,
                                   eps=1e-3, outlier_q=0.9,
                                   max_pca_samples=2000, n_jobs=-1):
    """Parallel per-cluster statistics for DGFM.

    Each cluster first fits its full available PCA basis. A global eta threshold
    is then set to the largest cumulative explained-variance ratio of the top
    `d_x` components across clusters. Every cluster keeps enough leading axes to
    reach that eta, with `d_x` as the minimum kept rank.
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

    mu_x, mu_c, B_full, eigvals, Sig_zz_full, Sig_zc_full, Sig_cc, weights = zip(*results)

    eta_candidates = []
    degenerate_clusters = 0
    for vals in eigvals:
        vals = np.asarray(vals, dtype=np.float64)
        total = float(vals.sum())
        if total <= 1e-12:
            degenerate_clusters += 1
            continue
        top = vals[:min(d_x, vals.shape[0])].sum()
        eta_candidates.append(float(top / total))
    eta = min(1.0, max(eta_candidates)) if eta_candidates else 1.0

    ranks = []
    retained_etas = []
    for vals in eigvals:
        vals = np.asarray(vals, dtype=np.float64)
        total = float(vals.sum())
        if total <= 1e-12:
            rank = min(d_x, vals.shape[0])
            retained_eta = 1.0
        else:
            cumulative = np.cumsum(vals) / total
            rank = int(np.searchsorted(cumulative, eta, side="left") + 1)
            rank = min(rank, vals.shape[0])
            retained_eta = float(cumulative[rank - 1])
        ranks.append(max(d_x, rank))
        retained_etas.append(retained_eta)

    packed_d = max(ranks) if ranks else d_x
    if ranks:
        ranks_arr = np.asarray(ranks)
        retained_arr = np.asarray(retained_etas, dtype=np.float64)
        eta_min = min(eta_candidates) if eta_candidates else 1.0
        eta_max = max(eta_candidates) if eta_candidates else 1.0
        eta_mean = float(np.mean(eta_candidates)) if eta_candidates else 1.0
        rank_counts = {
            int(rank): int(count)
            for rank, count in zip(*np.unique(ranks_arr, return_counts=True))
        }
        rank_p25, rank_median, rank_p75 = np.percentile(ranks_arr, [25, 50, 75])
        print(
            f"[DGFM] Global PCA eta={eta:.4f} "
            f"(top-{d_x} eta min/mean/max={eta_min:.4f}/{eta_mean:.4f}/{eta_max:.4f}); "
            f"rank min/p25/median/p75/max={ranks_arr.min()}/{rank_p25:.1f}/"
            f"{rank_median:.1f}/{rank_p75:.1f}/{ranks_arr.max()}, "
            f"mean={ranks_arr.mean():.2f}, packed rank={packed_d}, "
            f"retained eta min/mean={retained_arr.min():.4f}/{retained_arr.mean():.4f}, /"
            f"degenerate clusters={degenerate_clusters}."
        )
    else:
        print(f"[DGFM] Global PCA eta={eta:.4f}; no clusters found (packed rank={packed_d}).")

    B_list = []
    Sig_zz = []
    Sig_zc = []
    for Bx, szz, szc, rank in zip(B_full, Sig_zz_full, Sig_zc_full, ranks):
        B_keep = _pad_basis_to_rank(Bx[:, :min(rank, Bx.shape[1])], rank, Dx)
        szz_keep = szz[:min(rank, szz.shape[0]), :min(rank, szz.shape[0])]
        szc_keep = szc[:min(rank, szc.shape[0]), :]

        B_list.append(_pad_basis_to_rank(B_keep, packed_d, Dx))
        Sig_zz.append(_pad_square(szz_keep, packed_d, eps))
        Sig_zc.append(_pad_rows(szc_keep, packed_d))

    mu_x  = np.vstack(mu_x)                       # (K, Dx)
    mu_c  = np.vstack(mu_c)                       # (K, Dc)
    B     = np.stack(B_list, axis=0)              # (K, Dx, packed_d)
    Sig_zz = np.stack(Sig_zz, axis=0)             # (K, packed_d, packed_d)
    Sig_zc = np.stack(Sig_zc, axis=0)             # (K, packed_d, Dc)
    Sig_cc = np.stack(Sig_cc, axis=0)             # (K, Dc, Dc)
    weights = np.array(weights, dtype=np.float32) # (K,)
    return mu_x, mu_c, B, Sig_zz, Sig_zc, Sig_cc, weights


class DGFM(VanillaFM):
    """Dimension-guided FM trainer."""

    path_weight_atol = 1e-5

    def _path_weights(self, t, interpolation_path):
        if interpolation_path == "piecewise-linear-midpoint":
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
        
        if interpolation_path == "cosine-midpoint":
            # Fixed injection point tau = 0.5:
            # x0 -> y for t in [0, 0.5], then y -> x1 for t in [0.5, 1].
            tau = torch.as_tensor(0.5, dtype=t.dtype, device=t.device)
            left = t <= tau
            right = ~left

            # Smoothstep: s(r) = (1 - cos(pi r)) / 2
            # s'(r) = (pi / 2) sin(pi r)
            pi = torch.as_tensor(torch.pi, dtype=t.dtype, device=t.device)

            r_left = torch.clamp(t / tau, 0.0, 1.0)
            r_right = torch.clamp((t - tau) / (1.0 - tau), 0.0, 1.0)

            s_left = 0.5 * (1.0 - torch.cos(pi * r_left))
            s_right = 0.5 * (1.0 - torch.cos(pi * r_right))

            sdot_left = 0.5 * pi * torch.sin(pi * r_left) / tau
            sdot_right = 0.5 * pi * torch.sin(pi * r_right) / (1.0 - tau)

            # Left segment: x_t = (1 - s)x0 + s y
            a_left = 1.0 - s_left
            b_left = s_left
            c_left = torch.zeros_like(t)

            a_dot_left = -sdot_left
            b_dot_left = sdot_left
            c_dot_left = torch.zeros_like(t)

            # Right segment: x_t = (1 - s)y + s x1
            a_right = torch.zeros_like(t)
            b_right = 1.0 - s_right
            c_right = s_right

            a_dot_right = torch.zeros_like(t)
            b_dot_right = -sdot_right
            c_dot_right = sdot_right

            a = torch.where(left, a_left, a_right)
            b = torch.where(left, b_left, b_right)
            c = torch.where(left, c_left, c_right)

            a_dot = torch.where(left, a_dot_left, a_dot_right)
            b_dot = torch.where(left, b_dot_left, b_dot_right)
            c_dot = torch.where(left, c_dot_left, c_dot_right)

            self._check_path_partition(a, b, c)
            return a, b, c, a_dot, b_dot, c_dot
        
        if interpolation_path == "residual-cosine-midpoint":
            # This keeps the cosine schedule smooth, while preventing zero velocity
            # at the midpoint by adding a small direct x0 -> x1 component.

            tau = torch.as_tensor(0.5, dtype=t.dtype, device=t.device)
            lam = torch.as_tensor(0.2, dtype=t.dtype, device=t.device)  # try 0.2 first

            left = t <= tau
            right = ~left

            pi = torch.as_tensor(torch.pi, dtype=t.dtype, device=t.device)

            r_left = torch.clamp(t / tau, 0.0, 1.0)
            r_right = torch.clamp((t - tau) / (1.0 - tau), 0.0, 1.0)

            # Cosine smoothstep:
            # s(r) = (1 - cos(pi r)) / 2
            # s'(r) = (pi / 2) sin(pi r)
            s_left = 0.5 * (1.0 - torch.cos(pi * r_left))
            s_right = 0.5 * (1.0 - torch.cos(pi * r_right))

            sdot_left = 0.5 * pi * torch.sin(pi * r_left) / tau
            sdot_right = 0.5 * pi * torch.sin(pi * r_right) / (1.0 - tau)

            # ------------------------------------------------------------
            # 1. Cosine-DGFM base weights
            # ------------------------------------------------------------

            # Left segment: x_t^D = (1 - s)x0 + s y
            a_left_D = 1.0 - s_left
            b_left_D = s_left
            c_left_D = torch.zeros_like(t)

            a_dot_left_D = -sdot_left
            b_dot_left_D = sdot_left
            c_dot_left_D = torch.zeros_like(t)

            # Right segment: x_t^D = (1 - s)y + s x1
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

            # ------------------------------------------------------------
            # 2. Residual vanilla/OT path weights
            # ------------------------------------------------------------

            # x_t^FM = (1 - t)x0 + t x1
            a_FM = 1.0 - t
            b_FM = torch.zeros_like(t)
            c_FM = t

            a_dot_FM = -torch.ones_like(t)
            b_dot_FM = torch.zeros_like(t)
            c_dot_FM = torch.ones_like(t)

            # ------------------------------------------------------------
            # 3. Soft mixture
            # ------------------------------------------------------------

            a = (1.0 - lam) * a_D + lam * a_FM
            b = (1.0 - lam) * b_D + lam * b_FM
            c = (1.0 - lam) * c_D + lam * c_FM

            a_dot = (1.0 - lam) * a_dot_D + lam * a_dot_FM
            b_dot = (1.0 - lam) * b_dot_D + lam * b_dot_FM
            c_dot = (1.0 - lam) * c_dot_D + lam * c_dot_FM

            self._check_path_partition(a, b, c)
            return a, b, c, a_dot, b_dot, c_dot
        
        raise ValueError(
                f"Unsupported DGFM interpolation path '{interpolation_path}'. "
            )

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
        dgfm_truncated=True,
        dgfm_trunc_low=-1.5,
        dgfm_trunc_high=1.5,
    ):
        """Build DGFM interpolants along x_t = a(t)z + b(t)y + c(t)x."""
        idx = perm_t[:target_trajectories.shape[0]]
        x = target_trajectories[idx]
        c = conditions[idx, :]
        m = x.shape[0]

        z = torch.randn(m, self.horizon, self.dof, device=self.device)
        pis = self._sample_covering_clusters(idx, inv_cluster, cluster_sizes)
        y_flat, _, _ = mixture_sampler.sample_cond(
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
        recorded_control_freq: int | float | None = None,
        trajectory_control_freq: int | float | None = None,
    ):
        if mf is not None or n_t_local is not None or n_t_global is not None:
            print("[DGFM] Ignoring deprecated mf/n_t_local/n_t_global; using n_t only.")
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

        print("Clustering dataset . . .")
        X_np = target_trajectories.detach().cpu().numpy().reshape(target_trajectories.shape[0], -1)
        C_np = conditions.detach().cpu().numpy()

        clusters, inv_cluster = cluster_points_joint(
            X_np, C_np, m=cluster_size, jaccard_thresh=cluster_jaccard_thresh,
            merge_k=cluster_merge_k, standardize=cluster_standardize,
            scale_x=scale_x, scale_c=scale_c
        )
        cluster_sizes = np.asarray([len(c) for c in clusters], dtype=np.float64)

        print(f"{len(clusters)} clusters made! Applying PCA . . .")
        mu_x, mu_c, B, Szz, Szc, Scc, weights = compute_cluster_pca_fast_joint(
            X_np, C_np, clusters, d_x=cluster_d, eps=cluster_eps,
            outlier_q=cluster_outlier_q, max_pca_samples=max_pca_samples,
            n_jobs=pca_n_jobs
        )

        mixture_sampler = MixtureSampler(
            mu_x, mu_c, B, Szz, Szc, Scc, weights, device=self.device,
            reg=mixture_reg, orth_sigma=mixture_orth_sigma
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
                    success_rate, avg_reward = eval_model(
                        self.model, VectorField, self.task_name, self.horizon, self.dof,
                        self.condition_dim, self.gripper_idx, val_params,
                        env_settings_all, self.device, trials=val_trials,
                        recorded_control_freq=recorded_control_freq,
                        trajectory_control_freq=trajectory_control_freq
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

