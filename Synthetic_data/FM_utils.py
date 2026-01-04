import numpy as np
import torch
import torch.nn as nn
import ot
import time
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import truncnorm, chi2
from torch.distributions import Beta
from annoy import AnnoyIndex
from joblib import Parallel, delayed


class VectorField(nn.Module):
    # ----------------------------
    # Neural Vector Field for Flow Matching
    # ----------------------------
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, dim)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(1)], dim=1))


class MixtureSampler:
    # ----------------------------
    # Intermediate Mixture of Gaussians sampler
    # ----------------------------
    def __init__(self, mus, covs, weights, clusters, inv_cluster, device='cpu'):
        self.device = device
        self.mus = torch.tensor(mus, dtype=torch.float32, device=device) # (k, d)
        self.covs = torch.tensor(covs, dtype=torch.float32, device=device) # (k, d, d)
        self.weights = torch.tensor(weights/weights.sum(),dtype=torch.float32, device=device)  # normalize weights
        self.clusters = clusters
        self.inv_cluster = inv_cluster
        self.Ls = torch.linalg.cholesky(self.covs)
        self.k, self.d = self.mus.shape

    def sample(self, M, pis=None):
        if pis is None:
            pis = torch.multinomial(self.weights, M, replacement=True)             # (M,)

        eps = torch.randn(M, self.d, device=self.device)                      # (M, d)

        # gather the Cholesky for each sample
        L_sel   = self.Ls[pis]
        mu_sel  = self.mus[pis]
        
        # transform: x = mu + L @ eps
        x = torch.bmm(L_sel, eps.unsqueeze(-1)).squeeze(-1)  # (M, d)
        return x + mu_sel, pis
    
    def truncated_sample(self, M, pis=None):
        """
        Draw ~factor*M samples, keep the first M that lie within ±2σ
        (coordinate-wise), and if there aren’t enough, sample the rest.
        Args:
            M      : number of final samples wanted
            factor : oversampling factor (>1) to reduce loops
            pis    : (optional) precomputed component indices for sampling
        Returns:
            x      : (M, d) tensor of truncated samples
            pis    : (M,) tensor of component indices
        """

        if pis is None:
            pis = torch.multinomial(self.weights, M, replacement=True).to(self.device)

        # use truncnorm to sample from truncated normal distribution
        z = truncnorm.rvs(-1.5, 1.5, size=(M, self.d), random_state=None).astype(np.float32)
        eps = torch.from_numpy(z).to(self.device)

        L_sel  = self.Ls[pis]
        mu_sel = self.mus[pis]

        x = torch.bmm(L_sel, eps.unsqueeze(-1)).squeeze(-1) + mu_sel
        return x, pis
    

def cluster_points(X, m, jaccard_thresh=0.5, merge_k=10):
    # ----------------------------
    # Overlapping clusters formed with size & diameter constraints
    # Inputs - X: data points, m: min cluster size, delta: max cluster diameter
    # Outputs - clusters: list of sets of indices
    # ----------------------------
    N = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=m, algorithm='kd_tree').fit(X) # build KD-tree
    _, indices = nbrs.kneighbors(X, return_distance=True)
    raw = [set(neigh) for neigh in indices] # cluster for each points
 
    # filter raw clusters
    covered = np.zeros(N, dtype=bool) 
    seed_indices = []      # which seeds survive
    candidates = []        # their raw clusters
    for i in range(N):
        if not covered[i]:
            seed_indices.append(i)
            candidates.append(raw[i].copy())
            # mark all points in this cluster as covered
            covered[list(raw[i])] = True

    M = len(candidates)   # # of candidate clusters

    # merge similar clusters
    seed_pts = X[seed_indices]
    merge_k = min(merge_k, M-1)
    seed_nbrs = NearestNeighbors(n_neighbors=merge_k+1, algorithm='kd_tree').fit(seed_pts)
    _, seed_neighbors = seed_nbrs.kneighbors(seed_pts)

    parent = list(range(M))
    cluster_sets = {i: candidates[i] for i in range(M)}
    def find(u):
        while parent[u] != u: # probing to find parent
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if len(cluster_sets[ra]) < len(cluster_sets[rb]): # merge smaller into larger for efficiency
            ra, rb = rb, ra
        parent[rb] = ra
        cluster_sets[ra] |= cluster_sets.pop(rb) # merge cluster
        return ra
    
    for i in range(M):
        for j in seed_neighbors[i][1:]:      # skip j = i
            # only do one direction
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            Ci, Cj = cluster_sets[ri], cluster_sets[rj]
            inter = len(Ci & Cj)
            union_sz = len(Ci | Cj)
            if union_sz > 0 and (inter / union_sz) >= jaccard_thresh: # merge based on jaccard threshold
                union(ri, rj)

    merged_clusters = list(cluster_sets.values())
    # print(len(merged_clusters))

    # obtain inverse index
    inv_cluster = {j: [] for j in range(N)}
    for ci, cluster in enumerate(merged_clusters):
        for j in cluster:
            inv_cluster[j].append(ci)

    return merged_clusters, inv_cluster


def cluster_points_annoy(X, m, jaccard_thresh=0.5, merge_k=10,
                         n_trees=50, search_k=-1):
    """
    Approximate overlapping clustering via Annoy for fast neighbor queries.
    - X:       (N, D) data matrix
    - m:       desired cluster size (number of neighbors)
    - merge_k: how many seed‐to‐seed neighbors to consider
    - n_trees: number of trees to build in Annoy (more → higher accuracy, slower build)
    - search_k: Annoy search_k parameter (-1 uses default=m * n_trees)
    """
    N, D = X.shape

    # 1) Build Annoy index
    t = AnnoyIndex(D, metric='euclidean')
    for i, v in enumerate(X):
        t.add_item(i, v.tolist())
    t.build(n_trees)

    # 2) Get m nearest neighbors (approximate)
    raw_arr = []
    raw_sets = []
    for i in range(N):
        # search_k can be -1 (Annoy default) or e.g. m * 10 for higher accuracy
        neigh = t.get_nns_by_item(i, m, search_k=search_k, include_distances=False)
        arr = np.array(sorted(neigh), dtype=int)
        raw_arr.append(arr)
        raw_sets.append(set(arr.tolist()))

    # 3) Seed selection (exact same as before)
    covered = np.zeros(N, bool)
    seed_indices, candidates = [], []
    for i in range(N):
        if not covered[i]:
            seed_indices.append(i)
            candidates.append(raw_sets[i].copy())
            covered[raw_arr[i]] = True

    M = len(candidates)
    if M == 0:
        return [], {i: [] for i in range(N)}

    # 4) Seed‐to‐seed neighbor graph via Annoy on the seeds
    seed_pts = X[seed_indices]
    t2 = AnnoyIndex(D, metric='euclidean')
    for idx, v in enumerate(seed_pts):
        t2.add_item(idx, v.tolist())
    t2.build(n_trees)
    # get merge_k + 1 neighbors (including itself)
    seed_neighbors = [t2.get_nns_by_item(i, merge_k+1, search_k=search_k)
                      for i in range(M)]

    # 5) Union-Find merging based on Jaccard
    parent = list(range(M))
    cluster_sets = {i: candidates[i].copy() for i in range(M)}

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

    # 6) Collect merged clusters
    merged_clusters = list(cluster_sets.values())

    # 7) Build inverse index
    inv_cluster = {i: [] for i in range(N)}
    for ci, cluster in enumerate(merged_clusters):
        for j in cluster:
            inv_cluster[j].append(ci)

    return merged_clusters, inv_cluster


def compute_cluster_pca(X, clusters, d, eps=1e-3, outlier_thresh=0.9):
    # ----------------------------
    # Conduct PCA for clusters
    # Output:
    #   mus: means of clusters,
    #   covs: covariances using d dominant axes,
    #   weights: proportions of total points in each cluster (cluster weights)
    # ----------------------------
    mus = []
    covs = []
    weights = []
    N = X.shape[0]
    D = X.shape[1]

    for c in clusters:
        idx = list(c)
        Xi = X[idx]
        mu = Xi.mean(axis=0)

        # Estimate raw empirical covariance
        raw_cov = np.cov(Xi.T) + eps * np.eye(D)

        # Compute Mahalanobis distances
        centered = Xi - mu
        L = np.linalg.cholesky(raw_cov)
        y = np.linalg.solve(L, centered.T)
        dists = np.sum(y*y, axis=0)

        # Keep only inliers within chi2 threshold
        chi2_thresh = chi2.ppf(outlier_thresh, df=D)
        inlier_mask = dists <= chi2_thresh
        Xi_clean = Xi[inlier_mask]

        # Recompute mean and PCA on cleaned data
        mu_clean = Xi_clean.mean(axis=0)
        pca = PCA(n_components=d, svd_solver='randomized').fit(Xi_clean)
        Bi = pca.components_.T  # D x d
        Lambda = np.diag(np.clip(pca.explained_variance_, a_min=eps, a_max=None))
        Sigma = Bi @ Lambda @ Bi.T + eps * np.eye(D)
        Sigma = (Sigma + Sigma.T) / 2

        mus.append(mu_clean)
        covs.append(Sigma)
        weights.append(len(Xi_clean) / N)
    return np.array(mus), np.array(covs), np.array(weights)


def _process_one_cluster(X, idx, d, eps, chi2_thresh, max_pca_samples):
    """
    Compute (mu, Sigma, weight) for one cluster given full data X and index list idx.
    """
    Xi = X[idx]
    N_total, D = X.shape

    # 1) Raw mean + covariance via Cholesky for Mahalanobis
    mu = Xi.mean(axis=0)
    raw_cov = np.cov(Xi, rowvar=False) + eps * np.eye(D)
    L = np.linalg.cholesky(raw_cov)
    centered = (Xi - mu)
    y = np.linalg.solve(L, centered.T)
    dists = np.sum(y * y, axis=0)

    # 2) Inlier mask
    mask = dists <= chi2_thresh
    Xi_clean = Xi[mask]
    if Xi_clean.shape[0] == 0:
        # fallback: treat whole cluster as inliers
        Xi_clean = Xi.copy()

    # 3) Sub-sample if too large
    S = Xi_clean.shape[0]
    if S > max_pca_samples:
        sel = np.random.choice(S, max_pca_samples, replace=False)
        Xi_pca = Xi_clean[sel]
    else:
        Xi_pca = Xi_clean

    # 4) Incremental PCA (randomized)
    ipca = IncrementalPCA(n_components=d, batch_size=Xi_pca.shape[0], whiten=False)
    ipca.fit(Xi_pca)
    Bi = ipca.components_.T           # (D, d)
    variances = np.clip(ipca.explained_variance_, a_min=eps, a_max=None)
    Sigma = Bi @ np.diag(variances) @ Bi.T
    Sigma = (Sigma + Sigma.T) * 0.5

    # Reassure positive definite sigma
    eigvals = np.linalg.eigvalsh(Sigma)
    min_eig = eigvals.min()
    if min_eig < eps:
        Sigma += np.eye(D) * (eps - min_eig + 1e-8)

    mu_clean = Xi_clean.mean(axis=0)
    weight = Xi_clean.shape[0] / N_total

    return mu_clean, Sigma, weight


def compute_cluster_pca_fast(X, clusters, d, eps=1e-3, outlier_thresh=0.9,
                             max_pca_samples=2000, n_jobs=-1):
    """
    Fast per-cluster PCA with sub-sampling, IncrementalPCA, and parallelism.
    """
    N, D = X.shape
    chi2_thresh = chi2.ppf(outlier_thresh, df=D)

    # Launch one job per cluster
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_cluster)(
            X, list(c), d, eps, chi2_thresh, max_pca_samples
        ) for c in clusters
    )

    mus, covs, weights = zip(*results)
    return np.vstack(mus), np.stack(covs), np.array(weights)


def train_uniform_FM(model, optimizer, X_target, dim, device,
                     n_t=10, epochs=5, batch_size=256, early_stopping=True,tol=1e-3):
    # ----------------------------
    # Vanilla flow matching training
    # ----------------------------
    N = X_target.shape[0]
    perm = torch.randperm(N, device=device)
    split = int(0.9 * N)
    idx_train, idx_val = perm[:split], perm[split:]
    X_train = X_target[idx_train]
    X_val   = X_target[idx_val]
    stop_criteria = 0

    best_w2 = float("inf")
    best_model = None
    recs = []

    for epoch in range(epochs):
        # Training
        perm_t = torch.randperm(X_train.shape[0], device=device)
        for i in range(0, split, batch_size):
            idx = perm_t[i:min(i+batch_size, split)]
            x1 = X_train[idx]
            x0 = torch.randn(len(idx), dim, device=device) # standard Gaussian base distribution
            t  = torch.rand(len(idx)*n_t, device=device) # uniform t sampling

            x0r = x0.unsqueeze(1).repeat(1,n_t,1).view(-1,dim)
            x1r = x1.unsqueeze(1).repeat(1,n_t,1).view(-1,dim)            
            xt  = (1-t.unsqueeze(1))*x0r + t.unsqueeze(1)*x1r

            target_v = x1r - x0r
            pred_v   = model(xt,t)
            loss = ((pred_v-target_v)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        with torch.no_grad():
            X0   = np.random.randn(len(idx_val), dim)
            Xgen = run_flow(model, X0, device)
        w2 = np.sqrt(ot.emd2(np.ones(N-split)/(N-split), np.ones(N-split)/(N-split), ot.dist(Xgen.cpu().numpy(), X_val.cpu().numpy())**2))

        # Early stopping
        if early_stopping and ((best_w2 - w2) < tol):
            stop_criteria += 1
            if stop_criteria >= 3:  # stop after 3 epochs without improvement
                break
        else:
            stop_criteria = 0
            best_w2 = w2
            best_model = model

        # Record results
        recs.append(
            { "epoch": epoch, 
              "validation_w2": w2, 
              "train_loss": loss.item() }
        )
    return epoch, best_w2, recs, best_model


def train_shifted_FM(model, optimizer, X_target, dim, device, beta_a=0.5, beta_b=1,
                     n_t=10, epochs=5, batch_size=256, early_stopping=True,tol=1e-3):
    # ----------------------------
    # Vanilla flow matching training
    # ----------------------------
    N = X_target.shape[0]
    perm = torch.randperm(N, device=device)
    split = int(0.9 * N)
    idx_train, idx_val = perm[:split], perm[split:]
    X_train = X_target[idx_train]
    X_val   = X_target[idx_val]
    stop_criteria = 0

    best_w2 = float("inf")
    best_model = None
    recs = []

    for epoch in range(epochs):
        # Training
        perm_t = torch.randperm(X_train.shape[0], device=device)
        for i in range(0, split, batch_size):
            idx = perm_t[i:min(i+batch_size, split)]

            x1 = X_train[idx]
            x0 = torch.randn(len(idx), dim, device=device) # standard Gaussian base distribution
            t  = Beta(beta_a, beta_b).sample((len(idx)*n_t,)).to(device)  # Beta t sampling
            t  = torch.ones_like(t).to(device) - t

            x0r = x0.unsqueeze(1).repeat(1,n_t,1).view(-1,dim)
            x1r = x1.unsqueeze(1).repeat(1,n_t,1).view(-1,dim)            
            xt  = (1-t.unsqueeze(1))*x0r + t.unsqueeze(1)*x1r

            target_v = x1r - x0r
            pred_v   = model(xt,t)
            loss = ((pred_v-target_v)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        with torch.no_grad():
            X0   = np.random.randn(len(idx_val), dim)
            Xgen = run_flow(model, X0, device)
        w2 = np.sqrt(ot.emd2(np.ones(N-split)/(N-split), np.ones(N-split)/(N-split), ot.dist(Xgen.cpu().numpy(), X_val.cpu().numpy())**2))

        # Early stopping
        if early_stopping and ((best_w2 - w2) < tol):
            stop_criteria += 1
            if stop_criteria >= 3:  # stop after 3 epochs without improvement
                break
        else:
            stop_criteria = 0
            best_w2 = w2
            best_model = model

        # Record results
        recs.append(
            { "epoch": epoch, 
              "validation_w2": w2, 
              "train_loss": loss.item() }
        )
    return epoch, best_w2, recs, best_model


def train_dgfm(model, optimizer, X_target, dim, mf, device,
               mixture_sampler=None,
               n_t_global=1, n_t_local=1,
               epochs=5, batch_size=256,    
               cluster_size=50, cluster_d=3,
               early_stopping=True, tol=1e-3,
               approx_cluster=False, fast_PCA=True):
    
    # 0) Clustering & Form intermediate distribution
    base_sampler = lambda x: torch.randn(x, dim, device=device)
    N            = X_target.shape[0]
    perm         = np.random.permutation(N)
    M            = N - min(int(0.1*N), 2000) # training set size (validation set cannot exceed 2000 by numItermax of pot)
    
    idx_train, idx_val = perm[:M], perm[M:] # split into train and validation sets
    X_train            = X_target[idx_train]
    X_val              = X_target[idx_val]
    
    stop_criteria = 0

    if mixture_sampler is None:
        t0 = time.thread_time()
        if approx_cluster:
            clusters, inv_cluster = cluster_points_annoy(X_train.detach().cpu().numpy(), cluster_size)
        else:
            clusters, inv_cluster  = cluster_points(X_train.detach().cpu().numpy(), cluster_size)
        # print(f"Clustering took {time.thread_time() - t0:.2f} seconds")
        t0 = time.thread_time()
        if fast_PCA:
            mus, covs, weights = compute_cluster_pca_fast(X_train.detach().cpu().numpy(), clusters, d=cluster_d)
        else:
            mus, covs, weights = compute_cluster_pca(X_train.detach().cpu().numpy(), clusters, d=cluster_d)
        # print(f"PCA took {time.thread_time() - t0:.2f} seconds")
        mixture_sampler = MixtureSampler(mus, covs, weights, clusters, inv_cluster, device=device)
        

    best_w2    = float("inf")
    best_model = None
    recs       = []

    global_M = M * mf
    for epoch in range(epochs):
        perm_t = torch.randperm(X_train.shape[0], device=device)

        # 1) global FM
        t0 = time.thread_time()
        for i in range(0, global_M, batch_size):
            m       = min(batch_size, global_M-i) # batch size

            x0_t    = base_sampler(m)
            x1_t, _ = mixture_sampler.truncated_sample(m)
            t       = torch.rand(m*n_t_global, device=device)  # t sampling for global FM
            x0r     = x0_t.unsqueeze(1).repeat(1,n_t_global,1).view(-1,dim)
            x1r     = x1_t.unsqueeze(1).repeat(1,n_t_global,1).view(-1,dim)
            xt      = (1-t.unsqueeze(1))*x0r + t.unsqueeze(1)*x1r
            
            target_v = 2*(x1r - x0r)
            pred_v   = model(xt, 0.5*t)
            loss     = ((pred_v - target_v)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Global FM epoch {epoch} took {time.thread_time() - t0:.2f} seconds")

        # 2) local FM
        t0 = time.thread_time()
        for i in range(0, M, batch_size):
            m   = min(batch_size, M-i)
            idx = perm_t[i:i+m]            
            
            # Match within clusters
            inv_cluster = mixture_sampler.inv_cluster
            pis_np = np.array([ np.random.choice(inv_cluster[int(j.item())]) for j in idx ], dtype=np.int64)
            pis = torch.tensor(pis_np, device=device)

            x0, _ = mixture_sampler.truncated_sample(m, pis=pis)
            x1    = X_train[idx]            
            t     = torch.rand(m * n_t_local, device=device) # t sampling for local FM 
            x0r   = x0.unsqueeze(1).repeat(1, n_t_local, 1).view(-1, dim)
            x1r   = x1.unsqueeze(1).repeat(1, n_t_local, 1).view(-1, dim)
            xt    = (1 - t.unsqueeze(1)) * x0r + t.unsqueeze(1) * x1r

            target_v = 2*(x1r - x0r)
            pred_v = model(xt, 0.5*t+0.5)
            loss = ((pred_v - target_v) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Local FM epoch {epoch} took {time.thread_time() - t0:.2f} seconds")

        # Validation
        t0 = time.thread_time()
        with torch.no_grad():
            X0   = np.random.randn(len(idx_val), dim)
            Xgen = run_flow(model, X0, device)
        w2 = np.sqrt(ot.emd2(np.ones(N-M)/(N-M), np.ones(N-M)/(N-M), ot.dist(Xgen.cpu().numpy(), X_val.cpu().numpy())**2))
        # print(f"Validation epoch {epoch} took {time.thread_time() - t0:.2f} seconds")
        # Early stopping
        if early_stopping and ((best_w2 - w2) < tol):
            stop_criteria += 1
            if stop_criteria >= 3:  # stop after 3 epochs without improvement
                break
        else:
            stop_criteria = 0
            best_w2 = w2
            best_model = model

        # Record results
        recs.append(
            { "epoch": epoch, 
              "validation_w2": w2, 
              "train_loss": loss.item() }
        ) 
    return mixture_sampler, epoch, best_w2, recs, best_model


def train_gfm(model, optimizer, X_target, dim, device,
              mixture_sampler=None,
              n_t_global=1,
              epochs=5, batch_size=256,    
              cluster_size=50, cluster_d=3,
              early_stopping=True, tol=1e-3,
              approx_cluster=False, fast_PCA=True):
    
    # Clustering & Form intermediate distribution
    base_sampler = lambda x: torch.randn(x, dim, device=device)
    N            = X_target.shape[0]
    perm         = np.random.permutation(N)
    M            = N - min(int(0.1*N), 2000) # training set size (validation set cannot exceed 2000 by numItermax of pot)
    
    idx_train, idx_val = perm[:M], perm[M:] # split into train and validation sets
    X_train            = X_target[idx_train]
    X_val              = X_target[idx_val]

    stop_criteria = 0

    if mixture_sampler is None:
        t0 = time.thread_time()
        if approx_cluster:
            clusters, inv_cluster = cluster_points_annoy(X_train.detach().cpu().numpy(), cluster_size)
        else:
            clusters, inv_cluster  = cluster_points(X_train.detach().cpu().numpy(), cluster_size)
        # print(f"Clustering took {time.thread_time() - t0:.2f} seconds")
        t0 = time.thread_time()
        if fast_PCA:
            mus, covs, weights = compute_cluster_pca_fast(X_train.detach().cpu().numpy(), clusters, d=cluster_d)
        else:
            mus, covs, weights = compute_cluster_pca(X_train.detach().cpu().numpy(), clusters, d=cluster_d)
        # print(f"PCA took {time.thread_time() - t0:.2f} seconds")
        mixture_sampler = MixtureSampler(mus, covs, weights, clusters, inv_cluster, device=device)
        

    best_w2    = float("inf")
    best_model = None
    recs       = []

    for epoch in range(epochs):
        # global FM only
        t0 = time.thread_time()
        for i in range(0, M, batch_size):
            m       = min(batch_size, M-i) # batch size

            x0_t    = base_sampler(m)
            x1_t, _ = mixture_sampler.truncated_sample(m)
            t       = torch.rand(m*n_t_global, device=device)  # t sampling for global FM
            x0r     = x0_t.unsqueeze(1).repeat(1,n_t_global,1).view(-1,dim)
            x1r     = x1_t.unsqueeze(1).repeat(1,n_t_global,1).view(-1,dim)
            xt      = (1-t.unsqueeze(1))*x0r + t.unsqueeze(1)*x1r # OT interpolant

            target_v = x1r - x0r # target velocity
            pred_v   = model(xt, t)
            loss     = ((pred_v - target_v)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Global FM epoch {epoch} took {time.thread_time() - t0:.2f} seconds")

        # Validation
        t0 = time.thread_time()
        with torch.no_grad():
            X0   = np.random.randn(len(idx_val), dim)
            Xgen = run_flow(model, X0, device)
        w2 = np.sqrt(ot.emd2(np.ones(N-M)/(N-M), np.ones(N-M)/(N-M), ot.dist(Xgen.cpu().numpy(), X_val.cpu().numpy())**2))
        # print(f"Validation epoch {epoch} took {time.thread_time() - t0:.2f} seconds")

        # Early stopping
        if early_stopping and ((best_w2 - w2) < tol):
            stop_criteria += 1
            if stop_criteria >= 3:  # stop after 3 epochs without improvement
                break
        else:
            stop_criteria = 0
            best_w2 = w2
            best_model = model

        # Record results
        recs.append(
            { "epoch": epoch, 
              "validation_w2": w2, 
              "train_loss": loss.item() }
        ) 
    return mixture_sampler, epoch, best_w2, recs, best_model


def train_lfm(model, optimizer, X_target, dim, device,
              mixture_sampler=None,
              n_t_local=1,
              epochs=5, batch_size=256,    
              cluster_size=50, cluster_d=3,
              early_stopping=True, tol=1e-3,
              approx_cluster=False, fast_PCA=True):
    
    # 0) Clustering & Form intermediate distribution
    N            = X_target.shape[0]
    perm         = np.random.permutation(N)
    M            = N - min(int(0.1*N), 2000) # training set size (validation set cannot exceed 2000 by numItermax of pot)
    
    idx_train, idx_val = perm[:M], perm[M:] # split into train and validation sets
    X_train            = X_target[idx_train]
    X_val              = X_target[idx_val]
    
    stop_criteria = 0

    if mixture_sampler is None:
        t0 = time.thread_time()
        if approx_cluster:
            clusters, inv_cluster = cluster_points_annoy(X_train.detach().cpu().numpy(), cluster_size)
        else:
            clusters, inv_cluster  = cluster_points(X_train.detach().cpu().numpy(), cluster_size)
        # print(f"Clustering took {time.thread_time() - t0:.2f} seconds")
        t0 = time.thread_time()
        if fast_PCA:
            mus, covs, weights = compute_cluster_pca_fast(X_train.detach().cpu().numpy(), clusters, d=cluster_d)
        else:
            mus, covs, weights = compute_cluster_pca(X_train.detach().cpu().numpy(), clusters, d=cluster_d)
        # print(f"PCA took {time.thread_time() - t0:.2f} seconds")
        mixture_sampler = MixtureSampler(mus, covs, weights, clusters, inv_cluster, device=device)
    
    best_w2    = float("inf")
    best_model = None
    recs       = []

    for epoch in range(epochs):
        perm_t = torch.randperm(X_train.shape[0], device=device)

        # local FM only
        t0 = time.thread_time()
        for i in range(0, M, batch_size):
            m   = min(batch_size, M-i)
            idx = perm_t[i:i+m]
            
            inv_cluster = mixture_sampler.inv_cluster
            pis_np      = np.array([ np.random.choice(inv_cluster[int(j.item())]) for j in idx ], dtype=np.int64)
            pis         = torch.tensor(pis_np, device=device)

            x0, _ = mixture_sampler.truncated_sample(m, pis=pis)
            x1    = X_train[idx]            
            t     = torch.rand(m * n_t_local, device=device) # t sampling for local FM
            x0r   = x0.unsqueeze(1).repeat(1, n_t_local, 1).view(-1, dim)
            x1r   = x1.unsqueeze(1).repeat(1, n_t_local, 1).view(-1, dim)
            xt    = (1 - t.unsqueeze(1)) * x0r + t.unsqueeze(1) * x1r

            target_v = x1r - x0r
            pred_v   = model(xt, t)
            loss     = ((pred_v - target_v) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Local FM epoch {epoch} took {time.thread_time() - t0:.2f} seconds")

        # Validation
        t0 = time.thread_time()
        with torch.no_grad():
            X0, _ = mixture_sampler.truncated_sample(len(idx_val))
            Xgen  = run_flow(model, X0, device)
        w2 = np.sqrt(ot.emd2(np.ones(N-M)/(N-M), np.ones(N-M)/(N-M), ot.dist(Xgen.cpu().numpy(), X_val.cpu().numpy())**2))
        # print(f"Validation epoch {epoch} took {time.thread_time() - t0:.2f} seconds")

        # Early stopping
        if early_stopping and ((best_w2 - w2) < tol):
            stop_criteria += 1
            if stop_criteria >= 3:  # stop after 3 epochs without improvement
                break
        else:
            stop_criteria = 0
            best_w2 = w2
            best_model = model

        # Record results
        recs.append(
            { "epoch": epoch, 
              "validation_w2": w2, 
              "train_loss": loss.item() }
        ) 
    return mixture_sampler, epoch, best_w2, recs, best_model


def run_flow(model, x0, device, n_steps=100):
    # Robust input handling: x0 can be torch.Tensor, numpy.ndarray, list, etc.
    if torch.is_tensor(x0):
        x = x0.detach().clone().to(device=device, dtype=torch.float32)
    else:
        # numpy/list -> tensor
        x = torch.as_tensor(x0, dtype=torch.float32, device=device).clone()

    dt = 1 / n_steps
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=device, dtype=torch.float32)
            v = model(x, t)
            x = x + v * dt
    return x
