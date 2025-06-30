import numpy as np
import torch
import torch.nn as nn
import ot
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class VectorField(nn.Module):
    # ----------------------------
    # Neural Vector Field for Flow Matching
    # ----------------------------
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, dim)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(1)], dim=1))


class MixtureSampler:
    # ----------------------------
    # Intermediate Mixture of Gaussians sampler
    # ----------------------------
    def __init__(self, mus, covs, weights, clusters, device='cpu'):
        self.device = device
        self.mus = torch.tensor(mus, dtype=torch.float32, device=device) # (k, d)
        self.covs = torch.tensor(covs, dtype=torch.float32, device=device) # (k, d, d)
        self.weights = torch.tensor(weights/weights.sum(),dtype=torch.float32, device=device)  # normalize weights
        self.clusters = clusters
        self.Ls = torch.linalg.cholesky(self.covs)
        self.k, self.d = self.mus.shape

    def sample(self, M):
        pis = torch.multinomial(self.weights, M, replacement=True)             # (M,)

        eps = torch.randn(M, self.d, device=self.device)                      # (M, d)

        # gather the Cholesky for each sample
        L_sel   = self.Ls[pis]
        mu_sel  = self.mus[pis]
        
        # transform: x = mu + L @ eps
        x = torch.bmm(L_sel, eps.unsqueeze(-1)).squeeze(-1)  # (M, d)
        return x + mu_sel, pis
    
    def truncated_sample(self, M, factor=2.0):
        """
        Draw ~factor*M samples, keep the first M that lie within ±2σ
        (coordinate-wise), and if there aren’t enough, sample the rest.
        Args:
            M      : number of final samples wanted
            factor : oversampling factor (>1) to reduce loops
        Returns:
            x      : (M, d) tensor of truncated samples
            pis    : (M,) tensor of component indices
        """
        # precompute per-component std-dev along each axis (k, d)
        stds = torch.sqrt(torch.diagonal(self.covs, dim1=1, dim2=2))

        out_x   = []
        out_pis = []

        remaining = M
        while remaining > 0:
            # oversample
            N = int(remaining * factor)
            x, pis = self.sample(N)

            # filter valid ones
            mu_sel   = self.mus[pis]        # (N, d)
            std_sel  = stds[pis]            # (N, d)
            mask     = ((x - mu_sel).abs() <= 2 * std_sel).all(dim=1)

            x_valid   = x[mask]
            pis_valid = pis[mask]

            # take up to 'remaining'
            take = min(x_valid.size(0), remaining)
            out_x.append(  x_valid[:take]   )
            out_pis.append(pis_valid[:take])

            remaining -= take

        # concatenate and return exactly M
        x_final   = torch.cat(out_x,   dim=0)[:M]
        pis_final = torch.cat(out_pis, dim=0)[:M]
        return x_final, pis_final
    

def train_vanilla_FM(model, optimizer, X_target, dim, device,
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

        # Record results
        recs.append(
            { "epoch": epoch, 
              "validation_w2": w2, 
              "train_loss": loss.item() }
        )
    return epoch, best_w2, recs

def cluster_points(X, m, delta=np.inf):
    # ----------------------------
    # Overlapping clusters formed with size & diameter constraints
    # Inputs - X: data points, m: min cluster size, delta: max cluster diameter
    # Outputs - clusters: list of sets of indices
    # ----------------------------
    N = X.shape[0]
    visited = np.zeros(N, dtype=bool)
    clusters = []
    neighbors_model = NearestNeighbors(n_neighbors=min(m*5, N)).fit(X)

    while not visited.all():
        # Pick first unvisited point
        seed_idx = np.where(~visited)[0][0]
        _, neighbors = neighbors_model.kneighbors([X[seed_idx]], n_neighbors=N, return_distance=True)
        neighbors = neighbors[0]

        # Find first m unvisited neighbors
        cluster = []
        for idx in neighbors:
            if not visited[idx]:
                cluster.append(idx)
            if len(cluster) >= m:
                break

        if len(cluster) < m:
            # assign remaining unvisited points to nearest existing cluster
            for i in np.where(~visited)[0]:
                d = [np.linalg.norm(X[i] - X[list(c)].mean(axis=0)) for c in clusters]
                best = np.argmin(d)
                clusters[best].add(i)
                visited[i] = True

        clusters.append(set(cluster))
        visited[cluster] = True

    return clusters


def compute_cluster_pca(X, clusters, d, eps=1e-3):
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
    for c in clusters:
        idx = list(c)
        Xi = X[idx]
        mu = Xi.mean(axis=0)
        pca = PCA(n_components=d).fit(Xi)
        Bi = pca.components_.T  # D x d
        Lambda = np.diag(pca.explained_variance_)
        Sigma = Bi @ np.clip(Lambda, a_min=eps, a_max=None) @ Bi.T + eps * np.eye(X.shape[1])
        Sigma = (Sigma + Sigma.T)/2
        mus.append(mu)
        covs.append(Sigma)
        weights.append(len(idx) / N)
    return np.array(mus), np.array(covs), np.array(weights)


def train_dgfm(model, optimizer, X_target, dim, mf, device,
               mixture_sampler=None,
               n_t_global=5, n_t_local=1,
               epochs=5, batch_size=256,    
               cluster_size=50, cluster_d=3,
               early_stopping=True, tol=1e-3):
    
    # 0) Clustering & Form intermediate distribution
    base_sampler = lambda x: torch.randn(x, dim, device=device)

    N = X_target.shape[0]
    perm = np.random.permutation(N)
    M = int(0.9*N) # training set size
    idx_train, idx_val = perm[:M], perm[M:] # split into train and validation sets
    X_train = X_target[idx_train]
    X_val   = X_target[idx_val]
    stop_criteria = 0

    if mixture_sampler is None:
        t0 = time.thread_time()
        clusters  = cluster_points(X_train.detach().cpu().numpy(), cluster_size, delta=np.inf)
        mus, covs, weights = compute_cluster_pca(X_train.detach().cpu().numpy(), clusters, d=cluster_d)
        mixture_sampler = MixtureSampler(mus, covs, weights, clusters, device=device)
        # print(f"Clustering took {time.thread_time() - t0:.2f} seconds")

    best_w2 = float("inf")
    recs = []

    global_M = M * mf
    for epoch in range(epochs):
        # 1) global FM
        t0 = time.thread_time()
        for i in range(0, global_M, batch_size):
            m = min(batch_size, global_M-i) # batch size
            x0_t = base_sampler(m)
            x1_t, _ = mixture_sampler.truncated_sample(m)
            t  = torch.rand(m*n_t_global, device=device)  # t sampling for global FM

            x0r = x0_t.unsqueeze(1).repeat(1,n_t_global,1).view(-1,dim)
            x1r = x1_t.unsqueeze(1).repeat(1,n_t_global,1).view(-1,dim)
            xt  = (1-t.unsqueeze(1))*x0r + t.unsqueeze(1)*x1r
            target_v = 2*(x1r - x0r)
            pred_v   = model(xt, t)
            loss = ((pred_v - target_v)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Global FM epoch {epoch} took {time.thread_time() - t0:.2f} seconds")

        # 2) local FM
        t0 = time.thread_time()
        for i in range(0, M, batch_size):
            m = min(batch_size, M-i)
            
            # Match within clusters
            x0, pis = mixture_sampler.truncated_sample(m)
            idxs = [int(np.random.choice(list(clusters[pi]))) for pi in pis]
            x1 = X_train[idxs]

            t = torch.rand(m * n_t_local, device=device) # t sampling for local FM

            x0r = x0.unsqueeze(1).repeat(1, n_t_local, 1).view(-1, dim)
            x1r = x1.unsqueeze(1).repeat(1, n_t_local, 1).view(-1, dim)
            xt = (1 - t.unsqueeze(1)) * x0r + t.unsqueeze(1) * x1r
            target_v = 2*(x1r - x0r)
            pred_v = model(xt, t)
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

        # Record results
        recs.append(
            { "epoch": epoch, 
              "validation_w2": w2, 
              "train_loss": loss.item() }
        ) 
    return mixture_sampler, epoch, best_w2, recs

def run_flow(model, x0, device, n_steps=100):
    x = torch.tensor(x0, device=device, dtype=torch.float32)
    dt = 1/n_steps
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i*dt, device=device)
            v = model(x, t)
            x = x + v*dt
    return x
