from abc import ABC, abstractmethod
import torch
import numpy as np
import ot  # Optimal Transport library for Wasserstein distance
import torch.optim as optim
from scipy.stats import truncnorm


class Distribution(ABC):
    """
    Abstract base class for distributions.
    Attributes:
        ambient_dim (int): Dimension of the ambient space.
        latent_dim  (int): Intrinsic or latent dimension of the support manifold.
        equation    (str): Mathematical description of the distribution on a manifold.
        name        (str): Name of the distribution.
    """
    def __init__(self, ambient_dim, device, latent_dim=None, equation="", name=None):
        self.ambient_dim = ambient_dim
        self.latent_dim  = latent_dim or ambient_dim
        self.equation    = equation
        self.name        = name or self.__class__.__name__
        self.device      = device

        print(f"Distribution {self.name} has n = {self.ambient_dim}, d = {self.latent_dim}, equation is {self.equation}")

    def wasserstein2_distance(self, x: np.array, n_samples: int) -> float:
        """
        Computes the Wasserstein-2 distance between this distribution and another.
        Uses samples from both distributions.
        """
        x1 = self.sample(n_samples)
        # Placeholder for actual Wasserstein-2 distance computation
        return np.sqrt(ot.emd2(np.ones(n_samples)/n_samples, np.ones(n_samples)/n_samples, ot.dist(x1.cpu().numpy(), x)**2))

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Draw `n` samples from the distribution.
        Returns a tensor of shape (n, ambient_dim).
        """
        pass

    @abstractmethod
    def geometric_alignment(self, x: torch.Tensor) -> float:
        """
        Calculates the geometric alignment to the support manifold.
        Returns the average distance of the samples to the manifold.
        """
        pass
    
    @abstractmethod
    def initialize_parameters(self):
        """
        Initializes any parameters needed for the distribution.
        This can be used to set up the distribution before sampling.
        """
        pass


class NormalDistribution(Distribution):
    def __init__(self, ambient_dim: int, device):
        eq = f"N(0, I₍{ambient_dim}₎)"
        super().__init__(ambient_dim, device, latent_dim=ambient_dim, equation=eq, name="NormalDistribution")

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.ambient_dim, device=self.device)

    def geometric_alignment(self, x: torch.Tensor) -> float:
        # Placeholder implementation
        return (x ** 2).mean(dim=1).float()
    
    def initialize_parameters(self):
        pass  # No parameters to initialize for normal distribution


class Quadratic_Uniform(Distribution):
    def __init__(
        self,
        ambient_dim: int,
        device: torch.device,
        latent_dim: int,
        noise_std: float = 1e-4,
        A: torch.Tensor = None,
        Q: torch.Tensor = None
    ):
        eq = f"x = A·z + zᵀ·Q·z + noise, z~B_{latent_dim}(0,1) ball"
        super().__init__(ambient_dim, device, latent_dim=latent_dim, equation=eq, name="QuadraticManifoldDistribution")
        self.latent_dim = latent_dim
        self.noise_std  = noise_std
        self.A = A or torch.randn(ambient_dim, latent_dim, device=device)
        self.Q = Q or torch.randn(ambient_dim, latent_dim, latent_dim, device=device)

    def sample(self, n: int) -> torch.Tensor:
        # sample latent z on a sphere of radius sampled uniformly
        z = torch.randn(n, self.latent_dim, device=self.device)
        z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-6) * torch.rand(n,1, device=self.device)
        quad = torch.einsum('ni,kij,nj->nk', z, self.Q, z)
        linear = z @ self.A.T
        return linear + quad + self.noise_std * torch.randn_like(linear)

    def geometric_alignment(self, x: torch.Tensor) -> float:
        """
        Projects point(s) x onto the quadratic manifold by solving
            min_z  || x - (A z + zᵀ Q z) ||²
        via gradient descent. Returns the mean squared error.
        """
        num_steps = 100
        lr = 1e-2

        # initialize latent estimates
        z_hat = torch.zeros(x.shape[0], self.latent_dim, device=self.device, requires_grad=True)
        optimizer = optim.Adam([z_hat], lr=lr)

        # gradient descent to find z_hat that minimizes the error
        for _ in range(num_steps):
            optimizer.zero_grad()
            # linear term
            lin = z_hat @ self.A.T 
            # quadratic term
            quad = torch.einsum('ni,kij,nj->nk', z_hat, self.Q, z_hat)
            # prediction
            x_hat = lin + quad
            loss = ((x_hat - x) ** 2).mean()
            loss.backward()
            optimizer.step()

        # compute final error
        with torch.no_grad():
            lin  = z_hat @ self.A.T
            quad = torch.einsum('ni,kij,nj->nk', z_hat, self.Q, z_hat)
            err  = ((lin + quad - x) ** 2).mean().item()

        return err

    def initialize_parameters(self):
        """
        Initializes the parameters A and Q for the quadratic manifold distribution.
        This can be used to set up the distribution before sampling.
        """
        self.A = torch.randn(self.ambient_dim, self.latent_dim, device=self.device)
        self.Q = torch.randn(self.ambient_dim, self.latent_dim, self.latent_dim, device=self.device)


class Quadratic_Unimodal(Distribution):
    def __init__(
        self,
        ambient_dim: int,
        device: torch.device,
        latent_dim: int,
        noise_std: float = 1e-4,
        A: torch.Tensor = None,
        Q: torch.Tensor = None
    ):
        eq = f"x = A·z + zᵀ·Q·z + noise, z~N(0, I_{latent_dim})"
        super().__init__(ambient_dim, device, latent_dim=latent_dim, equation=eq, name="Quadratic_Unimodal")
        self.noise_std = noise_std
        self.A = A or torch.randn(ambient_dim, latent_dim, device=device) / np.sqrt(ambient_dim)
        self.Q = Q or torch.randn(ambient_dim, latent_dim, latent_dim, device=device) / np.sqrt(ambient_dim * latent_dim)

    def sample(self, n: int) -> torch.Tensor:
        # Sample n latent vectors from truncated normal
        z = torch.randn(n, self.latent_dim, device=self.device)
        '''
        r_np = truncnorm.rvs(0, 1, size=(n, 1))
        r = torch.from_numpy(r_np.astype(np.float32)).to(self.device)
        z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-6) * r
        '''
        # Map onto quadratic manifold
        quad = torch.einsum('ni,kij,nj->nk', z, self.Q, z)
        linear = z @ self.A.T
        return linear + quad + self.noise_std * torch.randn_like(linear)

    def geometric_alignment(self, x: torch.Tensor) -> float:
        """
        Projects point(s) x onto the quadratic manifold by solving
            min_z  || x - (A z + zᵀ Q z) ||²
        via gradient descent. Returns the mean squared error.
        """
        num_steps = 5000
        lr = 1e-2

        # initialize latent estimates
        z_hat = torch.zeros(x.shape[0], self.latent_dim, device=self.device, requires_grad=True)
        optimizer = optim.Adam([z_hat], lr=lr)
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        # gradient descent to find z_hat that minimizes the error
        for i in range(num_steps):
            optimizer.zero_grad()
            # linear term
            lin   = z_hat @ self.A.T 
            # quadratic term
            quad  = torch.einsum('ni,kij,nj->nk', z_hat, self.Q, z_hat)

            """
            with torch.no_grad():
                lin_norm  = lin.norm(dim=1).mean().item()
                quad_norm = quad.norm(dim=1).mean().item()
                print(f"[Align] ||Az|| = {lin_norm:.4e}, ||z^T Q z|| = {quad_norm:.4e}")
            """

            # prediction
            x_hat = lin + quad

            loss  = ((x_hat - x) ** 2).mean()
            loss.backward()
            optimizer.step()

            # Early stopping condition
            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:  # small threshold
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            if (i == num_steps - 1):
                print("Reached Max Iteration for geometric alignment error!!")

        # compute final error
        with torch.no_grad():
            lin   = z_hat @ self.A.T
            quad  = torch.einsum('ni,kij,nj->nk', z_hat, self.Q, z_hat)
            x_hat = lin + quad
            err   = ((x_hat - x) ** 2).mean().item()
            # print("Error by gradient descent: ", err)
            
            """
            c     = x @ torch.linalg.pinv(self.A).T          # (n, latent_dim)
            recon = torch.matmul(c, self.A.T)   # (n, ambient_dim)
            res   = x - recon             # (n, ambient_dim)
            proj_err = res.pow(2).mean().item()
            print("Error by projection: ", proj_err)
            """

        return err
    
    def initialize_parameters(self):
        """
        Initializes the parameters A and Q for the quadratic unimodal distribution.
        This can be used to set up the distribution before sampling.
        """
        self.A = torch.randn(self.ambient_dim, self.latent_dim, device=self.device) / np.sqrt(self.ambient_dim)
        self.Q = torch.randn(self.ambient_dim, self.latent_dim, self.latent_dim, device=self.device) / np.sqrt(self.ambient_dim * self.latent_dim)


class Quadratic_Multimodal(Distribution):
    def __init__(
        self,
        ambient_dim: int,
        device: torch.device,
        latent_dim: int,
        noise_std: float = 1e-4,
        A: torch.Tensor = None,
        Q: torch.Tensor = None,
        mode_num: int = 10
    ):
        eq = f"x = A·z + zᵀ·Q·z + noise, z~sum(N_i(u_i, I_{latent_dim}))"
        super().__init__(ambient_dim, device, latent_dim=latent_dim, equation=eq, name="Quadratic_Multimodal")
        self.noise_std = noise_std
        self.A = A or torch.randn(ambient_dim, latent_dim, device=device) / np.sqrt(ambient_dim)
        self.Q = Q or torch.randn(ambient_dim, latent_dim, latent_dim, device=device) / np.sqrt(ambient_dim * latent_dim)
        self.mode_num = mode_num

        # Sample modes
        z = torch.randn(mode_num, self.latent_dim, device=self.device)
        self.modes = z / z.norm(dim=1, keepdim=True).clamp(min=1e-6)
        
    def sample(self, n: int) -> torch.Tensor:
        mode_indices = torch.randint(0, self.mode_num, (n,), device=self.device) # Sample n mode indices
        
        # Sample n latent vectors from truncated GMM
        z = torch.randn(n, self.latent_dim, device=self.device)
        z = self.modes[mode_indices] + z
        
        # Map onto quadratic manifold
        quad = torch.einsum('ni,kij,nj->nk', z, self.Q, z)
        linear = z @ self.A.T
        return linear + quad + self.noise_std * torch.randn_like(linear)

    def geometric_alignment(self, x: torch.Tensor) -> float:
        """
        Projects point(s) x onto the quadratic manifold by solving
            min_z  || x - (A z + zᵀ Q z) ||²
        via gradient descent. Returns the mean squared error.
        """
        num_steps = 5000
        lr = 1e-2

        # initialize latent estimates
        z_hat = torch.zeros(x.shape[0], self.latent_dim, device=self.device, requires_grad=True)
        optimizer = optim.Adam([z_hat], lr=lr)
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        # gradient descent to find z_hat that minimizes the error
        for i in range(num_steps):
            optimizer.zero_grad()
            # linear term
            lin   = z_hat @ self.A.T 
            # quadratic term
            quad  = torch.einsum('ni,kij,nj->nk', z_hat, self.Q, z_hat)
            # prediction
            x_hat = lin + quad
            loss  = ((x_hat - x) ** 2).mean()
            loss.backward()
            optimizer.step()

            # Early stopping condition
            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:  # small threshold
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            if (i == num_steps - 1):
                print("Reached Max Iteration for geometric alignment error!!")

        # compute final error
        with torch.no_grad():
            lin   = z_hat @ self.A.T
            quad  = torch.einsum('ni,kij,nj->nk', z_hat, self.Q, z_hat)
            x_hat = lin + quad
            err   = ((x_hat - x) ** 2).mean().item()

        return err
    
    def initialize_parameters(self):
        """
        Initializes the parameters A, Q, and modes for the quadratic multimodal distribution.
        This can be used to set up the distribution before sampling.
        """
        self.A = torch.randn(self.ambient_dim, self.latent_dim, device=self.device) / np.sqrt(self.ambient_dim)
        self.Q = torch.randn(self.ambient_dim, self.latent_dim, self.latent_dim, device=self.device) / np.sqrt(self.ambient_dim * self.latent_dim)
        self.modes = torch.randn(self.mode_num, self.latent_dim, device=self.device)


class Linear_Branched(Distribution):
    def __init__(
            self,
            ambient_dim: int,
            device: torch.device,
            latent_dim: int,
            noise_std: float = 1e-4,
            branch_num: int = 3,
    ):
        eq = f"x = A_i z + offset + noise, A_i=[v_i_1, ..., v_i_{latent_dim}], z~sum(N_i(u_i, I_{latent_dim}))"
        super().__init__(ambient_dim, device, latent_dim=latent_dim, equation=eq, name="Linear_Branched")
        self.noise_std = noise_std
        self.branch_num = branch_num
        self.basis = torch.randn(ambient_dim, latent_dim, device=device) / np.sqrt(ambient_dim)  # Random basis for branches
        # self.offset = torch.randn(ambient_dim, device=device)  # shared center offset
        row_idx = torch.stack([
            torch.randperm(ambient_dim, device=device)
            for _ in range(branch_num)
        ], dim=0)  # index for basis rows
        self.branches = self.basis[row_idx]  # basis A_i for each branch
        self.branches_pinv = torch.linalg.pinv(self.branches)  # precompute pseudo-inverse for each branch

    def sample(self, n: int) -> torch.Tensor:
        pis = torch.randint(
            low=0,
            high=self.branch_num,
            size=(n,),
            device=self.device
        )  # sample branch indices
        
        # Sample latent vector from truncated normal
        z      = torch.randn(n, self.latent_dim, device=self.device)
        # Map onto branched manifold
        A_sel  = self.branches[pis] # Select the appropriate branch basis
        linear = torch.bmm(A_sel, z.unsqueeze(-1)).squeeze(-1)  # Linear transformation with branch basis and offset

        return linear + self.noise_std * torch.randn_like(linear)

    def geometric_alignment(self, x: torch.Tensor) -> float:
        """
        Compute the overall mean squared perpendicular error (MSE) of the batch x
        to the closest branch-affine subspace. Returns a single float.
        """
        x = x.to(self.device)
        xp = x # - self.offset  # center the points around the offset

        # squared errors per sample per branch: (n, branch_num)
        sq_err = torch.empty(xp.size(0), self.branch_num, device=self.device)
        for i in range(self.branch_num):
            A      = self.branches[i]      # branch basis (ambient_dim, latent_dim)
            A_pinv = self.branches_pinv[i] # branch pseudo-inverse (latent_dim, ambient_dim)

            # project and reconstruct
            c     = xp @ A_pinv.T          # (n, latent_dim)
            recon = torch.matmul(c, A.T)   # (n, ambient_dim)
            res   = xp - recon             # (n, ambient_dim)

            # sum squared errors
            sq_err[:, i] = res.pow(2).sum(dim=1)

        # minimum squared error per sample, then mean across samples
        min_sq = torch.min(sq_err, dim=1).values  # (n,)
        return min_sq.mean().item()
    
    def initialize_parameters(self):
        """
        Initializes the parameters for the branched linear distribution.
        This can be used to set up the distribution before sampling.
        """
        self.basis = torch.randn(self.ambient_dim, self.latent_dim, device=self.device) / np.sqrt(self.ambient_dim)
        # self.offset = torch.randn(self.ambient_dim, device=self.device)
        row_idx = torch.stack([
            torch.randperm(self.ambient_dim, device=self.device)
            for _ in range(self.branch_num)
        ], dim=0)
        self.branches = self.basis[row_idx]  # A_i for each branch
        self.branches_pinv = torch.linalg.pinv(self.branches)


class SwissRoll(Distribution):
    def __init__(
        self,
        ambient_dim: int,
        device: torch.device,
        latent_dim: int,
        noise_std: float = 1e-4,
    ):
        eq = (
            f"x = A [cos(t1), sin(t1), t2...t_{latent_dim}, 0...0], t1~U(0,2π), [t2...t_{latent_dim}]~N(0,1)"
        )
        super().__init__(
            ambient_dim,
            device,
            latent_dim=latent_dim,
            equation=eq,
            name="SwissRoll"
        )
        self.noise_std = noise_std
        self.device = device
        self.latent_dim = latent_dim
        self.ambient_dim = ambient_dim

        # Random linear transformation A (ambient_dim×ambient_dim)
        self.A = torch.randn(ambient_dim, ambient_dim, device=device) / np.sqrt(ambient_dim)
        self.A_pinv = torch.linalg.pinv(self.A)

    def sample(self, n: int) -> torch.Tensor:
        """
        Draw n samples on the swiss-roll manifold embedding.
        Returns:
            x: (n, ambient_dim) tensor
        """
        # draw t1 from [0, 1] and gaussian t2...t_(latent_dim)
        t1 = torch.rand(n, device=self.device)
        if self.latent_dim > 1:
            gauss = torch.randn(n, self.latent_dim - 1, device=self.device)
        else:
            gauss = None

        # form v ∈ ℝ^ambient_dim
        v = torch.zeros(n, self.ambient_dim, device=self.device)
        v[:, 0] = t1 * torch.cos(4 * torch.pi * t1)
        v[:, 1] = t1 * torch.sin(4 * torch.pi * t1)
        if gauss is not None:
            v[:, 2:self.latent_dim+1] = gauss  # fill t2...t_latent_dim

        # apply A (linear transformation)
        x = v @ self.A.T
        return x + self.noise_std * torch.randn_like(x)

    def geometric_alignment(self, x: torch.Tensor) -> float:
        """
        Estimate latent coordinates via gradient descent to minimize:
            || x - A v(z) ||^2
        where v(z) = [t1 cos(4*pi*t1), t1 sin(4*pi*t1), z2...z_latent_dim].
        Returns the mean squared error over the batch.
        """
        x = x.to(self.device)
        n = x.shape[0]

        # Initialize z_hat from linear pseudo-inverse (first latent_dim entries)
        z_init = (x @ self.A_pinv.T)[:, 1:]  # (n, ambient_dim)
        z_init[:, 0] = torch.zeros_like(z_init[:, 0])

        # keep only t1 and gaussian dims
        z_hat = z_init[:, :self.latent_dim].clone().detach()
        z_hat.requires_grad_(True)

        optimizer = optim.Adam([z_hat], lr=2e-1)

        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        num_steps = 100000

        for i in range(num_steps):
            optimizer.zero_grad()
            t1 = torch.sigmoid(z_hat[:, 0])  # (n,), within [0, 1]

            # build v from z_hat
            v = torch.zeros(n, self.ambient_dim, device=self.device)
            v[:, 0] = t1 * torch.cos(4 * torch.pi * t1)
            v[:, 1] = t1 * torch.sin(4 * torch.pi * t1)
            if self.latent_dim > 1:
                v[:, 2 : self.latent_dim + 1] = z_hat[:, 1:]

            x_hat = v @ self.A.T  # (n, ambient_dim)
            loss = torch.mean((x_hat - x) ** 2)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:  # small threshold
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            if (i == num_steps - 1):
                print("Reached Max Iteration for geometric alignment error!!")

        # final MSE
        with torch.no_grad():
            t1 = torch.sigmoid(z_hat[:, 0])
            v = torch.zeros(n, self.ambient_dim, device=self.device)
            v[:, 0] = t1 * torch.cos(4 * torch.pi * t1)
            v[:, 1] = t1 * torch.sin(4 * torch.pi * t1)
            if self.latent_dim > 1:
                v[:, 2 : self.latent_dim + 1] = z_hat[:, 1:]

            x_hat = v @ self.A.T
            mse = torch.mean((x_hat - x) ** 2).item()

        return mse

    def initialize_parameters(self):
        """
        Initializes the parameters A for the Swiss Roll distribution.
        This can be used to set up the distribution before sampling.
        """
        self.A = torch.randn(self.ambient_dim, self.ambient_dim, device=self.device) / np.sqrt(self.ambient_dim)
        self.A_pinv = torch.linalg.pinv(self.A)