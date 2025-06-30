from abc import ABC, abstractmethod
import torch
import numpy as np
import ot  # Optimal Transport library for Wasserstein distance
import torch.optim as optim


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

    def wasserstein2_distance(self, x: np.array, n_samples: int) -> float:
        """
        Computes the Wasserstein-2 distance between this distribution and another.
        Uses samples from both distributions.
        """
        x1 = self.sample(n_samples)
        # Placeholder for actual Wasserstein-2 distance computation
        return np.sqrt(ot.emd2(np.ones(n_samples)/n_samples, np.ones(n_samples)/n_samples, ot.dist(x1.cpu().numpy(), x)**2))

    def cross_entropy(self, x: torch.Tensor, num_samples: int) -> float:
        """
        Computes the cross entropy of the samples `x` under the distribution.
        Returns the cross entrop value (float).
        """
        X = self.sample(num_samples).to(self.device)
        

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Draw `n` samples from the distribution.
        Returns a tensor of shape (n, ambient_dim).
        """
        pass

    @abstractmethod
    def geometric_alignment(self, x: torch.Tensor, device) -> float:
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

    def geometric_alignment(self, x: torch.Tensor, device) -> float:
        # Placeholder implementation
        return (x ** 2).mean(dim=1).float()
    
    def initialize_parameters(self):
        pass  # No parameters to initialize for normal distribution

class QuadraticManifoldDistribution(Distribution):
    def __init__(
        self,
        ambient_dim: int,
        device: torch.device,
        latent_dim: int,
        noise_std: float = 1e-4,
        A: torch.Tensor = None,
        Q: torch.Tensor = None
    ):
        eq = "x = A·z + zᵀ·Q·z + noise"
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

    def geometric_alignment(self, x: torch.Tensor, device) -> float:
        """
        Projects point(s) x onto the quadratic manifold by solving
            min_z  || x - (A z + zᵀ Q z) ||²
        via gradient descent. Returns the mean squared error.
        """
        num_steps = 100
        lr = 1e-2

        # initialize latent estimates
        z_hat = torch.zeros(x.shape[0], self.latent_dim, device=device, requires_grad=True)
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