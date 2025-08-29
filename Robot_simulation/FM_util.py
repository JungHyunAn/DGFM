"""
Flow Matching (FM) training & evaluation functions on RoboSuite
===============================================================

This module implements the training / inference building blocks for learning
trajectory generators with Flow Matching on RoboSuite tasks. It includes:

What this module provides
-------------------------
- **VectorField**: a conditional velocity field (1D U-Net backbone with FiLM)
  mapping (x_t, t, c) → v_t used by UniformFM / ShiftedFM / DGFM trainers.
- **MixtureSampler**: a low-rank conditional Gaussian mixture over trajectories
  and environment parameters used by **Dimension-Guided FM (DGFM)** to provide
  intermediate (interpolated) samples.
- **Trainers**: `train_uniform_FM`, `train_shifted_FM`, `train_DGFM`—each returns
  `(best_model, last_model, records)` with validation metrics over epochs.
- **Evaluation**: `eval_model` runs many parallel rollouts in RoboSuite,
  reports success rate / average reward, and (optionally) renders a grid video.
- **Utilities**: lightweight helpers like `run_flow` for integrating the learned
  vector field from t=0→1.

Key design choices & assumptions
--------------------------------
1) **Shapes & conventions**
   - Trajectories are `(B, T, D)` = (batch, seq_len, dof).
   - Environment parameters are `(B, P)`.
   - Times `t ∈ [0, 1]`.
   - For gripper DoF, models can ignore (mask) those joints during training.

2) **DGFM's low-rank mixture (used only when training DGFM)**
   - Per mixture component *k* we model:
       x = μ_x^k + B^k z,  with  z|c ~ N(μ_{z|c}^k, Σ_{z|c}^k),   and   c ~ N(μ_c^k, Σ_cc^k)
     where B^k ∈ R^{D x d} is a PCA basis (d < D). We precompute for each cluster:
       Σ_zz^k, Σ_zc^k, Σ_cc^k and Cholesky factors; responsibilities use p(c|k) only.
   - Conditioning uses standard Gaussian identities:
       μ_{z|c}^k = Σ_zc^k {Σ_cc^k}^{-1} (c - μ_c^k),
       Σ_{z|c}^k = Σ_zz^k - Σ_zc^k {Σ_cc^k}^{-1} Σ_cz^k.
   - Sampling supports optional truncated normal in z (for robustness) and a small
     orthogonal noise on x (σ⊥) to account for off-subspace variability.
   - Numerical stability: all covariances are symmetrized and Tikhonov-regularized
     before Cholesky; log-determinants are cached.

3) **Trainers**
   - **UniformFM**: sample t ~ Uniform[0,1], regress v_t (standard FM).
   - **ShiftedFM**: shift / warp t to emphasize early (or late) time steps.
   - **DGFM**: use the mixture to define intermediate targets (configurable via `mf`, 
     `cluster_d`, `cluster_size`, `n_t_global`, `n_t_local`).
   - Schedules: cosine with warmup; Adam optimizer; early stopping supported.

4) **Evaluation loop**
   - For each trial: build env, extract env params c, sample a trajectory by
     integrating the vector field (via `run_flow`), upsample / smooth externally,
     replay, then compute success / reward. Multiprocessing is used for speed.
   - Rendering (optional) uses the external utilities from
     `Robot_simulation.heuristics_util` (not defined here).

Interfaces at a glance
----------------------
- `VectorField(seq_len, dof, param_len, gripper_idx=None)`
- `MixtureSampler(mu_x, mu_c, B, Sig_zz, Sig_zc, Sig_cc, weights, device='cpu', reg=1e-6, orth_sigma=0.0)`
    - `.sample_cond(c_in, deterministic_component=False, truncated=False, trunc=(-1.5,1.5), pis=None)`
    - `.sample_joint(M, truncated=False, trunc=(-1.5,1.5), pis=None)`
- `train_uniform_FM(model, optimizer, scheduler, task_name, target_trajectories, environment_parameters, ..., early_stopping, ...)`
- `train_shifted_FM(...)`
- `train_DGFM(..., mf, n_t_global, n_t_local, cluster_d, cluster_size, ...)`
- `eval_model(model, model_class, task_name, seq_len, dof, param_len, gripper_idx, device, trials, render_dir=..., ...)`
- `run_flow(model, x0, env_param, device, n_steps=500)`

Outputs & metrics
-----------------
- Trainers return: `best_model`, `last_model`, and a `records` dict per epoch
  (e.g., validation success rate).
- `eval_model` returns: `(success_rate, mean_reward)` and can save a grid video.

Gotchas / tips
--------------
- Ensure environment parameter dimension `P` matches the trained model; mixture
  responsibilities depend on `Σ_cc`.
- If Cholesky fails or produces NaNs, increase `reg` (Tikhonov) or check for
  degenerate clusters / too few inliers during PCA statistics.
- When forcing components (`pis`) in `.sample_cond`, responsibilities are skipped
  (weights return `None`), which is useful for ablating mixture selection.
- If you mask grippers during training, be sure the downstream replay / rendering
  logic reconstructs or clamps gripper channels consistently.

Note
----
This module focuses on learning and evaluation. Environment construction,
state restoration, trajectory smoothing, and rendering are provided by
`Robot_simulation.heuristics_util` and task-specific heuristic generators.
"""

import numpy as np
import random
import time
import os
import copy
import torch
import torch.nn as nn
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Tuple, List
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import truncnorm, chi2
from torch.distributions import Beta
from joblib import Parallel, delayed
import logging
logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)  
robosuite_logger.propagate = False        
for h in list(robosuite_logger.handlers): 
    robosuite_logger.removeHandler(h)

from Robot_simulation.heuristics_util import make_env, compute_smooth_trajectory, render_trajectory, write_grid_video

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for 1D feature maps.

    Applies a per-channel affine transform conditioned on a context vector.

    Args:
    in_channels: Number of channels in the feature map to be modulated.
    condition_dim: Dimensionality of the conditioning vector.

    Forward Args:
    x: Tensor of shape (B, C, T), feature map to modulate.
    condition: Tensor of shape (B, condition_dim), conditioning vector.

    Returns:
    Tensor of shape (B, C, T) after FiLM modulation.
    """
    def __init__(self, in_channels, condition_dim):
        super().__init__()
        self.scale_shift = nn.Sequential(
            nn.Linear(condition_dim, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels * 2)
        )

    def forward(self, x, condition):
        # x: (B, C, T), condition: (B, cond_dim)
        scale_shift = self.scale_shift(condition)           # (B, 2*C)
        scale, shift = scale_shift.chunk(2, dim=-1)         # each (B, C)
        scale = scale.unsqueeze(-1)                        # (B, C, 1)
        shift = shift.unsqueeze(-1)                        # (B, C, 1)
        return x * (1 + scale) + shift


class ConvBlock(nn.Module):
    """Conv → GroupNorm → ReLU → FiLM block for 1D sequences.

    Args:
    in_channels: Input channels to the conv layer.
    out_channels: Output channels for the conv layer.
    condition_dim: Dimensionality of the conditioning vector used by FiLM.

    Forward Args:
    x: (B, C_in, T) input tensor.
    condition: (B, condition_dim) conditioning vector.

    Returns:
    (B, C_out, T) tensor after convolution, normalization, activation, and FiLM.
    """
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.film = FiLM(out_channels, condition_dim)

    def forward(self, x, condition):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.film(x, condition)
        return x


class UNet1D(nn.Module):
    """A 1D U-Net with FiLM conditioning.

    This backbone models per-timestep dynamics across joint sequences.

    Args:
    dof: Number of channels == robot DoF to model (typically arm DoF only).
    condition_dim: Dimensionality of the environmental condition fed into FiLM.
    channels: Encoder channel progression; decoder mirrors these.

    Forward Args:
    x: (B, dof, T) input sequence (channels first).
    condition: (B, condition_dim) vector.

    Returns:
    (B, dof, T) tensor of predicted velocities.
    """
    def __init__(self, dof, condition_dim, channels=[160, 320, 640, 640]):
        super().__init__()
        self.channels = channels
        # Encoder
        self.enc_blocks = nn.ModuleList()
        in_ch = dof
        for ch in channels:
            self.enc_blocks.append(ConvBlock(in_ch, ch, condition_dim))
            in_ch = ch

        # Bottleneck
        self.bottleneck = ConvBlock(channels[-1], channels[-1], condition_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        prev_ch = channels[-1]
        for skip_ch in reversed(channels):
            in_ch = prev_ch + skip_ch
            out_ch = skip_ch
            self.dec_blocks.append(ConvBlock(in_ch, out_ch, condition_dim))
            prev_ch = out_ch

        # Final layer
        self.final_conv = nn.Conv1d(channels[0], dof, kernel_size=1)

    def forward(self, x, condition):
        # x: (B, dof, seq_len), condition: (B, cond_dim)
        skips = []
        # Encoding path
        for enc in self.enc_blocks:
            x = enc(x, condition)
            skips.append(x)
            x = F.avg_pool1d(x, kernel_size=2, ceil_mode=True)

        # Bottleneck
        x = self.bottleneck(x, condition)

        # Decoding path
        for dec, skip in zip(self.dec_blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x, condition)

        # Final projection
        return self.final_conv(x)


class VectorField(nn.Module):
    """Conditional vector field v(x, t, c) over joint trajectories.

    The model isolates arm DoF from gripper DoF and predicts a velocity field
    only for arm joints using a 1D U-Net; gripper channels are held at zero
    by a built-in mask. Time is encoded via a small MLP and concatenated with
    (optionally scaled) environment parameters for FiLM conditioning.

    Args:
        seq_len: Number of time steps (T) per trajectory.
        dof: Total joint DoF per time step (includes grippers if present).
        param_len: Dimensionality of environment parameter vector `c`.
        gripper_idx: Indices of gripper joints within the DoF. If provided,
            these channels are excluded from the U-Net and zeroed in the output.

    Attributes:
        arm_idx (Tensor): Indices of arm joints modeled by the U-Net.
        loss_mask (Tensor): (1, 1, dof) mask that zeros gripper dims in loss.

    Forward Args:
        x: (B, T, dof) current state on the straight path between x0 and x1.
        t: (B, 1) time in [0, 1].
        env_params: (B, param_len) environment parameters (scaled by 100 internally).

    Returns:
        (B, T, dof) velocity field with zeros on gripper dims.
    """

    def __init__(self, seq_len, dof, param_len, gripper_idx=None):
        super().__init__()
        self.seq_len = seq_len
        self.dof = dof
        self.param_len = param_len

        if gripper_idx is None:
            gripper_idx = []
        if isinstance(gripper_idx, list):
            gripper_idx = torch.tensor(gripper_idx, dtype=torch.long)
        arm_idx = torch.tensor([i for i in range(dof) if i not in set(gripper_idx.tolist())], dtype=torch.long)

        self.register_buffer("gripper_idx", gripper_idx, persistent=False)
        self.register_buffer("arm_idx", arm_idx, persistent=False)

        loss_mask = torch.ones(dof, dtype=torch.float32)
        if gripper_idx.numel() > 0:
            loss_mask[gripper_idx] = 0.0
        self.register_buffer("loss_mask", loss_mask.view(1, 1, dof), persistent=False)

        # Time embedding for FiLM conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Condition dimension = param_len + time embedding dim
        condition_dim = param_len + 32

        # 1D U-Net for sequence modeling
        self.unet = UNet1D(dof=len(arm_idx), condition_dim=condition_dim)

    def forward(self, x, t, env_params):
        # x: (B, seq_len, dof), t: (B, 1), env_params: (B, param_len)
        dev, dt = x.device, x.dtype
        if torch.is_tensor(t):
            t = t.to(dev)
            if torch.is_floating_point(t):
                t = t.to(dt)
            else:
                t = torch.as_tensor(t, device=dev, dtype=dt)

        if torch.is_tensor(env_params):
            env_params = env_params.to(dev)
            if torch.is_floating_point(env_params):
                env_params = env_params.to(dt)
        else:
            env_params = torch.as_tensor(env_params, device=dev, dtype=dt)

        B, seq_len, dof = x.shape
        assert seq_len == self.seq_len and dof == self.dof, "Shape mismatch"

        # Normalize environment parameters (optional preprocessing)
        env_params_norm = env_params * 100

        # Encode timestep
        t_embed = self.time_embed(t)

        # Create conditioning vector
        condition = torch.cat([env_params_norm, t_embed], dim=-1)  # (B, cond_dim)

        # ------ select arm channels only ------
        # (B, seq_len, n_arm) -> (B, n_arm, seq_len) for Conv1d
        x_arm = x[:, :, self.arm_idx].permute(0, 2, 1).contiguous()

        # ------ run the U-Net on arm channels ------
        v_arm = self.unet(x_arm, condition)        # (B, n_arm, seq_len)
        v_arm = v_arm.permute(0, 2, 1).contiguous() # -> (B, seq_len, n_arm)

        # ------ stitch back to full DOF with zeros in grippers ------
        v_full = x.new_zeros(B, seq_len, dof)
        v_full[:, :, self.arm_idx] = v_arm.to(v_full.dtype)
        # gripper dims remain zero -> "no flow" for grippers

        return v_full
    

class MixtureSampler:
    """Low-rank conditional Gaussian mixture over (x, c).

    Each component places a low-rank Gaussian on the flattened trajectory `x`
    via a basis `B` and a full Gaussian on environment parameters `c`. The
    conditional p(x|c) is efficient to sample using precomputed Choleskies.

    Construction inputs typically come from `compute_cluster_pca_fast_joint`.

    Args:
        mu_x: (K, Dx) component means for flattened trajectories.
        mu_c: (K, Dc) component means for environment parameters.
        B: (K, Dx, d) component bases, with `d` < `Dx`.
        Sig_zz: (K, d, d) covariance of latent coordinates z.
        Sig_zc: (K, d, Dc) cross-covariance between z and c.
        Sig_cc: (K, Dc, Dc) covariance of c.
        weights: (K,) mixture weights (will be normalized).
        device: Torch device for internal tensors and sampling.
        reg: Diagonal regularizer added to covariances for stability.
        orth_sigma: Optional isotropic noise added in x-space (orthogonal to B).

    Methods:
        sample_cond(c, ...): Sample x ~ p(x|c) optionally forcing components.
        sample_joint(M, ...): Sample (x, c) ~ p(x,c) by drawing component first.
    """
    def __init__(self, mu_x, mu_c, B, Sig_zz, Sig_zc, Sig_cc, weights,
                 device='cpu', reg=1e-6, orth_sigma=0.0):
        self.device = device
        # shapes
        self.K, self.Dx = mu_x.shape
        self.Dc = mu_c.shape[1]
        self.d  = B.shape[2]

        # tensors
        self.mu_x = torch.as_tensor(mu_x, dtype=torch.float32, device=device)   # (K, Dx)
        self.mu_c = torch.as_tensor(mu_c, dtype=torch.float32, device=device)   # (K, Dc)
        self.B    = torch.as_tensor(B,    dtype=torch.float32, device=device)   # (K, Dx, d)
        self.Szz  = torch.as_tensor(Sig_zz, dtype=torch.float32, device=device) # (K, d, d)
        self.Szc  = torch.as_tensor(Sig_zc, dtype=torch.float32, device=device) # (K, d, Dc)
        self.Scc  = torch.as_tensor(Sig_cc, dtype=torch.float32, device=device) # (K, Dc, Dc)

        w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.weights = (w / w.sum()).clamp_min(1e-12)                            # (K,)

        self.reg = float(reg)
        self.orth_sigma = float(orth_sigma)

        # Choleskies and inverses used in conditionals
        self.Lcc   = []   # cholesky(Σ_cc)
        self.invcc = []   # Σ_cc^{-1}
        self.logdet_cc = []
        self.Lzgc  = []   # cholesky(Σ_{z|c})

        for k in range(self.K):
            Scc_k = self.Scc[k] + self.reg * torch.eye(self.Dc, device=device)
            Lcc_k = torch.linalg.cholesky(Scc_k)
            self.Lcc.append(Lcc_k)
            invcc_k = torch.cholesky_inverse(Lcc_k)
            self.invcc.append(invcc_k)
            self.logdet_cc.append(2.0 * torch.log(torch.diag(Lcc_k)).sum())

            # Σ_{z|c} = Σ_zz - Σ_zc Σ_cc^{-1} Σ_cz
            Szgc_k = self.Szz[k] - self.Szc[k] @ invcc_k @ self.Szc[k].transpose(0,1)
            # stabilize
            Szgc_k = 0.5 * (Szgc_k + Szgc_k.transpose(0,1)) + self.reg * torch.eye(Szgc_k.shape[0], device=self.device)
            Lzgc_k = torch.linalg.cholesky(Szgc_k)
            self.Lzgc.append(Lzgc_k)

        # stack for vectorization convenience
        self.Lcc      = torch.stack(self.Lcc, dim=0)         # (K, Dc, Dc)
        self.invcc    = torch.stack(self.invcc, dim=0)       # (K, Dc, Dc)
        self.logdet_cc = torch.stack(self.logdet_cc, dim=0)  # (K,)
        self.Lzgc     = torch.stack(self.Lzgc, dim=0)        # (K, d, d)

    # ---- small helper: x ~ p_k(x | c) for a fixed component k ----
    def _x_given_c_fixed_k(self, k, c, truncated=False, trunc=(-1.5, 1.5)):
        # c: (n, Dc)
        delta   = c - self.mu_c[k].unsqueeze(0)                      # (n, Dc)
        # μ_{z|c} = Σ_zc Σ_cc^{-1} (c - μ_c)
        mu_zc   = delta @ (self.invcc[k].T @ self.Szc[k].T)          # (n, d)
        if truncated:
            lo, hi = trunc
            z_eps = torch.from_numpy(
                truncnorm.rvs(lo, hi, size=(c.shape[0], self.d)).astype(np.float32)
            ).to(self.device)
        else:
            z_eps = torch.randn(c.shape[0], self.d, device=self.device)
        z = mu_zc + z_eps @ self.Lzgc[k].T                           # (n, d)
        x = self.mu_x[k].unsqueeze(0) + z @ self.B[k].T              # (n, Dx)
        if self.orth_sigma > 0.0:
            x = x + torch.randn_like(x) * self.orth_sigma
        return x

    @torch.no_grad()
    def sample_cond(self, c_in, deterministic_component=False,
                    truncated=False, trunc=(-1.5,1.5), pis=None):
        """Sample trajectories x ~ p(x|c).

        If `pis` is given (component indices), sampling is performed per fixed
        component without computing responsibilities. Otherwise, responsibilities
        ω_k(c) ∝ π_k N(c; μ_c^k, Σ_cc^k) are computed and components are drawn.

        Args:
            c_in: (B, Dc) environment parameters.
            deterministic_component: If True, pick argmax component per item.
            truncated: If True, use truncated normal for latent sampling.
            trunc: (low, high) bounds in std units for latent truncation.
            pis: Optional (B,) long tensor of fixed component ids.

        Returns:
            x_out: (B, Dx) flattened trajectory samples.
            pis: (B,) component indices used.
            w: (B, K) responsibility weights or None when `pis` is forced.
        """
        c = torch.as_tensor(c_in, dtype=torch.float32, device=self.device)
        Bsz = c.shape[0]

        # Fast path: fixed components -> no responsibilities
        if pis is not None:
            pis = torch.as_tensor(pis, device=self.device, dtype=torch.long)
            x_out = torch.empty(Bsz, self.Dx, device=self.device)
            # process by unique component to reduce python overhead
            for k in pis.unique(sorted=True).tolist():
                mask = (pis == k)
                if mask.any():
                    x_out[mask] = self._x_given_c_fixed_k(
                        k, c[mask], truncated=truncated, trunc=trunc
                    )
            return x_out, pis, None  # no weights when forced

        # Otherwise compute responsibilities ω_k(c) ∝ π_k N(c; μ_c^k, Σ_cc^k)
        logp = []
        for k in range(self.K):
            delta = c - self.mu_c[k].unsqueeze(0)                      # (B, Dc)
            y     = torch.cholesky_solve(delta.T, self.Lcc[k])         # (Dc, B)
            quad  = (delta.T * y).sum(dim=0)                           # (B,)
            lp    = -0.5 * (quad + self.logdet_cc[k] + self.Dc * np.log(2*np.pi))
            logp.append(lp.unsqueeze(1))
        logp = torch.cat(logp, dim=1)                                  # (B, K)
        logw = logp + torch.log(self.weights).unsqueeze(0)             # (B, K)
        logw = logw - logw.logsumexp(dim=1, keepdim=True)
        w    = torch.exp(logw)                                         # (B, K)

        if deterministic_component:
            pis = torch.argmax(w, dim=1)
        else:
            pis = torch.multinomial(w, num_samples=1).squeeze(1)       # (B,)

        # draw x for each fixed component
        x_out = torch.empty(Bsz, self.Dx, device=self.device)
        for k in pis.unique(sorted=True).tolist():
            mask = (pis == k)
            if mask.any():
                x_out[mask] = self._x_given_c_fixed_k(
                    k, c[mask], truncated=truncated, trunc=trunc
                )
        return x_out, pis, w

    @torch.no_grad()
    def sample_joint(self, M, truncated=False, trunc=(-1.5, 1.5), pis=None):
        """Sample (x, c) pairs jointly from the mixture.

        Components are drawn first (unless `pis` is provided), then `c` is sampled
        from N(μ_c, Σ_cc) followed by x|c using the fixed-component sampler.

        Args:
            M: Number of samples to generate.
            truncated: If True, use truncated normal for latent sampling.
            trunc: (low, high) bounds in std units for latent truncation.
            pis: Optional (M,) fixed component indices.

        Returns:
            x: (M, Dx) flattened trajectories.
            c: (M, Dc) environment parameters.
            pis: (M,) component indices used.
        """
        # choose components if not given
        if pis is None:
            pis = torch.multinomial(self.weights, M, replacement=True).to(self.device)
        else:
            pis = torch.as_tensor(pis, device=self.device, dtype=torch.long)

        # sample c per chosen component
        c = torch.empty(M, self.Dc, device=self.device)
        for k in pis.unique(sorted=True).tolist():
            mask = (pis == k)
            if not mask.any():
                continue
            nk  = int(mask.sum().item())
            eps = torch.randn(nk, self.Dc, device=self.device)
            c_k = self.mu_c[k].unsqueeze(0) + eps @ self.Lcc[k].T
            c[mask] = c_k

        # sample x given c
        x = torch.empty(M, self.Dx, device=self.device)
        for k in pis.unique(sorted=True).tolist():
            mask = (pis == k)
            if mask.any():
                x[mask] = self._x_given_c_fixed_k(k, c[mask], truncated=truncated, trunc=trunc)
        return x, c, pis


@torch.no_grad()
def run_flow(model, x, c, device, n_steps=100):
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t  = torch.full((x.shape[0], 1), i*dt, device=device)
        v1 = model(x, t, c)
        x_mid = x + v1 * dt
        v2 = model(x_mid, t + dt, c)
        x  = x + 0.5 * (v1 + v2) * dt  # Heun (RK2)
    return x


def _get_environment_params(
    env,
    task_name: str,
):
    """Extract environment parameters used for conditioning.

    The function returns a 3‑tuple tailored to each task:
      - door:  (handle_x, handle_y, handle_yaw)
      - wipe:  (center_x, center_y, max_radius)
      - two_arm: ((x_L+x_R)/2, (y_L+y_R)/2, yaw_L)
      - nut:   (nut_x, nut_y, nut_yaw)

    Args:
        env: A RoboSuite environment instance.
        task_name: One of {"door", "wipe", "two_arm", "nut"}.

    Returns:
        Tuple[float, float, float]: Environment summary for conditioning. (all conditions have length of 3)

    Raises:
        AssertionError: If `task_name` is unsupported.
    """
    assert (task_name in {"door", "wipe", "two_arm", "nut"}), f"Unsupported task for {task_name}"
    if task_name == "door":
        handle_id = env.door_handle_site_id
        handle_pos = env._handle_xpos.copy()
        R_handle   = env.sim.data.site_xmat[handle_id].reshape(3,3)
        yaw = np.arctan2(R_handle[1,0], R_handle[0,0])
        environment_parameters = (handle_pos[0], handle_pos[1], yaw)
    elif task_name == "wipe":
        max_radius, center, _ = env._get_wipe_information()
        environment_parameters = (center[0], center[1], max_radius)
    elif task_name == "two_arm":
        handle_names = [n for n in env.sim.model.site_names if "handle" in n]
        handle_ids   = [env.sim.model.site_name2id(n) for n in handle_names]
        hidL, hidR = handle_ids[:2]
        posL = env.sim.data.site_xpos[hidL].copy()
        posR = env.sim.data.site_xpos[hidR].copy()
        R0   = env.sim.data.site_xmat[hidL].reshape(3,3)
        yaw  = np.arctan2(R0[1,0], R0[0,0])
        environment_parameters = ((posL[0]+posR[0])/2, (posL[1]+posR[1])/2, yaw)
    else:
        nut_handle_id = env.object_site_ids[0]
        nut_pos = env.sim.data.site_xpos[nut_handle_id].copy()
        nut_pos[2] = getattr(env, "table_offset", np.zeros(3))[2] # since the pegs drop from midair
        R0 = env.sim.data.site_xmat[nut_handle_id].reshape(3,3)
        yaw = np.arctan2(R0[1,0], R0[0,0])
        environment_parameters = (nut_pos[0], nut_pos[1], yaw)

    return environment_parameters


def _set_robot_qpos(env, q_first: np.ndarray, task_name: str):
    """Teleport the robot to the first pose and synchronize actuators.

    This writes joint positions (and zeros velocities) for each robot body
    part driven by the task, then attempts to align actuator control targets
    with the new state to avoid an initial snap when stepping the environment.

    Args:
        env: RoboSuite environment (must expose `sim.model`, `sim.data`).
        q_first: (D,) first joint vector of the planned trajectory.
        task_name: Task key to determine how many joints belong to each robot.

    Notes:
        - Calls `env.sim.forward()` at the end to refresh kinematics.
        - If actuator mapping is unavailable, control sync is skipped.
    """
    model = env.sim.model
    data  = env.sim.data

    offset = 0
    written_joint_ids = []   # joints we modify (for actuator sync)

    for ridx, robot in enumerate(env.robots):
        arm_names  = robot.robot_model.joints                    # 7 hinge joints
        grip_names = next(iter(robot.gripper.values())).joints   # 2 hinge joints (if present)

        # Decide which entries are available in q_first for this robot
        remaining = q_first.shape[0] - offset
        if remaining >= len(arm_names) + len(grip_names):
            names = arm_names + grip_names         # door/nut or two_arm
        elif remaining >= len(arm_names):
            names = arm_names                      # wipe (no finger values in q_first)
        else:
            raise ValueError(
                f"q_first too short: got {q_first.shape[0]} dims, "
                f"offset {offset}, robot {ridx} needs ≥{len(arm_names)}"
            )

        n = len(names)

        # Write positions (hinge/slide ⇒ 1-dim per joint) and zero velocities
        qpos_addrs = []
        dof_addrs  = []
        for nm in names:
            j_id = model.joint_name2id(nm)
            written_joint_ids.append(j_id)

            qpos_adr = model.jnt_qposadr[j_id]    # index into qpos
            dof_adr  = model.jnt_dofadr[j_id]     # index into qvel (hinge ⇒ +1 DOF)
            qpos_addrs.append(qpos_adr)
            dof_addrs.append(dof_adr)

        data.qpos[qpos_addrs] = q_first[offset:offset + n]
        data.qvel[dof_addrs]  = 0.0

        offset += n

    # Zero the rest to be conservative
    data.qacc[:] = 0.0
    # Align actuator targets for actuators that drive the joints we just set.
    # (Most robosuite joint-position controllers end up as JOINT actuators.)
    try:
        # 0 == JOINT transmission in MuJoCo
        trn_is_joint = (model.actuator_trntype == 0)
        for a in range(model.nu):
            if not trn_is_joint[a]:
                continue
            j_id = int(model.actuator_trnid[a, 0])
            if j_id in written_joint_ids:
                qpos_adr = model.jnt_qposadr[j_id]
                # position actuators: set target == current qpos
                data.ctrl[a] = float(data.qpos[qpos_adr])
    except Exception:
        # If anything about actuator mapping is unavailable, just skip syncing ctrl.
        pass

    # Recompute kinematics after the teleportion + ctrl sync
    env.sim.forward()


def _to_action_from_q(q, task_name):
    """Pack a joint vector into the action format expected by the controller.

    The packing mirrors existing controller wrappers for each task. In particular,
    gripper pairs are averaged into a single scalar control per gripper.

    Args:
        q: (D,) joint configuration for the whole system.
        task_name: One of {"door", "nut", "wipe", "two_arm"}.

    Returns:
        np.ndarray: Action vector compatible with `env.step(action)` for the task.
    """
    if task_name in ["door", "nut"]:
        arm_q   = q[:7]
        grip_sc = float((q[7] + q[8]) / 2.0)   # your current compression
        return np.concatenate([arm_q, [grip_sc]])
    elif task_name == "wipe":
        return q[:7]
    else:  # two_arm
        arm1 = q[0:7]
        grip1 = float((q[7] + q[8]) / 2.0)
        arm2 = q[9:16]
        grip2 = float((q[16] + q[17]) / 2.0)
        return np.concatenate([arm1, [grip1], arm2, [grip2]])


def _rollout_batch(
    task_name: str,
    q_low_batch: np.ndarray,          # (m, T, D) float32 on CPU
    env_settings: List[dict],         # length m
) -> Tuple[int, float, list, list]:
    """Worker: restore envs from settings, upsample to controller rate, roll out CPU-only."""
    successes, reward_sum = 0, 0.0
    success_info, fail_info = [], []

    m = len(env_settings)
    assert m == len(q_low_batch), "settings and q_low_batch length mismatch"

    for i in range(m):
        setting = env_settings[i]
        env = make_env(task_name,
                       use_joint_control=True,
                       environment_setting=setting)
        # plan at low rate -> smooth to control rate
        q_low  = q_low_batch[i]
        q_high = compute_smooth_trajectory(env, task_name, q_low, env.control_freq)

        # align sim to first pose and brief hold
        q0 = q_high[0].copy()
        _set_robot_qpos(env, q0, task_name)
        a0 = _to_action_from_q(q0, task_name)
        for _ in range(100):
            env.step(a0)

        for q in q_high[1:]:
            env.step(_to_action_from_q(q, task_name))

        if env._check_success():
            successes += 1
            success_info.append({"traj": q_high, "setting": setting})
        else:
            fail_info.append({"traj": q_high, "setting": setting})

        reward_sum += env.reward()
        env.close()

    return successes, reward_sum, success_info, fail_info


def eval_model(
    model,
    model_class,                 # kept for signature compatibility (not used here)
    task_name: str,
    seq_len: int,
    dof: int,
    param_len: int,
    gripper_idx,
    val_params,
    env_settings_all,
    device: str = "cuda",
    render_dir: str = " ",
    video_name: str = None,
    trials: int = 100,
    num_workers: int = 10,
    render_width: int = 0,
    render_num: int = 0,
    base_seed: int = 123,
    gpu_chunk_size: int | None = None,  # NEW: set to chunk the single GPU forward if memory-bound
) -> Tuple[float, float]:
    """
    New evaluation pipeline:
      1) Parent makes envs sequentially, saves settings/params, then closes.
      2) Parent runs ONE GPU forward (optionally chunked) to get all q_low.
      3) Workers restore envs, upsample, and roll out on CPU (no GPU in workers).
    """
    rng = np.random.RandomState(base_seed)
    torch.manual_seed(base_seed)
    random.seed(base_seed)

    # ---- (2) One GPU forward (optionally chunked) to produce all q_low ----
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        device = "cpu"

    model.eval()

    # pre-allocate to collect all q_low on CPU
    q_low_all = np.empty((trials, seq_len, dof), dtype=np.float32)

    if gpu_chunk_size is None or gpu_chunk_size <= 0:
        # single shot (ensure it fits!)
        x0 = torch.from_numpy(rng.randn(trials, seq_len, dof).astype(np.float32)).to(device)
        c  = torch.from_numpy(val_params).to(device)
        if use_cuda:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                q_low = run_flow(model, x0, c, device)          # (N, T, D) on CUDA
            q_low_all[:] = q_low.float().cpu().numpy()
            del q_low; torch.cuda.empty_cache()
        else:
            with torch.inference_mode():
                q_low = run_flow(model, x0, c, device)          # CPU path
            q_low_all[:] = q_low.cpu().numpy()
    else:
        # chunked single pass to cap VRAM
        N = trials
        for s in range(0, N, gpu_chunk_size):
            e = min(N, s + gpu_chunk_size)
            bs = e - s
            x0 = torch.from_numpy(rng.randn(bs, seq_len, dof).astype(np.float32)).to(device)
            c  = torch.from_numpy(val_params[s:e]).to(device)
            if use_cuda:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    q_low = run_flow(model, x0, c, device)
                q_low_all[s:e] = q_low.float().cpu().numpy()
                del q_low; torch.cuda.empty_cache()
            else:
                with torch.inference_mode():
                    q_low = run_flow(model, x0, c, device)
                q_low_all[s:e] = q_low.cpu().numpy()

    # ---- (3) Roll out on CPU with multiple workers ----
    # even split of trials across workers
    base, rem = divmod(trials, max(1, num_workers))
    splits: List[tuple[int, int]] = []
    off = 0
    for i in range(num_workers):
        n = base + (1 if i < rem else 0)
        if n > 0:
            splits.append((off, off + n))
            off += n

    total_success = 0
    total_reward  = 0.0
    success_info: List[dict] = []
    failure_info: List[dict] = []
    s_count = 0
    f_count = 0

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futs = []
        for (s, e) in splits:
            futs.append(ex.submit(
                _rollout_batch,
                task_name,
                q_low_all[s:e],                 # numpy slice (copies view to child)
                env_settings_all[s:e],
            ))
        for fut in futs:
            succ, rew, info_s, info_f = fut.result()
            total_success += succ
            total_reward  += rew

            if s_count < render_num:
                take = min(render_num - s_count, len(info_s))
                success_info += info_s[:take]; s_count += take
            if f_count < (render_width*render_width - render_num):
                take = min(render_width*render_width - render_num - f_count, len(info_f))
                failure_info += info_f[:take]; f_count += take

    # ---- (optional) render sample grid (unchanged) ----
    episode_frames = []
    s_left = min(s_count, render_num)
    f_left = min(f_count, render_width*render_width - render_num)

    while s_left > 0:
        env_r = make_env(task_name,
                         has_offscreen_renderer=True,
                         use_camera_obs=False,
                         use_joint_control=True,
                         environment_setting=success_info[s_left - 1]["setting"])
        frames = render_trajectory(env_r,
                                   task_name,
                                   success_info[s_left - 1]["traj"],
                                   success_info[s_left - 1]["traj"][0, :],
                                   camera_name="frontview",
                                   hold_init=True)
        episode_frames.append(frames)
        env_r.close()
        s_left -= 1

    while f_left > 0:
        env_r = make_env(task_name,
                         has_offscreen_renderer=True,
                         use_camera_obs=False,
                         use_joint_control=True,
                         environment_setting=failure_info[f_left - 1]["setting"])
        frames = render_trajectory(env_r,
                                   task_name,
                                   failure_info[f_left - 1]["traj"],
                                   failure_info[f_left - 1]["traj"][0, :],
                                   camera_name="frontview",
                                   hold_init=True)
        episode_frames.append(frames)
        env_r.close()
        f_left -= 1

    if render_num:
        grid_path = os.path.join(render_dir, f"{task_name}_grid_{video_name}.mp4")
        write_grid_video(episode_frames, grid_path,
                         grid_shape=(render_width, render_width))

    success_rate = total_success / trials
    mean_reward  = total_reward  / trials
    return success_rate, mean_reward


def _standardize_cols(A, eps=1e-8):
    """Z-score standardize each column of `A`.

    Args:
        A: (N, D) array.
        eps: Small floor for std to avoid division by zero.

    Returns:
        A_std: Standardized array with zero mean / unit variance per column.
        stats: Tuple (mean[1,D], std[1,D]) used for the transform.
    """
    mu = A.mean(axis=0, keepdims=True)
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
        print(f"Warning! {len(missing)} points are not covered by clusters, adding them to the first cluster")
        missing = [j for j in range(N) if j not in covered_all]
        for j in missing:
            merged_clusters[0].append(j)

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


def train_uniform_FM(
    model,
    optimizer,
    scheduler,
    task_name,
    target_trajectories,
    environment_parameters,
    seq_len,
    dof,
    param_len,
    gripper_idx,
    n_t,
    max_epochs,
    batch_size,
    device,
    val_period=5,
    early_stopping=True,
    stop_criteria=3,
    val_trials=100,
):
    """Train vanilla Flow Matching with uniform `t`.

    The data term draws (x0, x1) pairs where x0~N(0, I) and x1 is a ground-truth
    trajectory from the dataset. The network is trained to match v* = x1 - x0 at
    uniformly sampled times.

    Args:
        model: Vector field model to train.
        optimizer: Torch optimizer.
        scheduler: Optional LR scheduler (callable `.step()` per batch); may be None.
        task_name: Task key for periodic evaluation.
        target_trajectories: (N, T, D) ground truth joint trajectories.
        environment_parameters: (N, P) conditioning parameters.
        seq_len: T; time steps per trajectory.
        dof: D; joint dimensionality.
        param_len: P; conditioning dim.
        gripper_idx: Indices of gripper joints ignored by the loss (if any).
        n_t: Number of `t` samples per data pair (multiplies batch size).
        max_epochs: Training epochs.
        batch_size: Batch size (on x1 instances).
        device: Torch device string.
        val_period: Evaluate every `val_period` epochs.
        early_stopping: Whether to stop after `stop_criteria` non-improvements.
        stop_criteria: Number of consecutive validations allowed without improvement.

    Returns:
        best_model: Deep-copied best performing model (may be None if never improved).
        last_model: Model at the end of training / interruption.
        success_rate_recs: Dict[epoch → metrics] logged at validation epochs.
    """
    N = target_trajectories.shape[0]
    best_avg_reward = 0.0
    best_success_rate = 0.0
    best_model = None
    success_rate_recs = {}
    stop_count = 0

    # ---- Build environments sequentially for validation and close them ----
    env_params_list: List[np.ndarray] = []
    env_settings_all: List[dict] = []
    for _ in range(val_trials):
        env = make_env(task_name, use_joint_control=True)
        env.reset()

        env_settings_all.append({
            "qpos":      env.sim.data.qpos.copy(),
            "qvel":      env.sim.data.qvel.copy(),
            "body_pos":  env.sim.model.body_pos.copy(),
            "body_quat": env.sim.model.body_quat.copy(),
        })
        env_params_list.append(_get_environment_params(env, task_name))
        env.close()
    val_params = np.asarray(env_params_list, dtype=np.float32)         # (N, Dc)

    try: 
        model = model.to(device)
        target_trajectories = target_trajectories.to(device)
        environment_parameters = environment_parameters.to(device)
        for epoch in tqdm(range(1, max_epochs + 1),
                        desc="UniformFM Training",
                        unit="epoch"):
            model.train()

            perm_t = torch.randperm(N, device=device)
            loss_sum = 0
            for i in range(0, N, batch_size):
                idx = perm_t[i:min(i+batch_size, N)]
                x1 = target_trajectories[idx]
                x0 = torch.randn(len(idx), seq_len, dof, device=device)
                t = torch.rand(len(idx)*n_t, device=device).unsqueeze(-1) # uniform t sampling
                env_params = environment_parameters[idx, :]

                x1r = x1.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, seq_len, dof)
                x0r = x0.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, seq_len, dof)
                t_col = t.view(-1, 1, 1)
                xt = (1 - t_col) * x0r + t_col * x1r
                env_params_r = env_params.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, param_len)

                target_v = x1r - x0r
                with torch.enable_grad():
                    pred_v   = model(xt,t, env_params_r)
                    sq_err = (pred_v - target_v) ** 2
                    # If the model registered a (1,1,dof) mask that zeros gripper dims, use it:
                    if hasattr(model, "loss_mask") and model.loss_mask is not None:
                        sq_err = sq_err * model.loss_mask
                    loss = sq_err.mean()

                optimizer.zero_grad()
                loss.backward()
                loss_sum += loss.item()
                optimizer.step()
                
            scheduler.step()

            if epoch % val_period == 0:
                success_rate, avg_reward = eval_model(model, VectorField, task_name, seq_len, dof, param_len, gripper_idx, val_params, env_settings_all, device)                
                success_rate_recs[epoch] = {"success_rate": success_rate, "avg_reward": avg_reward, "loss": loss_sum}
                if success_rate < best_success_rate:
                    tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
                    if early_stopping:
                        if stop_count == stop_criteria:
                            tqdm.write("Early stopping triggered.")
                            break
                        else:
                            stop_count += 1
                else:
                    if (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = copy.deepcopy(model)
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
    except KeyboardInterrupt:
        tqdm.write("Training interrupted by user. Returning best model so far...")

    return best_model, model, success_rate_recs


def train_shifted_FM(
    model,
    optimizer,
    scheduler,
    task_name,
    target_trajectories,
    environment_parameters,
    seq_len,
    dof,
    param_len,
    gripper_idx,
    n_t,
    max_epochs,
    batch_size,
    device,
    val_period=5,
    early_stopping=True,
    stop_criteria=3,
    beta_a = 1.5,
    beta_b = 1,
    val_trials =100,
):
    """Train Flow Matching with Beta-biased time sampling ("Shifted FM").

    Draw time `t` from Beta(beta_a, beta_b), then reflect to emphasize later
    parts of the trajectory (`t <- 1 - t`). Useful when later stages carry more
    task-relevant signal.

    Args are the same as `train_uniform_FM` with two additional parameters:
        beta_a: Alpha parameter of Beta distribution.
        beta_b: Beta parameter of Beta distribution.

    Returns:
        best_model, last_model, success_rate_recs (same semantics as Uniform FM).
    """
    N = target_trajectories.shape[0]
    best_avg_reward = 0.0
    best_success_rate = 0.0
    best_model = None
    success_rate_recs = {}
    stop_count = 0

    # ---- Build environments sequentially for validation and close them ----
    env_params_list: List[np.ndarray] = []
    env_settings_all: List[dict] = []
    for _ in range(val_trials):
        env = make_env(task_name, use_joint_control=True)
        env.reset()

        env_settings_all.append({
            "qpos":      env.sim.data.qpos.copy(),
            "qvel":      env.sim.data.qvel.copy(),
            "body_pos":  env.sim.model.body_pos.copy(),
            "body_quat": env.sim.model.body_quat.copy(),
        })
        env_params_list.append(_get_environment_params(env, task_name))
        env.close()
    val_params = np.asarray(env_params_list, dtype=np.float32)         # (N, Dc)

    try: 
        model = model.to(device)
        target_trajectories = target_trajectories.to(device)
        environment_parameters = environment_parameters.to(device)
        for epoch in tqdm(range(1, max_epochs + 1),
                        desc="ShiftedFM Training",
                        unit="epoch"):
            model.train()

            perm_t = torch.randperm(N, device=device)
            loss_sum = 0

            for i in range(0, N, batch_size):
                idx = perm_t[i:min(i+batch_size, N)]
                x1 = target_trajectories[idx]
                x0 = torch.randn(len(idx), seq_len, dof, device=device)
                t  = Beta(beta_a, beta_b).sample((len(idx)*n_t, 1)).to(device)  # Beta t sampling
                t  = torch.ones_like(t).to(device) - t
                env_params = environment_parameters[idx, :]

                x1r = x1.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, seq_len, dof)
                x0r = x0.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, seq_len, dof)
                t_col = t.view(-1, 1, 1)
                xt = (1 - t_col) * x0r + t_col * x1r
                env_params_r = env_params.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, param_len)

                target_v = x1r - x0r

                with torch.enable_grad():
                    pred_v   = model(xt,t, env_params_r)
                    sq_err = (pred_v - target_v) ** 2
                    # If the model registered a (1,1,dof) mask that zeros gripper dims, use it:
                    if hasattr(model, "loss_mask") and model.loss_mask is not None:
                        sq_err = sq_err * model.loss_mask
                    loss = sq_err.mean()

                optimizer.zero_grad()
                loss.backward()
                loss_sum += loss.item()
                optimizer.step()
                
            scheduler.step()

            if epoch % val_period == 0:
                success_rate, avg_reward = eval_model(model, VectorField, task_name, seq_len, dof, param_len, gripper_idx, val_params, env_settings_all, device)
                
                if not torch.is_grad_enabled():
                    torch.set_grad_enabled(True)
                
                success_rate_recs[epoch] = {"success_rate": success_rate, "avg_reward": avg_reward, "loss": loss_sum}
                if success_rate < best_success_rate:
                    tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
                    if early_stopping:
                        if stop_count == stop_criteria:
                            tqdm.write("Early stopping triggered.")
                            break
                        else:
                            stop_count += 1
                else:
                    if (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = copy.deepcopy(model)
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
    except KeyboardInterrupt:
        tqdm.write("Training interrupted by user. Returning best model so far...")

    return best_model, model, success_rate_recs


def train_DGFM(
    model,
    optimizer,
    scheduler,
    task_name,
    target_trajectories,
    environment_parameters,
    seq_len,
    dof,
    param_len,
    gripper_idx,
    mf,                   # global multiplier (how many synthetic global batches)
    n_t_local,
    n_t_global,
    cluster_size,
    cluster_d,
    max_epochs,
    batch_size,
    device,
    val_period=5,
    early_stopping=True,
    stop_criteria=3,
    scale_x=1.0,
    scale_c=1.0,
    val_trials=100,
):
    """Train Dimension-Guided Flow Matching (DGFM, conditional).

    DGFM builds a conditional mixture over (x, c) by clustering joint features
    and computing per-cluster low-rank statistics on trajectories. Training then
    alternates two phases per epoch:

      1) Global FM (t ∈ [0, 0.5]):
         - Draw synthetic pairs (x0 ~ N, (x1, c1) ~ mixture joint).
         - Train at time 0.5*t with target scaled by 2.

      2) Local FM (t ∈ [0.5, 1]):
         - For each dataset c_true, draw x_tilde ~ mixture p(x|c_true) near the mode
           (optionally choosing the component using the data point's cluster).
         - Train at 0.5*t+0.5 with target scaled by 2 to focus on the refinement.

    Args:
        model, optimizer, scheduler: Standard training components.
        task_name: Task key for evaluation.
        target_trajectories: (N, T, D) dataset.
        environment_parameters: (N, P) dataset conditioning.
        seq_len, dof, param_len: T, D, P.
        gripper_idx: Optional indices ignored by loss.
        mf: Global synthetic data multiplier (global_N = mf * N).
        n_t_local: # of time samples per batch item in local phase.
        n_t_global: # of time samples per batch item in global phase.
        cluster_size: Local neighborhood size m for clustering.
        cluster_d: Target rank for per-cluster PCA basis on X.
        max_epochs, batch_size, device: Usual training hyperparameters.
        val_period, early_stopping, stop_criteria: Validation / ES settings.
        scale_x, scale_c: Feature scaling used during clustering.

    Returns:
        best_model: Best checkpoint by success rate / avg reward.
        last_model: Final model (or at interruption).
        success_rate_recs: Dict of validation metrics per epoch.
    """
    # -------- 0) Build mixture on (x, c) --------
    print("Clustering dataset . . .")
    X_np = target_trajectories.detach().cpu().numpy().reshape(target_trajectories.shape[0], -1)  # (N, Dx)
    C_np = environment_parameters.detach().cpu().numpy()                                         # (N, Dc)

    clusters, inv_cluster = cluster_points_joint(
        X_np, C_np, m=cluster_size, jaccard_thresh=0.8, merge_k=10,
        standardize=True, scale_x=scale_x, scale_c=scale_c
    )

    print(f"{len(clusters)} clusters made! Applying PCA . . .")
    mu_x, mu_c, B, Szz, Szc, Scc, weights = compute_cluster_pca_fast_joint(
        X_np, C_np, clusters, d_x=cluster_d, eps=1e-3, outlier_q=0.9,
        max_pca_samples=2000, n_jobs=-1
    )

    mixture_sampler = MixtureSampler(
        mu_x, mu_c, B, Szz, Szc, Scc, weights, device=device, reg=1e-6, orth_sigma=0.0
    )

    # -------- 1) Train loop (global + local, eval, early stop) --------
    N = target_trajectories.shape[0]
    global_N = mf * N
    best_avg_reward = 0.0
    best_success_rate = 0.0
    best_model = None
    success_rate_recs = {}
    stop_count = 0

    # ---- Build environments sequentially for validation and close them ----
    env_params_list: List[np.ndarray] = []
    env_settings_all: List[dict] = []
    for _ in range(val_trials):
        env = make_env(task_name, use_joint_control=True)
        env.reset()

        env_settings_all.append({
            "qpos":      env.sim.data.qpos.copy(),
            "qvel":      env.sim.data.qvel.copy(),
            "body_pos":  env.sim.model.body_pos.copy(),
            "body_quat": env.sim.model.body_quat.copy(),
        })
        env_params_list.append(_get_environment_params(env, task_name))
        env.close()
    val_params = np.asarray(env_params_list, dtype=np.float32)         # (N, Dc)

    try:
        model = model.to(device)
        target_trajectories = target_trajectories.to(device)          # (N, T, dof)
        environment_parameters = environment_parameters.to(device)    # (N, param_len)

        for epoch in tqdm(range(1, max_epochs + 1), desc=f"DGFM_mf{mf} Training", unit="epoch"):
            model.train()

            # ===== Global FM: t ∈ [0, 0.5] (scale factor 2) =====
            g_loss_sum = 0.0
            for i in range(0, global_N, batch_size):
                m = min(batch_size, global_N - i)

                # base noise
                x0 = torch.randn(m, seq_len, dof, device=device)

                # mixture joint sample (flattened x), reshape to (m, T, dof)
                x1_flat, c1, _ = mixture_sampler.sample_joint(m, truncated=True)
                x1 = x1_flat.reshape(m, seq_len, dof)

                # times in [0,1] -> we evaluate at 0.5 * t and scale target by 2
                t = torch.rand(m * n_t_global, device=device).unsqueeze(-1)  # (m*n_tg, 1)
                x0r = x0.unsqueeze(1).expand(-1, n_t_global, -1, -1).reshape(-1, seq_len, dof)
                x1r = x1.unsqueeze(1).expand(-1, n_t_global, -1, -1).reshape(-1, seq_len, dof)
                xt  = (1 - t.view(-1,1,1)) * x0r + t.view(-1,1,1) * x1r

                # broadcast env params from mixture
                env_r = c1.unsqueeze(1).expand(-1, n_t_global, -1).reshape(-1, param_len)
                target_v = 2.0 * (x1r - x0r)

                with torch.enable_grad():
                    pred_v   = model(xt, 0.5 * t, env_r)
                    sq_err   = (pred_v - target_v) ** 2
                    if hasattr(model, "loss_mask") and model.loss_mask is not None:
                        sq_err = sq_err * model.loss_mask
                    loss = sq_err.mean()
                    optimizer.zero_grad()
                    loss.backward()

                optimizer.step()
                g_loss_sum += float(loss.item())

            # ===== Local FM: t ∈ [0.5, 1.0] (scale factor 2) =====
            l_loss_sum = 0.0
            perm_t = torch.randperm(N, device=device)
            for i in range(0, N, batch_size):
                idx = perm_t[i:min(i + batch_size, N)]
                bsz = idx.shape[0]

                x_true = target_trajectories[idx]               # (bsz, T, dof)
                c_true = environment_parameters[idx, :]         # (bsz, P)

                # mixture conditional anchors: x̃ ~ p̃(x|c_true)  -> reshape to (bsz, T, dof)
                pis = torch.tensor(
                    [np.random.choice(inv_cluster[j]) for j in idx.detach().cpu().tolist()],
                    dtype=torch.long,
                    device=device,
                )
                x_tilde_flat, _, _ = mixture_sampler.sample_cond(c_true, truncated=True, pis=pis)
                x_tilde = x_tilde_flat.reshape(bsz, seq_len, dof)

                # times in [0,1] -> evaluate at 0.5 * t + 0.5 and scale target by 2
                t = torch.rand(bsz * n_t_local, device=device).unsqueeze(-1)      # (bsz*n_tl, 1)
                x0r = x_tilde.unsqueeze(1).expand(-1, n_t_local, -1, -1).reshape(-1, seq_len, dof)
                x1r = x_true.unsqueeze(1).expand(-1, n_t_local, -1, -1).reshape(-1, seq_len, dof)
                xt  = (1 - t.view(-1,1,1)) * x0r + t.view(-1,1,1) * x1r

                env_r = c_true.unsqueeze(1).expand(-1, n_t_local, -1).reshape(-1, param_len)

                target_v = 2.0 * (x1r - x0r)

                with torch.enable_grad():
                    pred_v   = model(xt, 0.5 * t + 0.5, env_r)
                    sq_err   = (pred_v - target_v) ** 2
                    if hasattr(model, "loss_mask") and model.loss_mask is not None:
                        sq_err = sq_err * model.loss_mask
                    loss = sq_err.mean()
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l_loss_sum += float(loss.item())

            if scheduler is not None:
                scheduler.step()

            # ===== Validation / early stopping =====
            if epoch % val_period == 0:
                model.eval()
                success_rate, avg_reward = eval_model(model, VectorField, task_name, seq_len, dof, param_len, gripper_idx, val_params, env_settings_all, device)
                
                if not torch.is_grad_enabled():
                    torch.set_grad_enabled(True)

                success_rate_recs[epoch] = {
                    "success_rate": success_rate,
                    "avg_reward":   avg_reward,
                    "g_loss":       g_loss_sum,
                    "l_loss":       l_loss_sum,
                }

                if success_rate < best_success_rate:
                    tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                               f"avg reward={avg_reward:.3f}, g_loss={g_loss_sum:.3f}, l_loss={l_loss_sum:.3f}")
                    if early_stopping:
                        if stop_count == stop_criteria:
                            tqdm.write("Early stopping triggered.")
                            break
                        else:
                            stop_count += 1
                else:
                    if (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = copy.deepcopy(model)
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                                   f"avg reward={avg_reward:.3f}, g_loss={g_loss_sum:.3f}, l_loss={l_loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, "
                                   f"avg reward={avg_reward:.3f}, g_loss={g_loss_sum:.3f}, l_loss={l_loss_sum:.3f}")

    except KeyboardInterrupt:
        tqdm.write("Training interrupted by user. Returning best model so far...")

    return best_model, model, success_rate_recs
