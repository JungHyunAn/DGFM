"""Dimension-guided FM utilities."""

import numpy as np
import torch
from scipy.stats import truncnorm


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


class DGFM:
    """DGFM trainer facade that inherits the common VanillaFM interface."""

    def __init__(self, *args, **kwargs):
        from Robot_simulation.FM_util import VanillaFM

        self._impl = VanillaFM(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def train(
        self,
        target_trajectories,
        conditions,
        *,
        mf: int,
        n_t_local: int,
        n_t_global: int,
        cluster_size: int,
        cluster_d: int,
        max_epochs: int,
        batch_size: int,
        val_period: int = 5,
        early_stopping: bool = True,
        stop_criteria: int = 3,
        scale_x: float = 1.0,
        scale_c: float = 1.0,
        val_trials: int = 25,
    ):
        from Robot_simulation.FM_util import train_DGFM

        return train_DGFM(
            model=self._impl.model,
            optimizer=self._impl.optimizer,
            scheduler=self._impl.scheduler,
            task_name=self._impl.task_name,
            target_trajectories=target_trajectories,
            environment_parameters=conditions,
            seq_len=self._impl.horizon,
            dof=self._impl.dof,
            param_len=self._impl.condition_dim,
            gripper_idx=self._impl.gripper_idx,
            mf=mf,
            n_t_local=n_t_local,
            n_t_global=n_t_global,
            cluster_size=cluster_size,
            cluster_d=cluster_d,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=self._impl.device,
            val_period=val_period,
            early_stopping=early_stopping,
            stop_criteria=stop_criteria,
            scale_x=scale_x,
            scale_c=scale_c,
            val_trials=val_trials,
        )
