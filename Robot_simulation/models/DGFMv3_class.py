"""DGFMv3 utilities with a joint X-C PCA intermediate and condition control path."""

import numpy as np
import torch
from tqdm import tqdm

from Robot_simulation.models.DGFM_class import (
    DGFM,
    MixtureSampler,
    cluster_points_joint,
    compute_cluster_pca_fast_joint,
)
from Robot_simulation.env_util import _generate_val_env, eval_model
from Robot_simulation.models.VanillaFM_class import VectorField, normalize_accumulated_gradients
from Robot_simulation.reproducibility import validation_result_is_better


class DGFMv3(DGFM):
    """DGFM trainer with a joint PCA intermediate and a moving condition path."""

    def _condition_path_weights(
        self,
        t,
        condition_interpolation_path="piecewise-linear-half",
    ):
        """Return weights for C_t = w_pca(t) * C_tilde + w_demo(t) * C_demo."""
        if condition_interpolation_path == "piecewise-linear-half":
            w_pca = torch.clamp(1.0 - 2.0 * t, min=0.0)
            w_demo = 1.0 - w_pca
            return w_pca, w_demo
        raise ValueError(
            "Unsupported DGFMv3 condition interpolation path "
            f"{condition_interpolation_path!r}"
        )

    @torch.no_grad()
    def _snapshot_conditions_for_pca(self, conditions, batch_size):
        """Build deterministic full policy conditions for joint PCA fitting."""
        vision_condition_fn = getattr(self, "vision_condition_fn", None)
        if vision_condition_fn is None:
            return conditions.detach()

        was_training = self.model.training
        self.model.eval()
        chunks = []
        try:
            for start in range(0, conditions.shape[0], batch_size):
                stop = min(start + batch_size, conditions.shape[0])
                source_idx = torch.arange(start, stop, device=self.device)
                chunks.append(
                    vision_condition_fn(source_idx, conditions[start:stop]).detach()
                )
        finally:
            if was_training:
                self.model.train()
        return torch.cat(chunks, dim=0)

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
        condition_interpolation_path="piecewise-linear-half",
        residual_lambda=0.2,
        dgfm_truncated=True,
        dgfm_trunc_low=-1.5,
        dgfm_trunc_high=1.5,
    ):
        idx = perm_t[:target_trajectories.shape[0]]
        x = target_trajectories[idx]
        demo_c = conditions[idx, :]
        m = x.shape[0]

        z = torch.randn(m, self.horizon, self.dof, device=self.device)
        pis = self._sample_covering_clusters(idx, inv_cluster, cluster_sizes)
        y_flat, tilde_c, _ = mixture_sampler.sample_joint(
            m,
            truncated=dgfm_truncated,
            trunc=(dgfm_trunc_low, dgfm_trunc_high),
            pis=pis,
        )
        y = y_flat.reshape(m, self.horizon, self.dof)

        t = self.sample_t(m * n_t)
        xr = x.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        yr = y.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        zr = z.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
        tilde_cr = tilde_c.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, tilde_c.shape[-1])
        demo_cr = demo_c.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, demo_c.shape[-1])
        source_idx = idx.unsqueeze(1).expand(-1, n_t).reshape(-1)

        a, b, cc, a_dot, b_dot, c_dot = self._path_weights(
            t, interpolation_path, residual_lambda=residual_lambda
        )
        xt = a.view(-1, 1, 1) * zr + b.view(-1, 1, 1) * yr + cc.view(-1, 1, 1) * xr
        vt = (
            a_dot.view(-1, 1, 1) * zr
            + b_dot.view(-1, 1, 1) * yr
            + c_dot.view(-1, 1, 1) * xr
        )
        w_pca, w_demo = self._condition_path_weights(
            t,
            condition_interpolation_path=condition_interpolation_path,
        )

        perm = torch.randperm(xt.shape[0], device=self.device)
        return (
            xt[perm],
            t[perm],
            vt[perm],
            tilde_cr[perm],
            demo_cr[perm],
            w_pca[perm],
            w_demo[perm],
            source_idx[perm],
        )

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
        gradient_accumulation_steps: int = 1,
        interpolation_path: str = "beizer",
        condition_interpolation_path: str = "piecewise-linear-half",
        residual_lambda: float = 0.2,
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
        pca_condition_batch_size: int = 1024,
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
        evaluator=None,
        eval_metadata: dict | None = None,
        **_,
    ):
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive, got "
                f"{gradient_accumulation_steps}"
            )
        if mf is not None or n_t_local is not None or n_t_global is not None:
            print("[DGFMv3] Ignoring deprecated mf/n_t_local/n_t_global; using n_t only.")
        if max_pca_samples <= 0:
            raise ValueError(f"max_pca_samples must be positive, got {max_pca_samples}")
        if pca_condition_batch_size <= 0:
            raise ValueError(
                "pca_condition_batch_size must be positive, got "
                f"{pca_condition_batch_size}"
            )
        if cluster_merge_k <= 0:
            raise ValueError(f"cluster_merge_k must be positive, got {cluster_merge_k}")
        if not 0.0 < cluster_outlier_q < 1.0:
            raise ValueError(f"cluster_outlier_q must be in (0, 1), got {cluster_outlier_q}")
        if dgfm_trunc_low >= dgfm_trunc_high:
            raise ValueError(
                f"dgfm_trunc_low must be smaller than dgfm_trunc_high, "
                f"got {dgfm_trunc_low} >= {dgfm_trunc_high}"
            )

        print("Clustering dataset for DGFMv3 . . .")
        self.model = self.model.to(self.device)
        pca_conditions = self._snapshot_conditions_for_pca(
            conditions.to(self.device),
            batch_size=pca_condition_batch_size,
        )
        X_np = target_trajectories.detach().cpu().numpy().reshape(
            target_trajectories.shape[0], -1
        )
        C_np = pca_conditions.detach().cpu().numpy()

        clusters, inv_cluster = cluster_points_joint(
            X_np,
            C_np,
            m=cluster_size,
            jaccard_thresh=cluster_jaccard_thresh,
            merge_k=cluster_merge_k,
            standardize=cluster_standardize,
            scale_x=scale_x,
            scale_c=scale_c,
        )
        cluster_sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.float64)

        print(f"{len(clusters)} clusters made! Applying joint X-C PCA . . .")
        mu_x, mu_c, B, Szz, Szc, Scc, weights = compute_cluster_pca_fast_joint(
            X_np,
            C_np,
            clusters,
            d_x=cluster_d,
            eps=cluster_eps,
            outlier_q=cluster_outlier_q,
            max_pca_samples=max_pca_samples,
            n_jobs=pca_n_jobs,
        )

        mixture_sampler = MixtureSampler(
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

        N = target_trajectories.shape[0]
        joint_N = N * n_t
        best_avg_reward = 0.0
        best_success_rate = 0.0
        best_model = self._copy_eval_model()
        success_rate_recs = {}
        stop_count = 0
        best_validation_rollouts = None
        self.best_validation_rollouts = None

        do_validation = val_period > 0 and val_trials > 0
        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials, eval_base_seed)

        try:
            self.model = self.model.to(self.device)
            self._init_ema()
            if self.use_ema and self.ema is None:
                raise RuntimeError("DGFMv3 EMA initialization failed while use_ema=True")
            target_trajectories = target_trajectories.to(self.device)
            conditions = conditions.to(self.device)

            for epoch in tqdm(range(1, max_epochs + 1), desc="DGFMv3 Training", unit="epoch"):
                self.model.train()
                vision_epoch_hook = getattr(self, "vision_epoch_hook", None)
                if vision_epoch_hook is not None:
                    vision_epoch_hook(epoch)
                perm_t = torch.randperm(N, device=self.device)

                (
                    XT,
                    TIN,
                    VT,
                    TILDE_CT,
                    DEMO_CT,
                    W_PCA,
                    W_DEMO,
                    SOURCE_IDX,
                ) = self._build_joint_interpolants(
                    target_trajectories=target_trajectories,
                    conditions=conditions,
                    perm_t=perm_t,
                    mixture_sampler=mixture_sampler,
                    inv_cluster=inv_cluster,
                    cluster_sizes=cluster_sizes,
                    n_t=n_t,
                    interpolation_path=interpolation_path,
                    condition_interpolation_path=condition_interpolation_path,
                    residual_lambda=residual_lambda,
                    dgfm_truncated=dgfm_truncated,
                    dgfm_trunc_low=dgfm_trunc_low,
                    dgfm_trunc_high=dgfm_trunc_high,
                )

                loss_sum = 0.0
                batch_count = 0
                accumulated_microbatches = 0
                accumulated_samples = 0
                self.optimizer.zero_grad()
                for i in range(0, joint_N, batch_size):
                    xb = XT[i:i + batch_size]
                    tb = TIN[i:i + batch_size]
                    vb = VT[i:i + batch_size]
                    tilde_cb = TILDE_CT[i:i + batch_size]
                    demo_cb = DEMO_CT[i:i + batch_size]
                    w_pca = W_PCA[i:i + batch_size]
                    w_demo = W_DEMO[i:i + batch_size]
                    source_idx = SOURCE_IDX[i:i + batch_size]

                    with torch.enable_grad():
                        vision_condition_fn = getattr(self, "vision_condition_fn", None)
                        if vision_condition_fn is not None:
                            demo_cb = vision_condition_fn(source_idx, demo_cb)
                        if tilde_cb.shape != demo_cb.shape:
                            raise ValueError(
                                "DGFMv3 PCA and demo condition dimensions disagree: "
                                f"{tilde_cb.shape[-1]} != {demo_cb.shape[-1]}"
                            )
                        cb = (
                            w_pca.unsqueeze(1) * tilde_cb
                            + w_demo.unsqueeze(1) * demo_cb
                        )
                        pred = self.model(xb, tb, cb)
                        sq_err = (pred - vb) ** 2
                        if hasattr(self.model, "loss_mask") and self.model.loss_mask is not None:
                            sq_err = sq_err * self.model.loss_mask
                        loss = sq_err.mean()

                        microbatch_samples = xb.shape[0]
                        (loss * microbatch_samples).backward()
                        accumulated_microbatches += 1
                        accumulated_samples += microbatch_samples
                        accumulation_boundary = (
                            accumulated_microbatches == gradient_accumulation_steps
                            or i + batch_size >= joint_N
                        )
                        if accumulation_boundary:
                            normalize_accumulated_gradients(
                                self.optimizer, accumulated_samples
                            )
                            vision_after_backward_hook = getattr(
                                self, "vision_after_backward_hook", None
                            )
                            if vision_after_backward_hook is not None:
                                vision_after_backward_hook()
                            self.optimizer.step()
                            self._step_ema()
                            self.optimizer.zero_grad()
                            accumulated_microbatches = 0
                            accumulated_samples = 0

                        loss_sum += float(loss.item())
                        batch_count += 1

                if self.scheduler is not None:
                    self.scheduler.step()

                avg_loss = loss_sum / max(1, batch_count)
                if do_validation and epoch % val_period == 0:
                    eval_model_obj = self._eval_model()
                    eval_model_obj.eval()
                    eval_kwargs = dict(
                        model=eval_model_obj,
                        model_class=VectorField,
                        task_name=self.task_name,
                        seq_len=self.horizon,
                        dof=self.dof,
                        param_len=self.condition_dim,
                        gripper_idx=self.gripper_idx,
                        val_params=val_params,
                        env_settings_all=env_settings_all,
                        device=self.device,
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
                    if evaluator is None:
                        success_rate, avg_reward, validation_rollouts = eval_model(**eval_kwargs)
                    else:
                        metadata = dict(eval_metadata or {})
                        metadata.update(
                            eval_kwargs={k: v for k, v in eval_kwargs.items() if k != "model"},
                            val_trials=val_trials,
                            eval_base_seed=eval_base_seed,
                            normalization_stats=self.normalization_stats,
                        )
                        response = evaluator.evaluate(eval_model_obj, epoch, metadata)
                        success_rate = response.success_rate
                        avg_reward = response.avg_reward
                        validation_rollouts = response.validation_rollouts

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
                        if validation_result_is_better(
                            success_rate, avg_reward, best_success_rate, best_avg_reward,
                            has_best=best_validation_rollouts is not None,
                        ):
                            best_avg_reward = avg_reward
                            best_success_rate = success_rate
                            best_model = self._copy_eval_model()
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
            best_model = self._copy_eval_model()
        return best_model, self.model, success_rate_recs, mixture_sampler

