"""
Flow Matching (FM) training & evaluation functions on RoboSuite
===============================================================

This module implements the training / inference building blocks for learning
trajectory generators with Flow Matching on RoboSuite tasks. It includes:

What this module provides
-------------------------
- **VectorField**: a conditional velocity field (1D U-Net backbone with FiLM)
  mapping (x_t, t, c) → v_t.
- **Flow classes**: `VanillaFM`, `UniformFM`, and `ShiftedFM` train fixed-horizon
  state-conditioned policies.
- **Evaluation**: 'eval_model' runs many parallel rollouts in RoboSuite,
  reports success rate / average reward, and (optionally) renders a grid video.
- **Utilities**: helpers like 'run_flow' for integrating the learned vector field from t=0→1.

Key design choices & assumptions
--------------------------------
1) **Shapes & conventions**
   - Trajectories are '(B, T, D)' = (batch, seq_len, dof).
   - Environment parameters are '(B, P)'.
   - Times 't ∈ [0, 1]'.
   - Gripper pose channels are learned like arm channels. For gripper tasks,
     datasets use compact normalized gripper poses: 0 closed, 1 open.

2) **Trainers**
   - **UniformFM**: sample t ~ Uniform[0,1], regress v_t (standard FM).
   - **ShiftedFM**: shift / warp t to emphasize early (or late) time steps via Beta distribution.
   - **DGFM** lives in `Robot_simulation.DGFM_util`.
   - Schedules: cosine with warmup; Adam optimizer; early stopping supported.

3) **Evaluation loop**
   - Before training: build env, extract env params c
   - For each trial: restore env, sample a trajectory by integrating the vector field (via `run_flow`), 
     upsample / smooth externally, replay, then compute success / reward. 
     Multiprocessing is used for speed.
   - Rendering (optional) uses the external utilities from
     'Robot_simulation.heuristics_util' (not defined here).

Note
----
This module focuses on learning and evaluation. Environment construction,
state restoration, trajectory smoothing, and rendering are provided by
'Robot_simulation.heuristics_util'.
For nut assembly task, grasping the nut is ensured by lifting the nut in '_align_handle_to_nut'
"""

import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Tuple, List
from torch.distributions import Beta
import logging
logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)  
robosuite_logger.propagate = False        
for h in list(robosuite_logger.handlers): 
    robosuite_logger.removeHandler(h)
from robosuite.utils.transform_utils import mat2quat, quat_multiply, quat_inverse

from Robot_simulation.env_util import make_env
from Robot_simulation.heuristics_util import get_dynamic_state, render_trajectory, write_grid_video, step_towards


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for 1D feature maps.

    Applies a per-channel affine transform conditioned on a context vector.

    Args:
    in_channels: Number of channels in the feature map to be modulated. (C)
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


PANDA_GRIPPER_OPEN_QPOS = 0.04


def _gripper_qpos_to_normalized(gripper_qpos: np.ndarray) -> np.ndarray:
    gripper_qpos = np.asarray(gripper_qpos, dtype=np.float32)
    return np.clip(np.mean(gripper_qpos, axis=-1) / PANDA_GRIPPER_OPEN_QPOS, 0.0, 1.0)


def _normalized_gripper_to_action(value) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 1.0 - 2.0 * value


class VectorField(nn.Module):
    """Conditional vector field v(x, t, c) over joint trajectories.

    The model predicts a velocity field for every policy channel, including
    normalized gripper pose channels. Time is encoded via a small MLP and
    concatenated with (optionally scaled) environment parameters for FiLM
    conditioning.

    Args:
        seq_len: Number of time steps (T) per trajectory.
        dof: Total joint DoF per time step (includes grippers if present).
        param_len: Dimensionality of environment parameter vector `c`.
        gripper_idx: Optional gripper channel indices, kept for compatibility.
            These channels are still modeled and included in the loss.

    Attributes:
        loss_mask (Tensor): (1, 1, dof) all-ones mask kept for compatibility.

    Forward Args:
        x: (B, T, dof) current state on the straight path between x0 and x1.
        t: (B, 1) time in [0, 1].
        env_params: (B, param_len) environment parameters (scaled by 100 internally).

    Returns:
        (B, T, dof) velocity field for all policy channels.
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
        self.register_buffer("gripper_idx", gripper_idx, persistent=False)

        loss_mask = torch.ones(dof, dtype=torch.float32)
        self.register_buffer("loss_mask", loss_mask.view(1, 1, dof), persistent=False)

        # Time embedding for FiLM conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32)
        )

        # additional encoding if needed
        """
        self.param_embed = nn.Sequential(
            nn.Linear(param_len, 16), nn.ReLU(),
            nn.Linear(16, 16)
        )
        """

        # Condition dimension = param_len + time embedding dim
        condition_dim = 32 + param_len

        # 1D U-Net for sequence modeling
        self.unet = UNet1D(dof=dof, condition_dim=condition_dim)

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
        # p_embed = self.param_embed(env_params_norm) # in cased of additional encoding
        p_embed = env_params_norm # without environment parameter encoding

        # Create conditioning vector
        condition = torch.cat([p_embed, t_embed], dim=-1)  # (B, cond_dim)

        v = self.unet(x.permute(0, 2, 1).contiguous(), condition)
        return v.permute(0, 2, 1).contiguous()


def build_state_conditioned_windows(
    trajectories: np.ndarray,
    dynamic_states: np.ndarray | None,
    static_env_params: np.ndarray,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert full episodes into diffusion-policy-style FM training windows.

    Each training item maps
    ``[current_joint_angles, current_dynamic_state, static_environment_params]``
    to the next fixed-horizon joint trajectory.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    xs, cs = [], []
    n_eps = len(trajectories)
    for ep in range(n_eps):
        q = np.asarray(trajectories[ep], dtype=np.float32)
        dyn = None if dynamic_states is None else dynamic_states[ep]
        if dyn is None:
            dyn = np.zeros((len(q), 0), dtype=np.float32)
        else:
            dyn = np.asarray(dyn, dtype=np.float32)
            if dyn.shape[0] != len(q):
                raise ValueError(f"dynamic_states[{ep}] length {dyn.shape[0]} != trajectory length {len(q)}")
        env_c = np.asarray(static_env_params[ep], dtype=np.float32)
        for start in range(0, len(q) - horizon + 1, stride):
            xs.append(q[start:start + horizon])
            cs.append(np.concatenate([q[start], dyn[start], env_c], axis=0))

    if not xs:
        raise ValueError(f"No training windows produced; horizon={horizon} is longer than all trajectories.")

    return np.asarray(xs, dtype=np.float32), np.asarray(cs, dtype=np.float32)


def make_policy_condition(
    current_q: np.ndarray,
    current_dynamic_state: np.ndarray | None,
    static_env_params: np.ndarray,
) -> np.ndarray:
    if current_dynamic_state is None:
        current_dynamic_state = np.zeros((0,), dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(current_q, dtype=np.float32).reshape(-1),
            np.asarray(current_dynamic_state, dtype=np.float32).reshape(-1),
            np.asarray(static_env_params, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )


class VanillaFM:
    """State-conditioned Flow Matching trainer for fixed-horizon policies."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        *,
        task_name: str,
        horizon: int,
        dof: int,
        condition_dim: int,
        gripper_idx=None,
        time_sampling: str = "uniform",
        beta_a: float = 1.5,
        beta_b: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.task_name = task_name
        self.horizon = horizon
        self.dof = dof
        self.condition_dim = condition_dim
        self.gripper_idx = gripper_idx
        self.time_sampling = time_sampling
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.device = device

    def sample_t(self, n: int) -> torch.Tensor:
        if self.time_sampling == "shifted":
            t = Beta(self.beta_a, self.beta_b).sample((n, 1)).to(self.device)
            return torch.ones_like(t) - t
        return torch.rand(n, device=self.device).unsqueeze(-1)

    def train(
        self,
        target_trajectories,
        conditions,
        *,
        n_t: int,
        max_epochs: int,
        batch_size: int,
        val_period: int = 5,
        early_stopping: bool = True,
        stop_criteria: int = 3,
        val_trials: int = 25,
    ):
        N = target_trajectories.shape[0]
        best_avg_reward = 0.0
        best_success_rate = 0.0
        best_model = copy.deepcopy(self.model)
        records = {}
        stop_count = 0

        do_validation = val_period > 0 and val_trials > 0
        env_settings_all, val_params = (None, None)
        if do_validation:
            env_settings_all, val_params = _generate_val_env(self.task_name, val_trials)

        try:
            self.model = self.model.to(self.device)
            target_trajectories = target_trajectories.to(self.device)
            conditions = conditions.to(self.device)
            for epoch in tqdm(range(1, max_epochs + 1), desc=f"{self.time_sampling.title()}FM Training", unit="epoch"):
                self.model.train()
                perm_t = torch.randperm(N, device=self.device)
                loss_sum = 0.0
                for i in range(0, N, batch_size):
                    idx = perm_t[i:min(i + batch_size, N)]
                    x1 = target_trajectories[idx]
                    x0 = torch.randn(len(idx), self.horizon, self.dof, device=self.device)
                    t = self.sample_t(len(idx) * n_t)
                    cond = conditions[idx, :]

                    x1r = x1.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
                    x0r = x0.unsqueeze(1).expand(-1, n_t, -1, -1).reshape(-1, self.horizon, self.dof)
                    t_col = t.view(-1, 1, 1)
                    xt = (1 - t_col) * x0r + t_col * x1r
                    cond_r = cond.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, self.condition_dim)
                    target_v = x1r - x0r

                    pred_v = self.model(xt, t, cond_r)
                    sq_err = (pred_v - target_v) ** 2
                    if hasattr(self.model, "loss_mask") and self.model.loss_mask is not None:
                        sq_err = sq_err * self.model.loss_mask
                    loss = sq_err.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_sum += float(loss.item())

                if self.scheduler is not None:
                    self.scheduler.step()

                if do_validation and epoch % val_period == 0:
                    success_rate, avg_reward = eval_model(
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
                    )
                    records[epoch] = {"success_rate": success_rate, "avg_reward": avg_reward, "loss": loss_sum}
                    if success_rate < best_success_rate:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
                        if early_stopping:
                            if stop_count == stop_criteria:
                                tqdm.write("Early stopping triggered.")
                                break
                            stop_count += 1
                    elif (success_rate > best_success_rate) or (best_avg_reward < avg_reward):
                        best_avg_reward = avg_reward
                        best_success_rate = success_rate
                        best_model = copy.deepcopy(self.model)
                        stop_count = 0
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f} | Best model saved")
                    else:
                        tqdm.write(f"Epoch {epoch}: success_rate={success_rate:.3f}, average reward={avg_reward:.3f}, loss={loss_sum:.3f}")
        except KeyboardInterrupt:
            tqdm.write("Training interrupted by user. Returning best model so far...")

        if not records:
            best_model = copy.deepcopy(self.model)
        return best_model, self.model, records


class UniformFM(VanillaFM):
    def __init__(self, *args, **kwargs):
        kwargs["time_sampling"] = "uniform"
        super().__init__(*args, **kwargs)


class ShiftedFM(VanillaFM):
    def __init__(self, *args, **kwargs):
        kwargs["time_sampling"] = "shifted"
        super().__init__(*args, **kwargs)

@torch.no_grad()
def run_flow(model, x, c, device, n_steps=100):
    """Generates samples by transporting noisy samples through the given vector field

    Uses Euler integration for the neural vector field v(x, c).

    Args:
        model: Neural vector field model.
        x: Samples from the base distribution.
        c: Environment parameters.
        n_steps: number of Euler steps.

    Returns:
        output x with same size as input x
    """
    dt = 1.0 / n_steps

    t = torch.empty((x.shape[0], 1), device=device, dtype=x.dtype)

    for i in range(n_steps):
        t.fill_(i * dt)
        v = model(x, t, c)
        x.add_(v, alpha=dt)

    return x


def _get_environment_params(
    env,
    task_name: str):
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
        handle_id  = env.door_handle_site_id
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

        peg_id  = env.peg1_body_id
        peg_pos = env.sim.data.body_xpos[peg_id].copy()
        environment_parameters = (nut_pos[0], nut_pos[1], yaw, peg_pos[0], peg_pos[1])

    return environment_parameters


def _to_action_from_q(q, task_name):
    """Pack a joint vector into the action format expected by the controller.

    The packing mirrors existing controller wrappers for each task. In particular,
    compact policy gripper channels are normalized poses where 0 is closed and
    1 is open; these are mapped to the controller's close/open command.

    Args:
        q: (D,) joint configuration for the whole system.
        task_name: One of {"door", "nut", "wipe", "two_arm"}.

    Returns:
        np.ndarray: Action vector compatible with `env.step(action)` for the task.
    """
    q = np.asarray(q, dtype=np.float32)
    if task_name in ["door", "nut"]:
        arm_q = q[:7]
        if q.shape[0] == 8:
            grip_sc = _normalized_gripper_to_action(q[7])
        else:
            grip_sc = float((q[7] + q[8]) / 2.0)
        return np.concatenate([arm_q, [grip_sc]])
    elif task_name == "wipe":
        return q[:7]
    else:  # two_arm
        arm1 = q[0:7]
        if q.shape[0] == 16:
            grip1 = _normalized_gripper_to_action(q[7])
            arm2 = q[8:15]
            grip2 = _normalized_gripper_to_action(q[15])
        else:
            grip1 = float((q[7] + q[8]) / 2.0)
            arm2 = q[9:16]
            grip2 = float((q[16] + q[17]) / 2.0)
        return np.concatenate([arm1, [grip1], arm2, [grip2]])


def _current_robot_q(env, task_name: str) -> np.ndarray:
    full_qpos = env.sim.data.qpos.copy()
    if task_name in ["door", "nut"]:
        return np.concatenate([full_qpos[:7], [float(_gripper_qpos_to_normalized(full_qpos[7:9]))]])
    if task_name == "wipe":
        return full_qpos[:7]
    return np.concatenate(
        [
            full_qpos[0:7],
            [float(_gripper_qpos_to_normalized(full_qpos[7:9]))],
            full_qpos[9:16],
            [float(_gripper_qpos_to_normalized(full_qpos[16:18]))],
        ]
    )


def _condition_from_env(env, task_name: str, static_c: np.ndarray, param_len: int) -> tuple[np.ndarray, np.ndarray]:
    q0 = _current_robot_q(env, task_name)
    dyn = get_dynamic_state(env, task_name)
    static_flat = np.asarray(static_c, dtype=np.float32).reshape(-1)
    dyn_dim = param_len - q0.shape[0] - static_flat.shape[0]
    if dyn_dim < 0:
        raise ValueError(f"Model param_len {param_len} is shorter than q + static condition length")
    if dyn_dim == 0:
        dyn = None
    else:
        dyn = np.asarray(dyn, dtype=np.float32).reshape(-1)
        if dyn.shape[0] < dyn_dim:
            dyn = np.pad(dyn, (0, dyn_dim - dyn.shape[0]))
        elif dyn.shape[0] > dyn_dim:
            dyn = dyn[:dyn_dim]
    cond = make_policy_condition(q0, dyn, static_flat)
    if cond.shape[0] != param_len:
        raise ValueError(f"Condition length {cond.shape[0]} != model param_len {param_len}")
    return cond.astype(np.float32), q0


def _state_policy_success(env, task_name: str) -> bool:
    if not env._check_success():
        return False
    if task_name == "two_arm":
        z0 = env._handle0_xpos[2]
        z1 = env._handle1_xpos[2]
        return abs(z1 - z0) < 0.05
    return True


def _state_policy_env_worker(
    conn,
    idx: int,
    task_name: str,
    setting: dict,
    static_c: np.ndarray,
    seq_len: int,
    param_len: int,
    max_policy_steps: int,
):
    env = None
    executed = []
    try:
        env = make_env(
            task_name,
            use_joint_control=True,
            environment_setting=setting,
            training=True,
        )
        steps = 0
        cond, _ = _condition_from_env(env, task_name, static_c, param_len)
        conn.send({"type": "cond", "idx": idx, "cond": cond})

        while True:
            msg = conn.recv()
            if msg.get("type") == "close":
                break
            if msg.get("type") != "act":
                raise ValueError(f"Unknown worker message: {msg}")

            q_low = np.asarray(msg["q_low"], dtype=np.float32)

            done = False
            for q in q_low:
                _, _, done, _ = env.step(_to_action_from_q(q, task_name))
                executed.append(q.copy())
                if done or env._check_success():
                    break

            steps += 1
            success = _state_policy_success(env, task_name)
            if success or done or steps >= max_policy_steps:
                conn.send({
                    "type": "result",
                    "idx": idx,
                    "success": bool(success),
                    "reward": float(env.reward()),
                    "traj": np.asarray(executed, dtype=np.float32),
                    "setting": setting,
                })
                break

            cond, _ = _condition_from_env(env, task_name, static_c, param_len)
            conn.send({"type": "cond", "idx": idx, "cond": cond})
    except Exception as exc:
        conn.send({"type": "error", "idx": idx, "error": repr(exc)})
    finally:
        if env is not None:
            env.close()
        conn.close()


def _align_handle_to_nut(env, 
                         delta_x: float = 0.05, delta_z: float = 0.1, 
                         use_vision: bool = False, camera_name: str = "frontview"):
    def record_q(): # helper for recording, empty
        return

    # randomize peg x and y coordinates & shift upward with the table (delta_z)
    """
    delta_x_range = [-0.015, 0.015]
    delta_y_range = [-0.015, 0.015]
    delta_x = np.random.uniform(delta_x_range[0], delta_x_range[1])
    delta_y = np.random.uniform(delta_y_range[0], delta_y_range[1])
    """

    # fixed peg position for now
    delta_x = -delta_x
    delta_y = 0
    
    env.sim.model.body_pos[env.peg1_body_id][0] += delta_x
    env.sim.data.body_xpos[env.peg1_body_id][0] += delta_x
    env.sim.model.body_pos[env.peg1_body_id][1] += delta_y
    env.sim.data.body_xpos[env.peg1_body_id][1] += delta_y
    env.sim.model.body_pos[env.peg1_body_id][2] += delta_z
    env.sim.data.body_xpos[env.peg1_body_id][2] += delta_z

    env.sim.model.body_pos[env.peg2_body_id][2] = 0
    env.sim.data.body_xpos[env.peg2_body_id][2] = 0
    
    robot = env.robots[0]
    # arm+gripper
    arm_joints  = robot.robot_model.joints
    grip_joints = robot.gripper["right"].joints
    all_joints  = arm_joints + grip_joints
    joint_idx   = [env.sim.model.get_joint_qpos_addr(n) for n in all_joints]
    # eef & square peg & nut handle
    eef_id      = list(robot.eef_site_id.values())[0]
    peg_id      = env.peg1_body_id
    nut_handle_id = env.object_site_ids[0]
    adim        = env.action_dim

    init_qpos = env.sim.data.qpos[joint_idx].copy()

    nut_pos = env.sim.data.site_xpos[nut_handle_id].copy()
    nut_pos[2] = getattr(env, "table_offset", np.zeros(3))[2] # since the pegs drop from midair

    peg_pos = env.sim.data.body_xpos[peg_id].copy()

    R0       = env.sim.data.site_xmat[nut_handle_id].reshape(3,3)
    quat0    = mat2quat(R0)
    for axis, ang in [(R0[:,0], np.pi), (R0[:, 2], -np.pi/2)]:
        axis = axis / np.linalg.norm(axis)
        q_rot = np.concatenate([axis * np.sin(ang/2), [np.cos(ang/2)]]).astype(np.float32)
        quat0 = quat_multiply(q_rot, quat0)
    # calculate minimum shift in orientation
    q_cur = mat2quat(env.sim.data.site_xmat[eef_id].reshape(3,3))
    q_rel = quat_multiply(quat_inverse(q_cur), quat0)
    angle = 2 * np.arccos(np.clip(q_rel[3], -1.0, 1.0))
    if angle > np.pi/2 and angle < np.pi*3/2:
        axis, ang = (R0[:, 2], np.pi)
        axis = axis / np.linalg.norm(axis)
        q_rot = np.concatenate([axis * np.sin(ang/2), [np.cos(ang/2)]]).astype(np.float32)
        quat0 = quat_multiply(q_rot, quat0)

    # ---------- PHASE1‑1: 100‑step approach to pre-grasp pose ----------
    pre_grasp = nut_pos + np.array([0.0, 0.0, 0.06], dtype=np.float32) # 6cm above the nut
    step_towards(env, eef_id, adim, record_q,
                 target_pos=pre_grasp,
                 target_quat=quat0,
                 steps=100,
                 gripper_val=-1)
    # ---------- PHASE1‑2: 20‑step careful approach to nut handle ----------
    grasp_height = nut_pos + np.array([0.0, 0.0, 0.015], dtype=np.float32) # 15mm above handle
    step_towards(env, eef_id, adim, record_q,
                 target_pos=grasp_height,
                 target_quat=quat0,
                 steps=20,
                 gripper_val=-1)
    # ---------- PHASE2: 10‑step grasping handle ----------
    for _ in range(10):
        a = np.zeros(adim); a[6] = 1.0
        obs, _, _, _ = env.step(a)

    # ---------- record environment setting (to record after nuts drop) ----------
    # environment_setting = save_mj_state(env) # save full mujoco settings
    environment_setting = {
        "qpos":     env.sim.data.qpos.copy(),
        "qvel":     env.sim.data.qvel.copy(),
        "body_pos": env.sim.model.body_pos.copy(),
        "body_quat":env.sim.model.body_quat.copy(),
        "act": env.sim.data.act.copy(),
        "ctrl": env.sim.data.ctrl.copy(),
        "mocap_pos": env.sim.data.mocap_pos.copy(),
        "mocap_quat": env.sim.data.mocap_quat.copy()
    }
    yaw = np.arctan2(R0[1,0], R0[0,0])
    # environment_parameters = (nut_pos[0], nut_pos[1], yaw, peg_pos[0], peg_pos[1]) # for peg variation
    environment_parameters = (nut_pos[0], nut_pos[1], yaw)
    if (use_vision):
        vision = env.sim.render(640, 480, camera_name=camera_name)
    else:
        vision = None

    # ---------- PHASE3: 20‑step lift to check grasp ----------
    check_grasp = False
    step_towards(env, eef_id, adim, record_q,
                 target_pos=pre_grasp,
                 target_quat=quat0,
                 steps=20,
                 gripper_val=1)
    if (env.sim.data.site_xpos[nut_handle_id][2] > nut_pos[2] + 0.02):
        check_grasp = True

    return environment_setting, environment_parameters, check_grasp, vision
def _generate_val_env(task_name, val_trials):
    env_params_list: List[np.ndarray] = []
    env_settings_all: List[dict] = []

    if (task_name == "nut"):
        for _ in range(val_trials):
            for i in range(100):
                env = make_env(task_name)
                env.reset()
                env_setting, env_param, check_grasp = _align_handle_to_nut(env)
                env.close()  

                if check_grasp:
                    env_settings_all.append(env_setting)
                    env_params_list.append(env_param)
                    break
                if i == 99:
                    print("Nut environment failed grasping!")
    else:
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

    return env_settings_all, val_params


def _rollout_batch(
    task_name: str,
    q_low_batch: np.ndarray,          # (m, T, D) on CPU
    env_settings: List[dict],         # length m
    print_true: bool) -> Tuple[int, float, list, list]:
    """Worker: restore envs from settings, upsample to controller rate, roll out CPU-only."""
    successes, reward_sum = 0, 0.0
    success_info, fail_info = [], []

    m = len(env_settings)
    assert m == len(q_low_batch), "settings and q_low_batch length mismatch"

    for i in range(m):
        setting = env_settings[i]

        env = make_env(task_name,
                       use_joint_control=True,
                       environment_setting=setting,
                       training=True)

        # Policy windows are already sampled at the environment control rate.
        q_low  = q_low_batch[i]
        
        for q in q_low:
            env.step(_to_action_from_q(q, task_name))

        if env._check_success():
            if task_name == "two_arm": # check grasp for two_arm
                z0 = env._handle0_xpos[2]
                z1 = env._handle1_xpos[2]

                if abs(z1 - z0) < 0.05:
                    successes += 1
                    success_info.append({"traj": q_low, "setting": setting})
                else:
                    fail_info.append({"traj": q_low, "setting": setting})
            else:
                successes += 1
                success_info.append({"traj": q_low, "setting": setting})
        else:
            fail_info.append({"traj": q_low, "setting": setting})

        reward_sum += env.reward()
        env.close()

    return successes, reward_sum, success_info, fail_info


def _run_flow_batched(
    model,
    cond_batch: np.ndarray,
    seq_len: int,
    dof: int,
    device: str,
    rng: np.random.RandomState,
    flow_steps: int,
    gpu_chunk_size: int | None = None,
) -> np.ndarray:
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        device = "cpu"

    n = cond_batch.shape[0]
    out = np.empty((n, seq_len, dof), dtype=np.float32)
    chunk = n if gpu_chunk_size is None or gpu_chunk_size <= 0 else gpu_chunk_size

    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        bs = e - s
        x0 = torch.from_numpy(rng.randn(bs, seq_len, dof).astype(np.float32)).to(device)
        c = torch.from_numpy(cond_batch[s:e].astype(np.float32)).to(device)
        if use_cuda:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                q_low = run_flow(model, x0, c, device, n_steps=flow_steps)
            out[s:e] = q_low.float().cpu().numpy()
            del q_low
            torch.cuda.empty_cache()
        else:
            with torch.inference_mode():
                q_low = run_flow(model, x0, c, device, n_steps=flow_steps)
            out[s:e] = q_low.cpu().numpy()
    return out


def _rollout_state_policy_synchronized(
    model,
    task_name: str,
    seq_len: int,
    dof: int,
    param_len: int,
    static_params: np.ndarray,
    env_settings: List[dict],
    device: str,
    trials: int,
    num_workers: int,
    base_seed: int,
    max_policy_steps: int,
    flow_steps: int,
    gpu_chunk_size: int | None = None,
) -> Tuple[float, float, list, list]:
    """Synchronize state-conditioned environments and batch flow inference in the parent."""
    rng = np.random.RandomState(base_seed)
    total_success = 0
    total_reward = 0.0
    success_info: List[dict] = []
    failure_info: List[dict] = []
    ctx = get_context("spawn")

    for batch_start in range(0, trials, max(1, num_workers)):
        batch_end = min(trials, batch_start + max(1, num_workers))
        conns = {}
        procs = {}
        active = set()

        for idx in range(batch_start, batch_end):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_state_policy_env_worker,
                args=(
                    child_conn,
                    idx,
                    task_name,
                    env_settings[idx],
                    static_params[idx],
                    seq_len,
                    param_len,
                    max_policy_steps,
                ),
            )
            proc.start()
            child_conn.close()
            conns[idx] = parent_conn
            procs[idx] = proc
            active.add(idx)

        pending_conditions = {}
        while active:
            for idx in list(active):
                if idx in pending_conditions:
                    continue
                msg = conns[idx].recv()
                mtype = msg.get("type")
                if mtype == "cond":
                    pending_conditions[idx] = msg["cond"]
                elif mtype == "result":
                    active.remove(idx)
                    if msg["success"]:
                        total_success += 1
                        success_info.append({"traj": msg["traj"], "setting": msg["setting"]})
                    else:
                        failure_info.append({"traj": msg["traj"], "setting": msg["setting"]})
                    total_reward += msg["reward"]
                elif mtype == "error":
                    raise RuntimeError(f"State rollout worker {idx} failed: {msg['error']}")
                else:
                    raise RuntimeError(f"Unexpected state rollout message from {idx}: {msg}")

            ready = [idx for idx in sorted(active) if idx in pending_conditions]
            if not ready:
                continue

            cond_batch = np.stack([pending_conditions.pop(idx) for idx in ready], axis=0)
            q_low_batch = _run_flow_batched(
                model,
                cond_batch,
                seq_len,
                dof,
                device,
                rng,
                flow_steps,
                gpu_chunk_size,
            )
            for local_i, idx in enumerate(ready):
                conns[idx].send({"type": "act", "q_low": q_low_batch[local_i]})

        for idx, conn in conns.items():
            conn.close()
        for idx, proc in procs.items():
            proc.join()
            if proc.exitcode not in (0, None):
                raise RuntimeError(f"State rollout worker {idx} exited with code {proc.exitcode}")

    return total_success / trials, total_reward / trials, success_info, failure_info


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
    gpu_chunk_size: int | None = None,
    q_low = None,
    base_mixture = False,
    max_policy_steps: int = 20,
    flow_steps: int = 100) -> Tuple[float, float]:
    """ Evaluates a flow model on given environments.

    Multiprocessing is used by calling _rollout_batch to evaluate multiple environments in parallel.

    Args:
        model: Neural vector field model to evaluate.
        task_name: Name of task, within {"door", "two arm", "nut", "wipe (currently unimplemented)"}
        trials: number of evaluation environments
        val_params: (trials, ) environment parameters of the evaluation environments
        env_settings_all: (trials, ) list of dicts to restore evaluation environments
        render_num: number of successful trajectories to render
        render_width: length of width/height for the rendered grid video (render_width^2 - render_num failed trajectories rendered)
        gpu_chunk_size: In case of multiple neural inference

    Pipeline:
        1) Environments are formed outside the function
        2) Environment parameters are passed to run the flow in a chunk on GPU (if available)
        3) Workers restore envs, upsample, and roll out on CPU
        4) For rendering, trajectory is replayed and saved to "render_dir/{task_name}_grid_{video_name}.mp4"

    Returns:
        Success rate and mean reward

    Evaluation pipeline:
      1) Parent makes envs sequentially, saves settings/params, then closes.
      2) Parent runs ONE GPU forward (optionally chunked) to get all q_low.
      3) Workers restore envs, upsample, and roll out on CPU (no GPU in workers).
    """
    np_rng = np.random.RandomState(base_seed)


    state_conditioned = val_params.shape[1] != param_len
    if state_conditioned:
        if model is None:
            raise ValueError("State-conditioned evaluation requires a model.")
        model.eval()
        success_rate, mean_reward, success_all, failure_all = _rollout_state_policy_synchronized(
            model=model,
            task_name=task_name,
            seq_len=seq_len,
            dof=dof,
            param_len=param_len,
            static_params=val_params,
            env_settings=env_settings_all,
            device=device,
            trials=trials,
            num_workers=num_workers,
            base_seed=base_seed,
            max_policy_steps=max_policy_steps,
            flow_steps=flow_steps,
            gpu_chunk_size=gpu_chunk_size,
        )
        success_info = success_all[:render_num]
        fail_slots = max(0, render_width * render_width - render_num)
        failure_info = failure_all[:fail_slots]
        s_count = len(success_info)
        f_count = len(failure_info)

        episode_frames = []
        s_left = min(s_count, render_num)
        f_left = min(f_count, render_width * render_width - render_num)
        while s_left > 0:
            env_r = make_env(task_name, has_offscreen_renderer=True, use_camera_obs=False,
                             use_joint_control=True, environment_setting=success_info[s_left - 1]["setting"],
                             training=True)
            frames = render_trajectory(env_r, task_name, success_info[s_left - 1]["traj"],
                                       success_info[s_left - 1]["traj"][0, :],
                                       camera_name="frontview", hold_init=False, set_init=False)
            episode_frames.append(frames)
            env_r.close()
            s_left -= 1
        while f_left > 0:
            env_r = make_env(task_name, has_offscreen_renderer=True, use_camera_obs=False,
                             use_joint_control=True, environment_setting=failure_info[f_left - 1]["setting"],
                             training=True)
            frames = render_trajectory(env_r, task_name, failure_info[f_left - 1]["traj"],
                                       failure_info[f_left - 1]["traj"][0, :],
                                       camera_name="frontview", hold_init=False, set_init=False)
            episode_frames.append(frames)
            env_r.close()
            f_left -= 1
        if render_num and episode_frames:
            grid_path = os.path.join(render_dir, f"{task_name}_grid_{video_name}.mp4")
            write_grid_video(episode_frames, grid_path, grid_shape=(render_width, render_width))
        return success_rate, mean_reward

    # ---- (1) One GPU forward (optionally chunked) to produce all q_low ----
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not use_cuda:
        device = "cpu"

    # pre-allocate to collect all q_low on CPU
    q_low_all = np.empty((trials, seq_len, dof), dtype=np.float32)

    if q_low is None and model is not None:
        model.eval()
        if gpu_chunk_size is None or gpu_chunk_size <= 0:
            # single shot (ensure it fits!)
            x0 = torch.from_numpy(np_rng.randn(trials, seq_len, dof).astype(np.float32)).to(device)
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
                x0 = torch.from_numpy(np_rng.randn(bs, seq_len, dof).astype(np.float32)).to(device)
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
    elif q_low is not None:
        q_low_all = q_low.cpu().numpy()

        if base_mixture:
            model.eval()
            x0 = torch.from_numpy(q_low_all).to(device)
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


    # ---- (2) Roll out on CPU with multiple workers ----
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
                s == 0
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

    success_rate = total_success / trials
    mean_reward  = total_reward  / trials

    # ---- (optional) render sample grid ----
    episode_frames = []
    s_left = min(s_count, render_num)
    f_left = min(f_count, render_width*render_width - render_num)

    while s_left > 0:
        env_r = make_env(task_name,
                         has_offscreen_renderer=True,
                         use_camera_obs=False,
                         use_joint_control=True,
                         environment_setting=success_info[s_left - 1]["setting"],
                         training=True)
        frames = render_trajectory(env_r,
                                   task_name,
                                   success_info[s_left - 1]["traj"],
                                   success_info[s_left - 1]["traj"][0, :],
                                   camera_name="frontview",
                                   hold_init=False,
                                   set_init=False)
        episode_frames.append(frames)
        env_r.close()
        s_left -= 1

    while f_left > 0:
        env_r = make_env(task_name,
                         has_offscreen_renderer=True,
                         use_camera_obs=False,
                         use_joint_control=True,
                         environment_setting=failure_info[f_left - 1]["setting"],
                         training=True)
        frames = render_trajectory(env_r,
                                   task_name,
                                   failure_info[f_left - 1]["traj"],
                                   failure_info[f_left - 1]["traj"][0, :],
                                   camera_name="frontview",
                                   hold_init=False,
                                   set_init=False)
        episode_frames.append(frames)
        env_r.close()
        f_left -= 1

    if render_num:
        grid_path = os.path.join(render_dir, f"{task_name}_grid_{video_name}.mp4")
        write_grid_video(episode_frames, grid_path,
                         grid_shape=(render_width, render_width))

    return success_rate, mean_reward
