"""Validation evaluator backends for RoboSuite training."""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Robot_simulation.env_util import eval_model


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _stats_to_json(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    if "joint_min" in stats:
        return _jsonable(stats)
    return {
        "joint_min": np.asarray(stats["min"], dtype=np.float32).tolist(),
        "joint_max": np.asarray(stats["max"], dtype=np.float32).tolist(),
        "joint_range": np.asarray(stats["range"], dtype=np.float32).tolist(),
    }


def _stats_from_json(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    if "min" in stats:
        return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}
    return {
        "min": np.asarray(stats["joint_min"], dtype=np.float32),
        "max": np.asarray(stats["joint_max"], dtype=np.float32),
        "range": np.asarray(stats["joint_range"], dtype=np.float32),
    }


@dataclass
class EvaluationRequest:
    request_id: str
    stream_id: str
    source_machine: str
    source_run_id: str | None
    task_name: str
    model_type: str
    epoch: int
    val_trials: int
    eval_base_seed: int
    seq_len: int
    dof: int
    param_len: int
    state_condition_dim: int
    gripper_idx: list[int] | None
    observation_type: str
    camera_names: list[str]
    condition_embed_dim: int
    vision_finetune: bool
    vision_finetune_mode: str
    vision_train_bn: bool
    vision_pool: str
    vision_spatial_softmax_temperature: float
    vision_feature_proj_dim: int
    vision_feature_norm: str
    vision_aug: bool
    vision_random_shift: int
    vision_color_jitter: float
    max_policy_steps: int
    executed_horizon: int
    observation_horizon: int
    recorded_control_freq: int | float
    trajectory_control_freq: int | float
    action_representation: str
    normalization_stats: dict | None
    use_ema: bool
    dp_T_diff: int | None
    dp_schedule_type: str | None
    dp_ddim_steps: int | None
    dp_eta: float | None
    dp_pred_type: str | None
    dp_clip_sample: bool | None
    dp_clip_sample_range: float | None
    render_video: bool
    capture_rollouts: bool
    num_convs_per_block: int = 1
    device: str = "cuda"
    flow_steps: int = 100
    latent_dim: int | None = None
    latent_hidden_dim: int | None = None
    latent_num_layers: int | None = None
    latent_residual: bool | None = None


@dataclass
class EvaluationResponse:
    request_id: str
    stream_id: str
    status: str
    epoch: int
    success_rate: float | None
    avg_reward: float | None
    eval_trials: int
    elapsed_seconds: float
    response_path: str | None
    video_path: str | None
    rollout_path: str | None
    error: str | None
    validation_rollouts: dict | None = None


class BaseEvaluator:
    def evaluate(self, model, epoch: int, metadata: dict) -> EvaluationResponse:
        raise NotImplementedError


class LocalEvaluator(BaseEvaluator):
    """Evaluator backend that preserves the in-process validation path."""

    def evaluate(self, model, epoch: int, metadata: dict) -> EvaluationResponse:
        start = time.time()
        kwargs = dict(metadata["eval_kwargs"])
        kwargs["model"] = model
        success_rate, avg_reward, validation_rollouts = eval_model(**kwargs)
        return EvaluationResponse(
            request_id=metadata.get("request_id", f"local-{epoch}"),
            stream_id=metadata.get("stream_id", "local"),
            status="ok",
            epoch=epoch,
            success_rate=float(success_rate),
            avg_reward=float(avg_reward),
            eval_trials=int(kwargs.get("trials", metadata.get("val_trials", 0))),
            elapsed_seconds=time.time() - start,
            response_path=None,
            video_path=None,
            rollout_path=None,
            error=None,
            validation_rollouts=validation_rollouts,
        )


class RemoteEvaluator(BaseEvaluator):
    """Synchronous remote evaluator using scp/ssh or a shared queue directory."""

    def __init__(
        self,
        *,
        exp_dir: str,
        config: dict,
        mode: str = "direct",
        timeout_sec: int = 3600,
        poll_interval_sec: float = 5.0,
        render_best: bool = False,
    ):
        self.exp_dir = Path(exp_dir)
        self.config = dict(config)
        self.mode = mode
        self.timeout_sec = int(timeout_sec)
        self.poll_interval_sec = float(poll_interval_sec)
        self.render_best = bool(render_best)
        self.stream_id = self.config.get("stream_id") or self.exp_dir.name
        self.source_run_id = self.config.get("source_run_id") or self.exp_dir.name
        self.debug_keep_requests = bool(self.config.get("debug_keep_requests", False))
        self.request_root = self.exp_dir / "remote_eval_requests"
        self.request_root.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model, epoch: int, metadata: dict) -> EvaluationResponse:
        request_id = f"eval-e{int(epoch):06d}-{uuid.uuid4().hex[:8]}"
        request_dir = self.request_root / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = request_dir / "candidate_model.pt"
        request_path = request_dir / "eval_request.json"
        response_path = request_dir / "eval_response.json"

        request = self._build_request(request_id, epoch, metadata)
        torch.save(model.state_dict(), checkpoint_path)
        with request_path.open("w") as f:
            json.dump(dataclasses.asdict(request), f, indent=2)

        start = time.time()
        if self.mode == "queue":
            self._submit_queue(request_dir)
            self._wait_for_file(response_path)
        elif self.mode == "direct":
            self._run_direct(request_dir)
        else:
            raise ValueError(f"Unsupported remote evaluator mode: {self.mode}")

        if not response_path.exists():
            remote_response = request_dir / "remote_eval_response.json"
            if remote_response.exists():
                remote_response.replace(response_path)
        with response_path.open("r") as f:
            payload = json.load(f)
        response = EvaluationResponse(**{k: payload.get(k) for k in EvaluationResponse.__dataclass_fields__ if k in payload})
        response.elapsed_seconds = float(response.elapsed_seconds or (time.time() - start))
        response.response_path = str(response_path)

        if not self.debug_keep_requests:
            try:
                checkpoint_path.unlink(missing_ok=True)
            except TypeError:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
        if response.status != "ok":
            raise RuntimeError(f"Remote evaluation failed for {request_id}: {response.error}")
        return response

    def _build_request(self, request_id: str, epoch: int, metadata: dict) -> EvaluationRequest:
        metadata = dict(metadata)
        if metadata.get("flow_steps") is None and metadata.get("n_t") is not None:
            metadata["flow_steps"] = metadata["n_t"]
        fields = set(EvaluationRequest.__dataclass_fields__)
        payload = {k: metadata.get(k) for k in fields if k in metadata}
        payload.update(
            request_id=request_id,
            stream_id=self.stream_id,
            source_machine=socket.gethostname(),
            source_run_id=self.source_run_id,
            epoch=int(epoch),
            render_video=bool(metadata.get("render_video", self.render_best)),
            capture_rollouts=bool(metadata.get("capture_rollouts", self.render_best)),
            normalization_stats=_stats_to_json(metadata.get("normalization_stats")),
        )
        missing = [k for k in fields if k not in payload or payload[k] is None]
        optional = {
            "source_run_id", "gripper_idx", "normalization_stats", "dp_T_diff",
            "dp_schedule_type", "dp_ddim_steps", "dp_eta", "dp_pred_type",
            "dp_clip_sample", "dp_clip_sample_range", "latent_dim",
            "latent_hidden_dim", "latent_num_layers", "latent_residual",
        }
        missing = [k for k in missing if k not in optional]
        if missing:
            raise ValueError(f"Remote evaluation metadata is missing required field(s): {missing}")
        return EvaluationRequest(**payload)

    def _remote(self, local_dir: Path) -> tuple[str, str, str]:
        host = self.config["host"]
        user = self.config.get("user")
        target = f"{user}@{host}" if user else host
        remote_root = self.config["remote_root"].rstrip("/")
        remote_dir = f"{remote_root}/{local_dir.name}"
        return target, remote_root, remote_dir

    def _run_direct(self, request_dir: Path) -> None:
        target, _remote_root, remote_dir = self._remote(request_dir)
        evaluator_workspace = self.config["evaluator_workspace"]
        conda_env = self.config.get("conda_env")
        python_bin = self.config.get("python", "python")
        ssh_opts = list(self.config.get("ssh_opts", []))
        scp_opts = list(self.config.get("scp_opts", []))

        self._run(["ssh", *ssh_opts, target, "mkdir", "-p", remote_dir])
        self._run(["scp", *scp_opts, str(request_dir / "candidate_model.pt"), str(request_dir / "eval_request.json"), f"{target}:{remote_dir}/"])
        remote_cmd = (
            f"cd {self._sh_quote(evaluator_workspace)} && "
            + (f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {self._sh_quote(conda_env)} && " if conda_env else "")
            + f"{self._sh_quote(python_bin)} -m Robot_simulation.evaluate_checkpoint "
            + f"--request {self._sh_quote(remote_dir + '/eval_request.json')} "
            + f"--checkpoint {self._sh_quote(remote_dir + '/candidate_model.pt')} "
            + f"--output {self._sh_quote(remote_dir + '/eval_response.json')}"
        )
        self._run(["ssh", *ssh_opts, target, "bash", "-lc", remote_cmd], timeout=self.timeout_sec)
        self._run(["scp", *scp_opts, f"{target}:{remote_dir}/eval_response.json", str(request_dir / "eval_response.json")])

    def _submit_queue(self, request_dir: Path) -> None:
        queue_dir = self.config.get("queue_dir")
        if queue_dir is None:
            raise ValueError("remote_eval_mode=queue requires queue_dir in remote_eval_config")
        queue_path = Path(queue_dir)
        queue_request = queue_path / "pending" / request_dir.name
        queue_request.mkdir(parents=True, exist_ok=True)
        shutil.copy2(request_dir / "candidate_model.pt", queue_request / "candidate_model.pt")
        shutil.copy2(request_dir / "eval_request.json", queue_request / "eval_request.json")
        response_src = queue_path / "done" / request_dir.name / "eval_response.json"
        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            if response_src.exists():
                shutil.copy2(response_src, request_dir / "eval_response.json")
                return
            time.sleep(self.poll_interval_sec)
        raise TimeoutError(f"Timed out waiting for queued evaluation response: {response_src}")

    def _wait_for_file(self, path: Path) -> None:
        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            if path.exists():
                return
            time.sleep(self.poll_interval_sec)
        raise TimeoutError(f"Timed out waiting for {path}")

    def _run(self, cmd: list[str], timeout: int | None = None) -> None:
        subprocess.run(cmd, check=True, timeout=timeout)

    @staticmethod
    def _sh_quote(value: str) -> str:
        return "'" + str(value).replace("'", "'\"'\"'") + "'"


def response_from_exception(request: EvaluationRequest, exc: BaseException, elapsed: float, output: str | None) -> EvaluationResponse:
    return EvaluationResponse(
        request_id=request.request_id,
        stream_id=request.stream_id,
        status="error",
        epoch=request.epoch,
        success_rate=None,
        avg_reward=None,
        eval_trials=request.val_trials,
        elapsed_seconds=elapsed,
        response_path=output,
        video_path=None,
        rollout_path=None,
        error=repr(exc),
    )

