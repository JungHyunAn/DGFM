"""Evaluate a saved RoboSuite policy checkpoint from a JSON request."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import torch

from Robot_simulation.env_util import _generate_val_env, eval_model
from Robot_simulation.evaluators import (
    EvaluationRequest,
    EvaluationResponse,
    _stats_from_json,
    response_from_exception,
)
from Robot_simulation.environments.heuristics_util import (
    DEFAULT_VISION_CAMERAS,
    DEFAULT_VISION_HEIGHT,
    DEFAULT_VISION_WIDTH,
)
from Robot_simulation.models.LatentFM_class import LatentFlowPolicy, LatentVectorField, TrajectoryAutoencoder
from Robot_simulation.models.VanillaFM_class import VectorField
from Robot_simulation.models.vision_encoder import FrozenResNet18Encoder


def _load_request(path: str) -> EvaluationRequest:
    with open(path, "r") as f:
        payload = json.load(f)
    return EvaluationRequest(**payload)


def _build_vision_encoder(request: EvaluationRequest):
    if request.observation_type != "vision":
        return None
    return FrozenResNet18Encoder(
        request.camera_names or DEFAULT_VISION_CAMERAS,
        finetune=request.vision_finetune,
        finetune_mode=request.vision_finetune_mode,
        train_bn=request.vision_train_bn,
        pool=request.vision_pool,
        spatial_softmax_temperature=request.vision_spatial_softmax_temperature,
        feature_proj_dim=request.vision_feature_proj_dim,
        feature_norm=request.vision_feature_norm,
        augmentation=request.vision_aug,
        random_shift=request.vision_random_shift,
        color_jitter=request.vision_color_jitter,
    )


def _build_model(request: EvaluationRequest, device: torch.device):
    if request.model_type == "LatentFM":
        if request.latent_dim is None:
            raise ValueError("LatentFM remote evaluation requires latent_dim metadata")
        hidden_dim = request.latent_hidden_dim or 256
        autoencoder = TrajectoryAutoencoder(request.seq_len, request.dof, request.latent_dim, hidden_dim=hidden_dim).to(device)
        latent_model = LatentVectorField(
            request.latent_dim,
            request.param_len,
            hidden_dim=hidden_dim,
            num_layers=request.latent_num_layers or 4,
            residual=bool(request.latent_residual),
        ).to(device)
        model = LatentFlowPolicy(
            autoencoder,
            latent_model,
            horizon=request.seq_len,
            dof=request.dof,
            latent_dim=request.latent_dim,
            condition_dim=request.param_len,
            device=device,
        ).to(device)
    else:
        model = VectorField(
            request.seq_len,
            request.dof,
            request.param_len,
            gripper_idx=request.gripper_idx,
            num_convs_per_block=request.num_convs_per_block,
            observation_type=request.observation_type,
            condition_embed_dim=request.condition_embed_dim,
        ).to(device)
        vision_encoder = _build_vision_encoder(request)
        if vision_encoder is not None:
            model.vision_encoder = vision_encoder

    model.observation_type = request.observation_type
    model.camera_names = tuple(request.camera_names or DEFAULT_VISION_CAMERAS)
    model.state_condition_dim = request.state_condition_dim
    model.vision_image_height = DEFAULT_VISION_HEIGHT
    model.vision_image_width = DEFAULT_VISION_WIDTH
    return model


def evaluate_checkpoint(request_path: str, checkpoint_path: str, output_path: str) -> EvaluationResponse:
    request = _load_request(request_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rollout_path = None
    video_path = None

    try:
        device = torch.device(request.device if str(request.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        model = _build_model(request, device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        env_settings_all, val_params = _generate_val_env(request.task_name, request.val_trials)
        render_dir = output.parent if request.render_video else " "
        video_name = f"{request.stream_id}_{request.request_id}"
        sampler_type = "diffusion" if request.model_type == "DP" else "flow"
        result = eval_model(
            model=model,
            model_class=None,
            task_name=request.task_name,
            seq_len=request.seq_len,
            dof=request.dof,
            param_len=request.param_len,
            gripper_idx=request.gripper_idx,
            render_dir=str(render_dir),
            video_name=video_name if request.render_video else None,
            val_params=val_params,
            env_settings_all=env_settings_all,
            device=str(device),
            trials=request.val_trials,
            num_workers=5 if request.task_name == "two_arm" else 10,
            render_width=4 if request.render_video else 0,
            render_num=8 if request.render_video else 0,
            base_seed=request.eval_base_seed,
            max_policy_steps=request.max_policy_steps,
            executed_horizon=request.executed_horizon,
            observation_horizon=request.observation_horizon,
            flow_steps=request.flow_steps,
            recorded_control_freq=request.recorded_control_freq,
            trajectory_control_freq=request.trajectory_control_freq,
            normalization_stats=_stats_from_json(request.normalization_stats),
            sampler_type=sampler_type,
            T_diff=request.dp_T_diff or 100,
            schedule_type=request.dp_schedule_type or "cosine",
            ddim_steps=request.dp_ddim_steps,
            eta=request.dp_eta or 0.0,
            pred_type=request.dp_pred_type or "x0",
            clip_sample=True if request.dp_clip_sample is None else request.dp_clip_sample,
            clip_sample_range=request.dp_clip_sample_range or 1.0,
            return_rollouts=request.capture_rollouts,
            action_representation=request.action_representation,
        )
        if request.capture_rollouts:
            success_rate, avg_reward, validation_rollouts = result
            rollout_path = str(output.parent / "validation_rollouts.pkl")
            with open(rollout_path, "wb") as f:
                pickle.dump(validation_rollouts, f)
        else:
            success_rate, avg_reward = result

        if request.render_video:
            matches = sorted(output.parent.glob(f"{request.task_name}_grid_{video_name}*.mp4"))
            if matches:
                video_path = str(matches[-1])

        response = EvaluationResponse(
            request_id=request.request_id,
            stream_id=request.stream_id,
            status="ok",
            epoch=request.epoch,
            success_rate=float(success_rate),
            avg_reward=float(avg_reward),
            eval_trials=request.val_trials,
            elapsed_seconds=time.time() - start,
            response_path=str(output),
            video_path=video_path,
            rollout_path=rollout_path,
            error=None,
        )
    except BaseException as exc:
        response = response_from_exception(request, exc, time.time() - start, str(output))

    with output.open("w") as f:
        json.dump(dataclasses.asdict(response), f, indent=2)
    return response


def worker_loop(queue_dir: str, poll_interval_sec: float = 5.0, once: bool = False) -> None:
    queue = Path(queue_dir)
    pending = queue / "pending"
    running = queue / "running"
    done = queue / "done"
    failed = queue / "failed"
    for path in (pending, running, done, failed):
        path.mkdir(parents=True, exist_ok=True)

    while True:
        jobs = sorted(p for p in pending.iterdir() if p.is_dir())
        if not jobs:
            if once:
                return
            time.sleep(poll_interval_sec)
            continue
        job = jobs[0]
        active = running / job.name
        try:
            job.rename(active)
            evaluate_checkpoint(
                str(active / "eval_request.json"),
                str(active / "candidate_model.pt"),
                str(active / "eval_response.json"),
            )
            target = done / active.name
            if target.exists():
                shutil.rmtree(target)
            active.rename(target)
        except BaseException:
            target = failed / job.name
            source = active if active.exists() else job
            if target.exists():
                shutil.rmtree(target)
            if source.exists():
                source.rename(target)
            if once:
                raise
        if once:
            return


def main() -> None:
    parser = argparse.ArgumentParser("evaluate_checkpoint")
    parser.add_argument("--request", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--queue_dir", type=str, default=None)
    parser.add_argument("--poll_interval_sec", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.queue_dir is not None:
        worker_loop(args.queue_dir, poll_interval_sec=args.poll_interval_sec, once=args.once)
        return
    missing = [name for name in ("request", "checkpoint", "output") if getattr(args, name) is None]
    if missing:
        parser.error(f"Missing required option(s): {', '.join('--' + name for name in missing)}")
    response = evaluate_checkpoint(args.request, args.checkpoint, args.output)
    if response.status != "ok":
        raise SystemExit(response.error or "evaluation failed")


if __name__ == "__main__":
    main()

