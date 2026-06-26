"""Validate remote-evaluation request metadata construction."""

from pathlib import Path
from tempfile import TemporaryDirectory

from Robot_simulation.evaluators import RemoteEvaluator


def _base_metadata() -> dict:
    return {
        "stream_id": "metadata-smoke",
        "source_run_id": "metadata-smoke",
        "task_name": "door",
        "model_type": "UniformFM",
        "val_trials": 1,
        "eval_base_seed": 1,
        "flow_steps": 1,
        "n_t": 1,
        "n_t_global": 1,
        "n_t_local": 1,
        "seq_len": 16,
        "dof": 8,
        "param_len": 12,
        "state_condition_dim": 12,
        "gripper_idx": [7],
        "observation_type": "vision",
        "camera_names": ["frontview", "robot0_eye_in_hand"],
        "condition_embed_dim": 256,
        "vision_finetune": False,
        "vision_finetune_mode": "last_layer",
        "vision_train_bn": False,
        "vision_pool": "spatial_softmax",
        "vision_spatial_softmax_temperature": 1.0,
        "vision_feature_proj_dim": 128,
        "vision_feature_norm": "none",
        "vision_aug": False,
        "vision_random_shift": 0,
        "vision_color_jitter": 0.0,
        "max_policy_steps": 400,
        "executed_horizon": 16,
        "observation_horizon": 1,
        "recorded_control_freq": 20,
        "trajectory_control_freq": 20,
        "action_representation": "joint_space",
        "normalization_stats": None,
        "use_ema": False,
        "dp_T_diff": None,
        "dp_schedule_type": None,
        "dp_ddim_steps": None,
        "dp_eta": None,
        "dp_pred_type": None,
        "dp_clip_sample": None,
        "dp_clip_sample_range": None,
        "render_video": False,
        "capture_rollouts": False,
        "num_convs_per_block": 1,
        "device": "cuda",
        "latent_dim": None,
        "latent_hidden_dim": None,
        "latent_num_layers": None,
        "latent_residual": None,
    }


def _evaluator(exp_dir: str) -> RemoteEvaluator:
    return RemoteEvaluator(
        exp_dir=exp_dir,
        config={
            "host": "example.invalid",
            "remote_root": "/tmp/dgfm-remote-eval",
            "evaluator_workspace": "/tmp/dgfm",
        },
        mode="direct",
    )


def main() -> None:
    with TemporaryDirectory() as tmp:
        evaluator = _evaluator(tmp)
        request = evaluator._build_request("metadata-smoke-80", 80, _base_metadata())
        if request.flow_steps != 1:
            raise AssertionError(f"flow_steps={request.flow_steps}, expected 1")

        legacy_metadata = _base_metadata()
        legacy_metadata.pop("flow_steps")
        legacy_request = evaluator._build_request(
            "metadata-smoke-legacy-80",
            80,
            legacy_metadata,
        )
        if legacy_request.flow_steps != legacy_metadata["n_t"]:
            raise AssertionError(
                f"legacy flow_steps={legacy_request.flow_steps}, "
                f"expected {legacy_metadata['n_t']}"
            )

        if not Path(tmp, "remote_eval_requests").exists():
            raise AssertionError("RemoteEvaluator did not initialize request directory")

    print("Remote evaluation metadata validation passed.")


if __name__ == "__main__":
    main()
