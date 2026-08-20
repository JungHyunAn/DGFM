"""Non-destructive fixture tests for clean_v2 method-aware resume."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from Robot_simulation.reproducibility import (
    DP_EVAL_POLICY_SEED_SCHEME,
    DP_EVAL_SAMPLING_MODE,
    dataset_fingerprint,
    environment_grid_episode_order,
    environment_grid_sampler_metadata,
    stable_hash,
)
from Robot_simulation.run_eval_sweep import (
    SHARED_CONFIG,
    apply_sweep_overrides,
    completed_result_path,
    config_to_cli_args,
)


class ResumeCompatibilityTest(unittest.TestCase):
    def make_fixture(self, root: Path, method: str, extra: dict | None = None):
        dataset_path = root / "dataset.hdf5"
        if not dataset_path.exists():
            with h5py.File(dataset_path, "w") as dataset:
                data = dataset.create_group("data")
                for idx in range(10):
                    episode = data.create_group(f"entire_episode_{idx}")
                    parameters = episode.create_group("environment_parameters")
                    parameters.create_dataset(
                        "values",
                        data=np.asarray(
                            [-0.07 + 0.014 * idx, -0.29 + 0.018 * idx, -1.95 + 0.035 * idx],
                            dtype=np.float32,
                        ),
                    )
        results_path = root / method
        run_path = results_path / "run"
        run_path.mkdir(parents=True, exist_ok=True)
        config = {
            "results_path": str(results_path),
            "dataset_path": str(dataset_path),
            "FM_type": method,
            "task_name": "door",
            "N": 4,
            "seed": 1000,
            "max_epochs": 20,
            "warmup_steps": 4,
            "use_ema": True,
        }
        config["sweep_config_sha256"] = stable_hash(config)
        signature = {"fixture": method}
        fixture_parameters = np.asarray([
            [-0.07 + 0.014 * idx, -0.29 + 0.018 * idx, -1.95 + 0.035 * idx]
            for idx in range(10)
        ], dtype=np.float32)
        fixture_order = environment_grid_episode_order(
            fixture_parameters, "door", 1000
        )
        fixture_sampler = environment_grid_sampler_metadata(
            fixture_parameters, "door", 1000, fixture_order
        )
        fixture_sampler["dataset_base_seed"] = None
        fixture_sampler["selected_prefix_length"] = 4
        result = {
            "experiment_version": "clean_v2",
            "sweep_config_sha256": config["sweep_config_sha256"],
            "run_signature": signature,
            "run_signature_sha256": stable_hash(signature),
            "dataset_sha256": dataset_fingerprint(dataset_path)["dataset_sha256"],
            "selected_episode_indices": fixture_order[:4],
            "episode_sampler": fixture_sampler,
            "model_type": method,
            "task_name": "door",
            "N": 4,
            "seed": 1000,
            "maximum epoch": 20,
            "warmup steps": 4,
            "use_ema": True,
        }
        result.update(extra or {})
        with (run_path / "results.json").open("w") as stream:
            json.dump(result, stream)
        return config

    def test_legacy_uniform_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.make_fixture(Path(tmp), "UniformFM")
            self.assertIsNotNone(completed_result_path(config))

    def test_prefix_dp_and_ngfm_are_rejected_then_corrected_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dp = self.make_fixture(root, "DP")
            self.assertIsNone(completed_result_path(dp))
            dp = self.make_fixture(root, "DP", {
                "ema_applied": True,
                "dp_eta": 1.0,
                "eval_policy_rng_isolated": True,
                "eval_policy_seed_scheme": DP_EVAL_POLICY_SEED_SCHEME,
            })
            self.assertIsNone(completed_result_path(dp))
            dp = self.make_fixture(root, "DP", {
                "ema_applied": True,
                "dp_eta": 1.0,
                "eval_policy_rng_isolated": True,
                "eval_policy_seed_scheme": DP_EVAL_POLICY_SEED_SCHEME,
                "dp_eval_sampling_mode": DP_EVAL_SAMPLING_MODE,
            })
            self.assertIsNotNone(completed_result_path(dp))

            ngfm = self.make_fixture(root, "DGFMv2")
            self.assertIsNone(completed_result_path(ngfm))
            ngfm = self.make_fixture(root, "DGFMv2", {
                "ema_applied": True,
                "pca_covariance_mode": "full_rank_regularized",
                "pca_rank_truncation": False,
                "cluster_partition": 5,
                "residual_lambda": 0.2,
            })
            ngfm["cluster_partition"] = 5
            ngfm["residual_lambda"] = 0.2
            self.assertIsNotNone(completed_result_path(ngfm))

    def test_vision_augmentation_defaults_and_cli_overrides(self):
        self.assertIs(SHARED_CONFIG["vision_aug"], True)
        self.assertEqual(SHARED_CONFIG["vision_feature_norm"], "none")

        args = SimpleNamespace(
            validation_backend=None,
            remote_eval_config=None,
            remote_eval_mode=None,
            remote_eval_render_best=False,
            remote_eval_timeout_sec=None,
            remote_eval_poll_interval_sec=None,
            vision_aug=False,
            vision_random_shift=2,
            vision_color_jitter=0.05,
        )
        base_config = {
            "vision_aug": True,
            "vision_random_shift": 4,
            "vision_color_jitter": 0.1,
        }

        overridden = apply_sweep_overrides(base_config, args)
        cli_args = config_to_cli_args(overridden)

        self.assertEqual(base_config["vision_random_shift"], 4)
        self.assertIs(overridden["vision_aug"], False)
        self.assertEqual(overridden["vision_random_shift"], 2)
        self.assertEqual(overridden["vision_color_jitter"], 0.05)
        self.assertIn("--no-vision_aug", cli_args)
        self.assertEqual(
            cli_args[cli_args.index("--vision_random_shift") + 1],
            "2",
        )


if __name__ == "__main__":
    unittest.main()
