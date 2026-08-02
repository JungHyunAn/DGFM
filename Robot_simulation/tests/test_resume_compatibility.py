"""Non-destructive fixture tests for clean_v1 method-aware resume."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from Robot_simulation.reproducibility import (
    DP_EVAL_POLICY_SEED_SCHEME,
    DP_EVAL_SAMPLING_MODE,
    dataset_fingerprint,
    environment_grid_episode_indices,
    stable_hash,
)
from Robot_simulation.run_eval_sweep import completed_result_path


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
                        "values", data=np.asarray([idx], dtype=np.float32)
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
        result = {
            "experiment_version": "clean_v1",
            "sweep_config_sha256": config["sweep_config_sha256"],
            "run_signature": signature,
            "run_signature_sha256": stable_hash(signature),
            "dataset_sha256": dataset_fingerprint(dataset_path)["dataset_sha256"],
            "selected_episode_indices": environment_grid_episode_indices(
                np.arange(10, dtype=np.float32).reshape(-1, 1), 1000, 4
            ),
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


if __name__ == "__main__":
    unittest.main()
