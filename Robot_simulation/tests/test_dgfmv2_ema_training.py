"""Minimal DGFMv2 EMA training-path test with PCA and simulation mocked."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

import Robot_simulation.models.DGFMv2_class as module
from Robot_simulation.models.DGFMv2_class import DGFMv2


class TinyTrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x, t, c):
        return self.linear(x)


class DGFMv2EMATrainingTest(unittest.TestCase):
    def test_validation_and_best_copy_use_ema(self):
        model = TinyTrajectoryModel()
        trainer = DGFMv2(
            model, torch.optim.SGD(model.parameters(), lr=0.1), None,
            task_name="door", horizon=1, dof=2, condition_dim=1,
            device="cpu", use_ema=True,
        )
        target = torch.tensor([[[1.0, -1.0]], [[0.5, 0.25]]])
        conditions = torch.zeros((2, 1))
        captured_models = []

        def fake_interpolants(**kwargs):
            x = kwargs["target_trajectories"]
            count = x.shape[0]
            return (
                x.clone(),
                torch.full((count, 1), 0.5),
                torch.zeros_like(x),
                kwargs["conditions"].clone(),
                torch.arange(count),
            )

        def fake_eval(**kwargs):
            captured_models.append(kwargs["model"])
            return 0.5, 1.0, {"success": [], "failure": []}

        trainer._build_joint_interpolants = fake_interpolants
        with mock.patch.object(
            module, "cluster_points_x",
            return_value=([np.array([0, 1])], [np.array([0]), np.array([0])]),
        ), mock.patch.object(
            module, "compute_cluster_pca_fast_x_only",
            return_value=(
                np.zeros((1, 2), dtype=np.float32),
                np.eye(2, dtype=np.float32)[None],
                np.eye(2, dtype=np.float32)[None],
                np.ones(1, dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.ones((1, 1, 1), dtype=np.float32),
            ),
        ), mock.patch.object(
            module, "_generate_val_env",
            return_value=([{}], np.zeros((1, 1), dtype=np.float32)),
        ), mock.patch.object(module, "eval_model", side_effect=fake_eval):
            best, raw_last, records, _ = trainer.train(
                target, conditions, n_t=1, cluster_size=2, max_epochs=3,
                batch_size=2, val_period=3, val_trials=1, pca_n_jobs=1,
            )

        self.assertEqual(list(records), [3])
        self.assertIs(captured_models[0], trainer.ema.averaged_model)
        for saved_param, ema_param in zip(best.parameters(), trainer.ema.averaged_model.parameters()):
            self.assertTrue(torch.equal(saved_param, ema_param))
        self.assertTrue(any(
            not torch.equal(ema_param, raw_param)
            for ema_param, raw_param in zip(
                trainer.ema.averaged_model.parameters(), raw_last.parameters()
            )
        ))


if __name__ == "__main__":
    unittest.main()
