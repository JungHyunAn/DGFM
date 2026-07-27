"""Focused method-consistency tests without RoboSuite training or rollouts."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch
import torch.nn as nn

from Robot_simulation.env_util import _make_eval_torch_generator, _run_diffusion_batched
from Robot_simulation.environments.heuristics_util import is_two_arm_lift_success
from Robot_simulation.models.DGFMv2_class import DGFMv2
from Robot_simulation.reproducibility import (
    DP_EVAL_POLICY_SEED_SCHEME,
    evaluation_policy_seed_plan,
    summarize_validation_records,
)
from Robot_simulation.run_eval_sweep import method_metadata_compatible


class ZeroDiffusionModel(nn.Module):
    def forward(self, x, t, c):
        return torch.zeros_like(x)


class MethodConsistencyTest(unittest.TestCase):
    def test_dp_sampling_is_reproducible_stochastic_and_rng_isolated(self):
        model = ZeroDiffusionModel()
        conditions = np.zeros((2, 1), dtype=np.float32)

        def sample(base_seed):
            seeds = evaluation_policy_seed_plan(base_seed, len(conditions))
            generators = [_make_eval_torch_generator("cpu", seed) for seed in seeds]
            return _run_diffusion_batched(
                model, "door", conditions, 4, 2, "cpu",
                np.random.RandomState(999),
                T_diff=6, eta=1.0, pred_type="epsilon",
                clip_sample=False, policy_generators=generators,
            )

        torch.manual_seed(31415)
        before = torch.random.get_rng_state().clone()
        first = sample(123)
        after = torch.random.get_rng_state().clone()
        second = sample(123)
        different = sample(124)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, different))
        self.assertTrue(torch.equal(before, after))

    def test_dgfmv2_ema_models_are_independent_and_copy_eval_weights(self):
        raw = nn.Linear(2, 2, bias=False)
        optimizer = torch.optim.SGD(raw.parameters(), lr=0.1)
        trainer = DGFMv2(
            raw, optimizer, None, task_name="door", horizon=1, dof=2,
            condition_dim=1, device="cpu", use_ema=True,
        )
        trainer._init_ema()
        self.assertIsNot(trainer.model.weight, trainer.ema.averaged_model.weight)
        initial_ema = trainer.ema.averaged_model.weight.detach().clone()
        for delta in (1.0, 2.0, 3.0):
            with torch.no_grad():
                trainer.model.weight.add_(delta)
            trainer._step_ema()
        self.assertFalse(torch.equal(initial_ema, trainer.ema.averaged_model.weight))
        saved = trainer._copy_eval_model()
        self.assertTrue(torch.equal(saved.weight, trainer.ema.averaged_model.weight))
        self.assertFalse(torch.equal(saved.weight, trainer.model.weight))

        raw_only = copy.deepcopy(raw)
        raw_trainer = DGFMv2(
            raw_only, torch.optim.SGD(raw_only.parameters(), lr=0.1), None,
            task_name="door", horizon=1, dof=2, condition_dim=1,
            device="cpu", use_ema=False,
        )
        raw_trainer._init_ema()
        self.assertIsNone(raw_trainer.ema)
        self.assertTrue(torch.equal(raw_trainer._copy_eval_model().weight, raw_only.weight))

    def test_validation_summary_tail_sizes_and_reward_ties(self):
        records = {
            epoch: {"success_rate": epoch / 20.0, "avg_reward": float(epoch)}
            for epoch in range(1, 13)
        }
        summary = summarize_validation_records(records)
        self.assertEqual(summary["avg_success_rate_num_checkpoints"], 10)
        self.assertEqual(summary["final_validation_epochs"], list(range(3, 13)))
        self.assertAlmostEqual(
            summary["avg_success_rate"],
            np.mean([epoch / 20.0 for epoch in range(3, 13)]),
        )

        exact = summarize_validation_records(dict(list(records.items())[:10]))
        short = summarize_validation_records(dict(list(records.items())[:4]))
        self.assertEqual(exact["avg_success_rate_num_checkpoints"], 10)
        self.assertEqual(short["avg_success_rate_num_checkpoints"], 4)

        ties = {
            10: {"success_rate": 0.8, "avg_reward": 2.0},
            20: {"success_rate": 0.8, "avg_reward": 3.0},
            30: {"success_rate": 0.8, "avg_reward": 3.0},
        }
        tied = summarize_validation_records(ties)
        self.assertEqual(tied["best_validation_epoch"], 20)
        self.assertEqual(tied["best_validation_reward"], 3.0)

    def test_two_arm_strict_boundary(self):
        class Env:
            def __init__(self, success, difference):
                self.success = success
                self._handle0_xpos = np.array([0.0, 0.0, 0.5])
                self._handle1_xpos = np.array([0.0, 0.0, 0.5 + difference])

            def _check_success(self):
                return self.success

        self.assertFalse(is_two_arm_lift_success(Env(False, 0.01)))
        self.assertTrue(is_two_arm_lift_success(Env(True, 0.049)))
        self.assertFalse(is_two_arm_lift_success(Env(True, 0.05)))
        self.assertFalse(is_two_arm_lift_success(Env(True, 0.051)))

    def test_method_aware_resume_metadata(self):
        base = {"task_name": "door", "use_ema": True}
        uniform = {**base, "FM_type": "UniformFM"}
        self.assertTrue(method_metadata_compatible({"use_ema": True}, uniform))

        dp = {**base, "FM_type": "DP"}
        self.assertFalse(method_metadata_compatible({"use_ema": True}, dp))
        self.assertTrue(method_metadata_compatible({
            "use_ema": True, "ema_applied": True, "dp_eta": 1.0,
            "eval_policy_rng_isolated": True,
            "eval_policy_seed_scheme": DP_EVAL_POLICY_SEED_SCHEME,
        }, dp))

        ngfm = {**base, "FM_type": "DGFMv2"}
        self.assertFalse(method_metadata_compatible({
            "use_ema": True, "ema_applied": True,
        }, ngfm))
        self.assertTrue(method_metadata_compatible({
            "use_ema": True, "ema_applied": True,
            "pca_covariance_mode": "full_rank_regularized",
            "pca_rank_truncation": False,
        }, ngfm))

        two_arm = {"task_name": "two_arm", "FM_type": "UniformFM", "use_ema": True}
        self.assertFalse(method_metadata_compatible({"use_ema": True}, two_arm))
        self.assertTrue(method_metadata_compatible({
            "use_ema": True, "success_criterion": "two_arm_lift_strict_v1",
        }, two_arm))


if __name__ == "__main__":
    unittest.main()
