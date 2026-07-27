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
from Robot_simulation.models.DP_class import randn_per_sample, run_diffusion
from Robot_simulation.reproducibility import (
    DP_EVAL_POLICY_SEED_SCHEME,
    DP_EVAL_SAMPLING_MODE,
    evaluation_policy_seed_plan,
    summarize_validation_records,
)
from Robot_simulation.run_eval_sweep import method_metadata_compatible


class ZeroDiffusionModel(nn.Module):
    def forward(self, x, t, c):
        return torch.zeros_like(x)


class CountingConditionedDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.batch_sizes = []

    def forward(self, x, t, c):
        self.forward_calls += 1
        self.batch_sizes.append(x.shape[0])
        return 0.125 * x + 0.01 * c[:, :1].view(-1, 1, 1)


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

    def test_dp_batched_forward_count_and_per_trial_independence(self):
        conditions = np.arange(4, dtype=np.float32).reshape(-1, 1)
        seeds = evaluation_policy_seed_plan(555, 4)

        model = CountingConditionedDiffusionModel()
        full = _run_diffusion_batched(
            model, "door", conditions, 4, 2, "cpu", np.random.RandomState(0),
            T_diff=7, eta=1.0, pred_type="epsilon", clip_sample=False,
            policy_generators=[_make_eval_torch_generator("cpu", seed) for seed in seeds],
        )
        self.assertEqual(model.forward_calls, 7)
        self.assertEqual(model.batch_sizes, [4] * 7)

        subset_indices = [0, 2, 3]
        subset_model = CountingConditionedDiffusionModel()
        subset = _run_diffusion_batched(
            subset_model, "door", conditions[subset_indices], 4, 2, "cpu",
            np.random.RandomState(0), T_diff=7, eta=1.0, pred_type="epsilon",
            clip_sample=False, policy_generators=[
                _make_eval_torch_generator("cpu", seeds[idx]) for idx in subset_indices
            ],
        )
        self.assertTrue(np.array_equal(full[subset_indices], subset))
        self.assertEqual(subset_model.forward_calls, 7)
        self.assertEqual(subset_model.batch_sizes, [3] * 7)

        changed_seeds = list(seeds)
        changed_seeds[1] += 1
        changed = _run_diffusion_batched(
            CountingConditionedDiffusionModel(), "door", conditions, 4, 2, "cpu",
            np.random.RandomState(0), T_diff=7, eta=1.0, pred_type="epsilon",
            clip_sample=False, policy_generators=[
                _make_eval_torch_generator("cpu", seed) for seed in changed_seeds
            ],
        )
        self.assertTrue(np.array_equal(full[[0, 2, 3]], changed[[0, 2, 3]]))
        self.assertFalse(np.array_equal(full[1], changed[1]))

    def test_dp_noise_stacking_and_generators_persist_across_calls(self):
        seeds = evaluation_policy_seed_plan(777, 3)
        generators = [_make_eval_torch_generator("cpu", seed) for seed in seeds]
        first = randn_per_sample(generators, (4, 2), dtype=torch.float32, device="cpu")
        second = randn_per_sample(generators, (4, 2), dtype=torch.float32, device="cpu")

        recreated = [_make_eval_torch_generator("cpu", seed) for seed in seeds]
        expected_first = torch.cat([
            torch.randn((1, 4, 2), generator=generator) for generator in recreated
        ])
        expected_second = torch.cat([
            torch.randn((1, 4, 2), generator=generator) for generator in recreated
        ])
        self.assertTrue(torch.equal(first, expected_first))
        self.assertTrue(torch.equal(second, expected_second))
        self.assertFalse(torch.equal(first, second))

        subset = randn_per_sample(
            [_make_eval_torch_generator("cpu", seeds[idx]) for idx in [0, 2]],
            (4, 2), dtype=torch.float32, device="cpu",
        )
        self.assertTrue(torch.equal(first[[0, 2]], subset))

    def test_dp_eta_one_uses_isolated_reverse_noise(self):
        x_t = torch.ones((1, 4, 2), dtype=torch.float32)
        c = torch.zeros((1, 1), dtype=torch.float32)
        deterministic_a = run_diffusion(
            ZeroDiffusionModel(), x_t, c, "cpu", T_diff=6, eta=0.0,
            pred_type="epsilon", clip_sample=False,
            generator=_make_eval_torch_generator("cpu", 1),
        )
        deterministic_b = run_diffusion(
            ZeroDiffusionModel(), x_t, c, "cpu", T_diff=6, eta=0.0,
            pred_type="epsilon", clip_sample=False,
            generator=_make_eval_torch_generator("cpu", 2),
        )
        stochastic_a = run_diffusion(
            ZeroDiffusionModel(), x_t, c, "cpu", T_diff=6, eta=1.0,
            pred_type="epsilon", clip_sample=False,
            generator=_make_eval_torch_generator("cpu", 1),
        )
        stochastic_b = run_diffusion(
            ZeroDiffusionModel(), x_t, c, "cpu", T_diff=6, eta=1.0,
            pred_type="epsilon", clip_sample=False,
            generator=_make_eval_torch_generator("cpu", 2),
        )
        self.assertTrue(torch.equal(deterministic_a, deterministic_b))
        self.assertFalse(torch.equal(stochastic_a, stochastic_b))

    def test_dp_evaluation_does_not_change_subsequent_training_update(self):
        def train_with_optional_eval(insert_eval):
            torch.manual_seed(20260727)
            model = nn.Linear(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            first_x = torch.randn(5, 3)
            first_y = torch.randn(5, 2)
            optimizer.zero_grad()
            torch.square(model(first_x) - first_y).mean().backward()
            optimizer.step()
            if insert_eval:
                _run_diffusion_batched(
                    ZeroDiffusionModel(), "door", np.zeros((2, 1), dtype=np.float32),
                    4, 2, "cpu", np.random.RandomState(0), T_diff=6, eta=1.0,
                    pred_type="epsilon", clip_sample=False, policy_generators=[
                        _make_eval_torch_generator("cpu", seed)
                        for seed in evaluation_policy_seed_plan(999, 2)
                    ],
                )
            second_x = torch.randn(5, 3)
            second_y = torch.randn(5, 2)
            optimizer.zero_grad()
            torch.square(model(second_x) - second_y).mean().backward()
            optimizer.step()
            return second_x, second_y, copy.deepcopy(model.state_dict())

        reference = train_with_optional_eval(False)
        evaluated = train_with_optional_eval(True)
        self.assertTrue(torch.equal(reference[0], evaluated[0]))
        self.assertTrue(torch.equal(reference[1], evaluated[1]))
        for name in reference[2]:
            self.assertTrue(torch.equal(reference[2][name], evaluated[2][name]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_dp_sampling_does_not_advance_global_cuda_rng(self):
        before = torch.cuda.get_rng_state().clone()
        _run_diffusion_batched(
            ZeroDiffusionModel().cuda(), "door", np.zeros((2, 1), dtype=np.float32),
            4, 2, "cuda", np.random.RandomState(0), T_diff=6, eta=1.0,
            pred_type="epsilon", clip_sample=False, policy_generators=[
                _make_eval_torch_generator("cuda", seed)
                for seed in evaluation_policy_seed_plan(888, 2)
            ],
        )
        after = torch.cuda.get_rng_state().clone()
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
            "dp_eval_sampling_mode": DP_EVAL_SAMPLING_MODE,
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
