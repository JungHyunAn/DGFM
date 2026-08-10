"""Lightweight reproducibility checks; no simulator, training, or sweep is run."""

from __future__ import annotations

import random
import unittest
from unittest import mock

import numpy as np
import torch

from Robot_simulation.reproducibility import (
    environment_grid_episode_indices,
    episode_seed_plan,
    selected_episode_indices,
    stable_hash,
    validation_suite_spec,
)


class ReproducibilityHelpersTest(unittest.TestCase):
    def test_episode_randomness_is_unique_and_worker_count_invariant(self):
        episode_ids = list(range(200))
        expected = episode_seed_plan(12345, episode_ids)
        self.assertEqual(len(expected), len(set(expected)))

        for worker_count in (1, 2, 5, 13):
            partitions = [episode_ids[offset::worker_count] for offset in range(worker_count)]
            scheduled = {
                episode_id: seed
                for partition in reversed(partitions)
                for episode_id, seed in zip(
                    partition, episode_seed_plan(12345, partition), strict=True
                )
            }
            self.assertEqual([scheduled[idx] for idx in episode_ids], expected)
            self.assertEqual(len(scheduled), len(episode_ids))

        mock_settings = [
            np.random.default_rng(seed).uniform(size=3).tolist() for seed in expected
        ]
        self.assertEqual(
            mock_settings,
            [
                np.random.default_rng(seed).uniform(size=3).tolist()
                for seed in episode_seed_plan(12345, episode_ids)
            ],
        )

    def test_demo_budgets_are_nested_and_method_independent(self):
        for training_seed in (1000, 2000, 3000):
            selections = {
                budget: selected_episode_indices(1000, training_seed, budget)
                for budget in (20, 40, 80, 160)
            }
            self.assertLessEqual(set(selections[20]), set(selections[40]))
            self.assertLessEqual(set(selections[40]), set(selections[80]))
            self.assertLessEqual(set(selections[80]), set(selections[160]))
            for method in ("VanillaFM", "DP", "NGFM"):
                received = selected_episode_indices(1000, training_seed, 160)
                self.assertEqual(received, selections[160], method)

    def test_environment_grid_selection_is_nested_balanced_and_deterministic(self):
        cells = np.stack(
            np.unravel_index(np.arange(27), (3, 3, 3)), axis=1
        ).astype(np.float64)
        parameters = np.repeat(cells, 4, axis=0)

        selections = {
            budget: environment_grid_episode_indices(parameters, 1000, budget)
            for budget in (20, 40, 80)
        }

        self.assertEqual(
            environment_grid_episode_indices(parameters, 1000, 80),
            selections[80],
        )
        self.assertLessEqual(set(selections[20]), set(selections[40]))
        self.assertLessEqual(set(selections[40]), set(selections[80]))
        for budget, selected in selections.items():
            self.assertEqual(len(selected), len(set(selected)))
            selected_cells = np.asarray(selected) // 4
            counts = np.bincount(selected_cells, minlength=27)
            self.assertLessEqual(counts.max() - counts.min(), 1, budget)

    def test_environment_grid_selection_validates_fixed_bin_count(self):
        parameters = np.stack(
            np.unravel_index(np.arange(8), (2, 2, 2)), axis=1
        ).astype(np.float64)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            environment_grid_episode_indices(
                parameters, 1000, 7, bins_per_dimension=0
            )

    def test_validation_spec_is_method_and_training_seed_independent(self):
        expected = validation_suite_spec("two_arm", 50)
        for method in ("VanillaFM", "DP", "NGFM"):
            for training_seed in (1000, 2000, 3000):
                random.seed(training_seed)
                np.random.seed(training_seed)
                torch.manual_seed(training_seed)
                _ = (method, random.random(), np.random.random(), torch.rand(1))
                self.assertEqual(validation_suite_spec("two_arm", 50), expected)
        self.assertEqual(len(expected["eval_trial_seeds"]), 50)


class ValidationGeneratorDebugTest(unittest.TestCase):
    def test_shared_model_paths_produce_identical_ordered_fake_states(self):
        from Robot_simulation import env_util

        class Namespace:
            pass

        class FakeEnv:
            def __init__(self):
                self.sim = Namespace()
                self.sim.data = Namespace()
                self.sim.model = Namespace()
                self.reset()

            def seed(self, seed):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            def reset(self):
                self.sim.data.qpos = np.random.random(3)
                self.sim.data.qvel = np.random.random(2)
                self.sim.data.act = np.random.random(1)
                self.sim.data.ctrl = np.random.random(1)
                self.sim.data.mocap_pos = np.random.random((1, 3))
                self.sim.data.mocap_quat = np.random.random((1, 4))
                self.sim.model.body_pos = np.random.random((2, 3))
                self.sim.model.body_quat = np.random.random((2, 4))

            def close(self):
                pass

        def fake_params(env, task_name):
            return np.concatenate([env.sim.data.qpos, [len(task_name)]])

        suites = []
        with mock.patch.object(env_util, "make_env", side_effect=lambda *a, **k: FakeEnv()), \
             mock.patch.object(env_util, "_get_environment_params", side_effect=fake_params):
            for method, disturbance in (("VanillaFM", 11), ("DP", 29), ("NGFM", 47)):
                random.seed(disturbance)
                np.random.seed(disturbance)
                torch.manual_seed(disturbance)
                settings, params = env_util._generate_val_env("two_arm", 5)
                serialized = [
                    {key: value.tolist() for key, value in setting.items()}
                    for setting in settings
                ]
                suites.append((method, stable_hash({"settings": serialized, "params": params.tolist()})))

        self.assertEqual(len({suite_hash for _, suite_hash in suites}), 1, suites)


if __name__ == "__main__":
    unittest.main()
