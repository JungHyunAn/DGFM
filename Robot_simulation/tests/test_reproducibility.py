"""Lightweight reproducibility checks; no simulator, training, or sweep is run."""

from __future__ import annotations

import random
import unittest
from unittest import mock

import numpy as np
import torch

from Robot_simulation.reproducibility import (
    environment_grid_episode_indices,
    environment_grid_episode_order,
    environment_grid_sampler_metadata,
    episode_seed_plan,
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

    @staticmethod
    def _door_grid_parameters(cell_counts):
        lower = np.asarray([-0.075, -0.3, -(np.pi / 2 + np.pi / 8)])
        upper = np.asarray([0.075, -0.1, -np.pi / 2])
        rows = []
        labels = []
        for coordinate, count in cell_counts.items():
            center = lower + (np.asarray(coordinate) + 0.5) * (upper - lower) / 3.0
            rows.extend([center] * count)
            labels.extend([coordinate] * count)
        return np.asarray(rows, dtype=np.float64).reshape(-1, 3), labels

    def test_grid_prefixes_are_deterministic_nested_unique_and_method_independent(self):
        parameters, _ = self._door_grid_parameters({
            coordinate: 4
            for coordinate in np.ndindex(3, 3, 3)
        })
        selections = {
            budget: environment_grid_episode_indices(
                parameters, 1000, budget, "door"
            )
            for budget in (20, 40, 80)
        }

        self.assertEqual(len(selections[20]), 20)
        self.assertEqual(len(selections[40]), 40)
        self.assertEqual(len(selections[80]), 80)
        self.assertEqual(len(selections[80]), len(set(selections[80])))
        self.assertLess(set(selections[20]), set(selections[40]))
        self.assertLess(set(selections[40]), set(selections[80]))
        self.assertEqual(selections[20], selections[40][:20])
        self.assertEqual(selections[40], selections[80][:40])
        self.assertEqual(
            environment_grid_episode_indices(parameters, 1000, 80, "door"),
            selections[80],
        )
        different_seed = environment_grid_episode_indices(
            parameters, 2000, 80, "door"
        )
        self.assertNotEqual(different_seed, selections[80])
        self.assertEqual(len(different_seed), len(set(different_seed)))

        for method in ("UniformFM", "DP", "DGFMv2"):
            random.random()
            np.random.random()
            torch.rand(1)
            received = environment_grid_episode_indices(
                parameters, 1000, 80, "door"
            )
            self.assertEqual(received, selections[80], method)

    def test_grid_round_robin_balances_sparse_occupied_cells(self):
        cell_counts = {(0, 0, 0): 4, (1, 1, 1): 2, (2, 2, 2): 1}
        parameters, episode_cells = self._door_grid_parameters(cell_counts)
        ordering = environment_grid_episode_order(parameters, "door", 1234)
        ordered_cells = [episode_cells[index] for index in ordering]

        self.assertEqual(set(ordered_cells[:3]), set(cell_counts))
        self.assertEqual(len(set(ordered_cells[:3])), 3)
        self.assertEqual(set(ordered_cells[3:5]), {(0, 0, 0), (1, 1, 1)})
        self.assertEqual(ordered_cells[5:], [(0, 0, 0), (0, 0, 0)])
        self.assertEqual(len(ordering), len(set(ordering)))

        metadata = environment_grid_sampler_metadata(
            parameters, "door", 1234, ordering
        )
        self.assertEqual(metadata["bins_per_dimension"], 3)
        self.assertEqual(metadata["total_grid_cells"], 27)
        self.assertEqual(metadata["occupied_grid_cells"], 3)
        self.assertEqual(metadata["episode_order_length"], 7)

    def test_grid_handles_empty_dataset_and_zero_parameter_dimensions(self):
        empty = np.empty((0, 3), dtype=np.float64)
        self.assertEqual(environment_grid_episode_order(empty, "door", 5), [])
        self.assertEqual(
            environment_grid_episode_indices(empty, 5, 0, "door"), []
        )

        parameterless = np.empty((7, 0), dtype=np.float64)
        ordering = environment_grid_episode_order(parameterless, "wipe", 5)
        self.assertEqual(len(ordering), 7)
        self.assertEqual(len(ordering), len(set(ordering)))
        self.assertEqual(
            environment_grid_sampler_metadata(parameterless, "wipe", 5, ordering)[
                "occupied_grid_cells"
            ],
            1,
        )

    def test_two_arm_yaw_wraps_into_configured_grid_range(self):
        lower = np.asarray([-0.015, -0.015, np.pi - np.pi / 6])
        upper = np.asarray([0.015, 0.015, np.pi + np.pi / 6])
        coordinates = np.asarray(list(np.ndindex(3, 3, 3)))
        parameters = lower + (coordinates + 0.5) * (upper - lower) / 3.0
        parameters[:, 2] = (parameters[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

        ordering = environment_grid_episode_order(parameters, "two_arm", 99)
        metadata = environment_grid_sampler_metadata(
            parameters, "two_arm", 99, ordering
        )
        self.assertEqual(len(ordering), 27)
        self.assertEqual(metadata["occupied_grid_cells"], 27)

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
