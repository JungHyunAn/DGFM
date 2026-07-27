"""Numerical contract for intentional full-rank regularized local PCA."""

from __future__ import annotations

import unittest

import numpy as np

from Robot_simulation.models.DGFMv2_class import compute_cluster_pca_fast_x_only


class FullRankPCATest(unittest.TestCase):
    def test_basis_and_covariance_retain_ambient_dimension(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(12, 6)).astype(np.float32)
        c = rng.normal(size=(12, 2)).astype(np.float32)
        clusters = [np.arange(6), np.arange(6, 12)]
        _, basis, covariance, _, _, _ = compute_cluster_pca_fast_x_only(
            x, c, clusters, eps=1e-3, outlier_q=0.99,
            max_pca_samples=100, n_jobs=1,
        )
        self.assertEqual(basis.shape, (2, 6, 6))
        self.assertEqual(covariance.shape, (2, 6, 6))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))


if __name__ == "__main__":
    unittest.main()
