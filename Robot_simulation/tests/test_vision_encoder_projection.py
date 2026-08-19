import unittest

import torch
import torch.nn as nn

from Robot_simulation.models.vision_encoder import FrozenResNet18Encoder


class TestVisionEncoderProjection(unittest.TestCase):
    def _build_encoder(self, seed: int) -> FrozenResNet18Encoder:
        torch.manual_seed(seed)
        return FrozenResNet18Encoder(
            ["frontview"],
            pretrained=False,
            pool="spatial_softmax",
            spatial_softmax_temperature=1.0,
            feature_proj_dim=128,
        )

    def test_original_diffusion_policy_style_projection_shape(self):
        encoder = self._build_encoder(1000)

        self.assertEqual(encoder.spatial_softmax_num_keypoints, 32)
        self.assertIsInstance(encoder.spatial_keypoint_projection, nn.Conv2d)
        self.assertEqual(encoder.spatial_keypoint_projection.in_channels, 512)
        self.assertEqual(encoder.spatial_keypoint_projection.out_channels, 32)
        self.assertEqual(encoder.raw_feature_dim, 64)
        self.assertIsInstance(encoder.projection_head, nn.Linear)
        self.assertEqual(encoder.projection_head.in_features, 64)
        self.assertEqual(encoder.projection_head.out_features, 128)

        feature_map = torch.randn(2, 512, 7, 7)
        pooled = encoder._pool_features(feature_map)
        projected = encoder.projection_head(pooled)
        self.assertEqual(tuple(pooled.shape), (2, 64))
        self.assertEqual(tuple(projected.shape), (2, 128))

    def test_projection_parameter_count(self):
        encoder = self._build_encoder(1000)
        parameter_count = sum(
            parameter.numel() for parameter in encoder.projection_parameters()
        )
        self.assertEqual(parameter_count, 24_736)

    def test_projection_initialization_is_seed_reproducible(self):
        first = self._build_encoder(1000)
        second = self._build_encoder(1000)
        different = self._build_encoder(2000)

        first_parameters = torch.cat([
            parameter.detach().reshape(-1)
            for parameter in first.projection_parameters()
        ])
        second_parameters = torch.cat([
            parameter.detach().reshape(-1)
            for parameter in second.projection_parameters()
        ])
        different_parameters = torch.cat([
            parameter.detach().reshape(-1)
            for parameter in different.projection_parameters()
        ])

        self.assertTrue(torch.equal(first_parameters, second_parameters))
        self.assertFalse(torch.equal(first_parameters, different_parameters))


if __name__ == "__main__":
    unittest.main()
