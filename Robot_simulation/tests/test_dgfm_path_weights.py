import torch

from Robot_simulation.models.DGFM_class import DGFM


def test_beizer_path_weights_and_derivatives() -> None:
    dgfm = DGFM.__new__(DGFM)
    t = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64)

    a, b, c, a_dot, b_dot, c_dot = dgfm._path_weights(t, "beizer")

    torch.testing.assert_close(a, (1.0 - t).square())
    torch.testing.assert_close(b, 2.0 * (1.0 - t) * t)
    torch.testing.assert_close(c, t.square())
    torch.testing.assert_close(a_dot, 2.0 * (t - 1.0))
    torch.testing.assert_close(b_dot, 2.0 * (1.0 - 2.0 * t))
    torch.testing.assert_close(c_dot, 2.0 * t)
    torch.testing.assert_close(a + b + c, torch.ones_like(t))
    torch.testing.assert_close(a_dot + b_dot + c_dot, torch.zeros_like(t))


def test_beizer_path_has_expected_endpoints_and_midpoint() -> None:
    dgfm = DGFM.__new__(DGFM)
    t = torch.tensor([[0.0], [0.5], [1.0]])

    a, b, c, *_ = dgfm._path_weights(t, "beizer")

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.5, 0.25],
            [0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(torch.cat([a, b, c], dim=1), expected)
