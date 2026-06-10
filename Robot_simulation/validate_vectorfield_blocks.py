"""Validate VectorField convolution-block capacity settings."""

import torch

from Robot_simulation.models.VanillaFM_class import VectorField


def _check_vectorfield(num_convs_per_block: int) -> None:
    batch_size = 2
    seq_len = 16
    dof = 8
    param_len = 12

    model = VectorField(
        seq_len,
        dof,
        param_len,
        gripper_idx=[7],
        num_convs_per_block=num_convs_per_block,
    ).eval()
    x = torch.randn(batch_size, seq_len, dof)
    t = torch.rand(batch_size, 1)
    env_params = torch.randn(batch_size, param_len)

    with torch.no_grad():
        out = model(x, t, env_params)

    expected_shape = (batch_size, seq_len, dof)
    if out.shape != expected_shape:
        raise AssertionError(
            f"num_convs_per_block={num_convs_per_block} produced {tuple(out.shape)}, "
            f"expected {expected_shape}"
        )


def main() -> None:
    _check_vectorfield(num_convs_per_block=1)  # door/default architecture
    _check_vectorfield(num_convs_per_block=2)  # nut architecture-capacity ablation
    print("VectorField block validation passed.")


if __name__ == "__main__":
    main()
