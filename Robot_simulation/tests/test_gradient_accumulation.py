import copy

import torch
from torch import nn

from Robot_simulation.models.VanillaFM_class import (
    VanillaFM,
    normalize_accumulated_gradients,
)


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs) -> None:
        super().__init__(parameters, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class _TinyVectorField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, trajectory, time, condition):
        del time, condition
        return self.scale * trajectory


def test_sample_weighted_accumulation_matches_full_batch_with_partial_microbatch() -> None:
    torch.manual_seed(7)
    full_model = nn.Linear(3, 2)
    accumulated_model = copy.deepcopy(full_model)
    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 2)

    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
    full_optimizer.zero_grad()
    torch.mean((full_model(inputs) - targets) ** 2).backward()
    full_optimizer.step()

    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    accumulated_optimizer.zero_grad()
    accumulated_samples = 0
    for start, stop in ((0, 2), (2, 5)):
        microbatch_size = stop - start
        loss = torch.mean(
            (accumulated_model(inputs[start:stop]) - targets[start:stop]) ** 2
        )
        (loss * microbatch_size).backward()
        accumulated_samples += microbatch_size
    normalize_accumulated_gradients(accumulated_optimizer, accumulated_samples)
    accumulated_optimizer.step()

    for full_parameter, accumulated_parameter in zip(
        full_model.parameters(), accumulated_model.parameters()
    ):
        torch.testing.assert_close(full_parameter, accumulated_parameter)


def test_accumulation_rejects_empty_sample_group() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        normalize_accumulated_gradients(optimizer, 0)
    except ValueError as error:
        assert "sample_count must be positive" in str(error)
    else:
        raise AssertionError("Expected an empty accumulation group to be rejected")


def test_vanilla_trainer_steps_optimizer_and_ema_per_accumulation_group() -> None:
    model = _TinyVectorField()
    optimizer = _CountingSGD(model.parameters(), lr=0.01)
    trainer = VanillaFM(
        model,
        optimizer,
        scheduler=None,
        task_name="test",
        horizon=2,
        dof=1,
        condition_dim=1,
        device="cpu",
        use_ema=True,
    )
    trajectories = torch.randn(5, 2, 1)
    conditions = torch.randn(5, 1)

    trainer.train(
        trajectories,
        conditions,
        n_t=1,
        max_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=2,
        val_period=0,
        val_trials=0,
    )

    # Three physical microbatches form one full group and one final partial group.
    assert optimizer.step_count == 2
    assert trainer.ema.optimization_step == 2
