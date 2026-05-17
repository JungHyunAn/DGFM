"""Uniform-time Flow Matching trainer."""

from Robot_simulation.models.VanillaFM_class import VanillaFM


class UniformFM(VanillaFM):
    def __init__(self, *args, **kwargs):
        kwargs["time_sampling"] = "uniform"
        super().__init__(*args, **kwargs)
