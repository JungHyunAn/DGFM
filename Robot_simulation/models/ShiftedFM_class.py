"""Shifted-time Flow Matching trainer."""

from Robot_simulation.models.VanillaFM_class import VanillaFM


class ShiftedFM(VanillaFM):
    def __init__(self, *args, **kwargs):
        kwargs["time_sampling"] = "shifted"
        super().__init__(*args, **kwargs)
