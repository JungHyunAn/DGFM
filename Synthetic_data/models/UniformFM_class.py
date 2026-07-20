"""Uniform or shifted-time Flow Matching for synthetic data."""

from Synthetic_data.models.VanillaFM_class import VanillaFM


class UniformFM(VanillaFM):
    """Vanilla FM with selectable uniform or shifted time sampling.

    ``time_sampling='shifted'`` replaces the former separate ShiftedFM method.
    """

    def __init__(self, *args, time_sampling: str = "uniform", **kwargs):
        super().__init__(*args, time_sampling=time_sampling, **kwargs)
