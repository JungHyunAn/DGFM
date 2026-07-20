"""Compatibility imports for the refactored synthetic-data models.

New code should import from Synthetic_data.models. The former GFM and LFM
implementations were intentionally removed; shifted-time FM is now selected
with UniformFM(..., time_sampling="shifted").
"""

from Synthetic_data.models import DGFM, DGFMv2, OT_CFM, UniformFM, VectorField, run_flow

__all__ = [
    "DGFM",
    "DGFMv2",
    "OT_CFM",
    "UniformFM",
    "VectorField",
    "run_flow",
]
