"""Shared defaults for the Robot_simulation package."""

from pathlib import Path


_PREFERRED_DATASET_DIR = Path("/PublicHDD/ajh916/DGFM/Synthetic_data/heuristic_dataset")
_PREFERRED_RECORDS_DIR = Path("/PublicHDD/ajh916/DGFM/Synthetic_data/eval_results")

_LOCAL_ROOT = Path(__file__).resolve().parent
_LOCAL_DATASET_DIR = _LOCAL_ROOT / "heuristic_dataset"
_LOCAL_RECORDS_DIR = _LOCAL_ROOT / "eval_results"


DEFAULT_DATASET_DIR = str(
    _PREFERRED_DATASET_DIR if _PREFERRED_DATASET_DIR.exists() else _LOCAL_DATASET_DIR
)
DEFAULT_RECORDS_DIR = str(
    _PREFERRED_RECORDS_DIR if _PREFERRED_RECORDS_DIR.exists() else _LOCAL_RECORDS_DIR
)
