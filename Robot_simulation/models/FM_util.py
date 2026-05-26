"""Compatibility exports for policy evaluation utilities.

Evaluation helpers now live in :mod:`Robot_simulation.env_util`. This module is
kept so older training code that imports from ``models.FM_util`` continues to
work while new code imports directly from ``Robot_simulation.env_util``.
"""

import logging

logging.disable(logging.WARNING)
robosuite_logger = logging.getLogger("robosuite")
robosuite_logger.setLevel(logging.ERROR)
robosuite_logger.propagate = False
for h in list(robosuite_logger.handlers):
    robosuite_logger.removeHandler(h)

from Robot_simulation.env_util import (  # noqa: E402,F401
    _generate_val_env,
    _maybe_denormalize_policy_data,
    _rollout_batch,
    _state_policy_env_worker,
    _upsample_policy_trajectory,
    build_state_conditioned_windows,
    denormalize_policy_data,
    eval_model,
    get_trajectory_sample_step,
    make_env,
    make_policy_condition,
    normalize_policy_data,
)
from Robot_simulation.environments.heuristics_util import (  # noqa: E402,F401
    _clip_policy_gripper_dims,
    _current_robot_q,
    _get_environment_params,
    _state_policy_success,
    _to_action_from_q,
    configure_nut_pegs,
    get_dynamic_state,
    render_trajectory,
    write_grid_video,
)

__all__ = [
    "_clip_policy_gripper_dims",
    "_current_robot_q",
    "_generate_val_env",
    "_get_environment_params",
    "_maybe_denormalize_policy_data",
    "_rollout_batch",
    "_state_policy_env_worker",
    "_state_policy_success",
    "_upsample_policy_trajectory",
    "_to_action_from_q",
    "build_state_conditioned_windows",
    "configure_nut_pegs",
    "denormalize_policy_data",
    "eval_model",
    "get_dynamic_state",
    "get_trajectory_sample_step",
    "make_env",
    "make_policy_condition",
    "normalize_policy_data",
    "render_trajectory",
    "write_grid_video",
]
