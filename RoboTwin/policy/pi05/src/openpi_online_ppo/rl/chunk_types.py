from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ChunkSample:
    obs_t: dict[str, Any]
    obs_t_plus_1: dict[str, Any]
    model_inputs_t: dict[str, Any]
    action_chunk: np.ndarray
    action_tokens: np.ndarray
    action_token_mask: np.ndarray
    old_token_logprobs: np.ndarray
    reward: float
    value: float
    next_value: float
    sampled_action_tokens: np.ndarray | None = None
    sampled_action_token_mask: np.ndarray | None = None
    sampled_dct_coeffs: np.ndarray | None = None
    executed_dct_coeffs: np.ndarray | None = None
    keyframe_prob: float | None = None
    phase_class: int | None = None
    next_phase_class: int | None = None
    exploration_applied: bool = False
    bootstrap_mask: float = 1.0
    advantage: float | None = None
    value_target: float | None = None
    task: str | None = None
    done: bool = False
    done_reason: str | None = None
    policy_version: int = 0
    step_id: int = 0
