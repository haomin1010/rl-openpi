from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np


class ChunkRewardValueProvider(Protocol):
    def evaluate_chunk(
        self,
        *,
        obs_t: dict[str, Any],
        action_chunk: np.ndarray,
        obs_t_plus_1: dict[str, Any],
        task: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Return at least a chunk-level `reward` for a transition."""


class CallableChunkRewardValueProvider:
    def __init__(self, fn: Callable[..., dict[str, float]]):
        self._fn = fn

    def evaluate_chunk(
        self,
        *,
        obs_t: dict[str, Any],
        action_chunk: np.ndarray,
        obs_t_plus_1: dict[str, Any],
        task: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        result = self._fn(
            obs_t=obs_t,
            action_chunk=action_chunk,
            obs_t_plus_1=obs_t_plus_1,
            task=task,
            metadata=metadata,
        )
        if "reward" not in result:
            raise KeyError("Chunk reward/value provider must return a `reward` field.")
        return result


class FixedRewardProvider:
    def __init__(self, reward: float = 0.0):
        self._reward = float(reward)

    def evaluate_chunk(
        self,
        *,
        obs_t: dict[str, Any],
        action_chunk: np.ndarray,
        obs_t_plus_1: dict[str, Any],
        task: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del obs_t, action_chunk, obs_t_plus_1, task, metadata
        return {"reward": self._reward}


class EnvChunkRewardProvider:
    """Chunk-level reward from env transition metadata.

    Rule:
    - Non-terminal after consuming i actions in this chunk: reward = -i
    - Terminal success at i-th consumed action: reward = -i
    - Terminal failure (step limit reached) at i-th consumed action: reward = -100 - i
    """

    def evaluate_chunk(
        self,
        *,
        obs_t: dict[str, Any],
        action_chunk: np.ndarray,
        obs_t_plus_1: dict[str, Any],
        task: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        del obs_t, obs_t_plus_1, task
        env_info = (metadata or {}).get("env_info", {})
        consumed = int(env_info.get("consumed_actions", int(np.asarray(action_chunk).shape[0])))
        consumed = max(consumed, 0)
        done = bool(env_info.get("eval_success", False) or env_info.get("take_action_cnt", -1) >= env_info.get("step_lim", 10**9))
        success = bool(env_info.get("eval_success", False))

        if not done:
            return {"reward": float(-consumed)}
        if success:
            return {"reward": float(-consumed)}
        return {"reward": float(-100 - consumed)}
