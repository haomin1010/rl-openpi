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

