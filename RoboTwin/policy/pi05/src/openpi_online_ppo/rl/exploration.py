from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class KeyframeExplorationConfig:
    # User requirement: always-on in phase-1.
    always_on: bool = True
    explore_dct_dims: int = 0
    relaxed_fast_decoding: bool = True
    noise_std: float = 0.01


class DCTNoiseFn(Protocol):
    def __call__(
        self,
        *,
        dct_coeffs: np.ndarray,
        num_noisy_dims: int,
        obs: dict[str, Any],
        keyframe_prob: float,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Mutate or return a copy of DCT coefficients used for exploration."""


class CallableDCTNoiseFn:
    def __init__(self, fn: Callable[..., np.ndarray]):
        self._fn = fn

    def __call__(
        self,
        *,
        dct_coeffs: np.ndarray,
        num_noisy_dims: int,
        obs: dict[str, Any],
        keyframe_prob: float,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self._fn(
                dct_coeffs=np.asarray(dct_coeffs),
                num_noisy_dims=int(num_noisy_dims),
                obs=obs,
                keyframe_prob=float(keyframe_prob),
                metadata=metadata,
            ),
            dtype=np.float32,
        )


def default_dct_noise(
    *,
    dct_coeffs: np.ndarray,
    num_noisy_dims: int,
    obs: dict[str, Any],
    keyframe_prob: float,
    metadata: dict[str, Any] | None = None,
    noise_std: float = 0.01,
) -> np.ndarray:
    del obs, keyframe_prob, metadata
    coeffs = np.asarray(dct_coeffs, dtype=np.float32).copy()
    if num_noisy_dims <= 0:
        return coeffs
    dims = min(int(num_noisy_dims), coeffs.shape[-1])
    coeffs[..., :dims] += np.random.normal(0.0, noise_std, size=coeffs[..., :dims].shape).astype(np.float32)
    return coeffs


def maybe_apply_dct_exploration(
    *,
    dct_coeffs: np.ndarray,
    cfg: KeyframeExplorationConfig,
    keyframe_prob: float,
    noise_fn: DCTNoiseFn | None,
    obs: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool]:
    coeffs = np.asarray(dct_coeffs, dtype=np.float32)
    if not cfg.always_on or noise_fn is None or cfg.explore_dct_dims <= 0:
        return coeffs, False
    updated = noise_fn(
        dct_coeffs=coeffs,
        num_noisy_dims=min(int(cfg.explore_dct_dims), coeffs.shape[-1]),
        obs=obs,
        keyframe_prob=float(keyframe_prob),
        metadata=metadata,
    )
    return np.asarray(updated, dtype=np.float32), True

