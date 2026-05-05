from __future__ import annotations

import numpy as np

from .chunk_types import ChunkSample


def compute_default_advantage(sample: ChunkSample) -> float:
    del sample
    return 0.0


def compute_chunk_td_targets(samples: list[ChunkSample], gamma: float) -> None:
    del gamma
    for sample in samples:
        if bool(sample.uses_value) and np.isfinite(sample.value) and np.isfinite(sample.next_value):
            value = float(sample.value)
            next_value = float(sample.next_value)
            sample.advantage = (next_value - value) / 5.0 + 0.1
            sample.value_target = next_value
        else:
            sample.advantage = compute_default_advantage(sample)
            sample.value_target = float("nan")


def normalize_advantages(samples: list[ChunkSample], eps: float = 1e-8) -> None:
    adv = np.asarray([0.0 if s.advantage is None else s.advantage for s in samples], dtype=np.float32)
    if adv.size == 0:
        return
    std = float(np.std(adv))
    if std < eps:
        centered = adv - float(np.mean(adv))
    else:
        centered = (adv - float(np.mean(adv))) / (std + eps)
    for sample, norm_adv in zip(samples, centered, strict=True):
        sample.advantage = float(norm_adv)
