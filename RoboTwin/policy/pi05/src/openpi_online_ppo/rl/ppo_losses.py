from __future__ import annotations

import numpy as np

from .chunk_types import ChunkSample


def compute_chunk_td_targets(samples: list[ChunkSample], gamma: float) -> None:
    for sample in samples:
        reward = float(sample.reward)
        value = float(sample.value)
        next_value = float(sample.next_value)
        bootstrap_mask = float(sample.bootstrap_mask)
        sample.advantage = reward + gamma * bootstrap_mask * next_value - value
        sample.value_target = reward + gamma * bootstrap_mask * next_value


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

