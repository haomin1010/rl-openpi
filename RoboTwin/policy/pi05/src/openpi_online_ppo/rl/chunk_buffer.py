from __future__ import annotations

from collections import deque
import random

from .chunk_types import ChunkSample


class ChunkRolloutBuffer:
    def __init__(self, capacity_chunks: int):
        if capacity_chunks <= 0:
            raise ValueError("capacity_chunks must be positive.")
        self._samples: deque[ChunkSample] = deque(maxlen=capacity_chunks)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: ChunkSample) -> None:
        self._samples.append(sample)

    def extend(self, samples: list[ChunkSample]) -> None:
        self._samples.extend(samples)

    def as_list(self) -> list[ChunkSample]:
        return list(self._samples)

    def drop_stale(self, current_policy_version: int, max_policy_lag: int) -> int:
        if max_policy_lag < 0:
            return 0
        kept = [
            sample
            for sample in self._samples
            if current_policy_version - int(sample.policy_version) <= max_policy_lag
        ]
        removed = len(self._samples) - len(kept)
        self._samples = deque(kept, maxlen=self._samples.maxlen)
        return removed

    def make_train_batches(self, mini_batch_size: int) -> list[list[ChunkSample]]:
        if mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be positive.")
        samples = list(self._samples)
        random.shuffle(samples)
        return [samples[i : i + mini_batch_size] for i in range(0, len(samples), mini_batch_size)]

