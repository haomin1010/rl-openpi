from __future__ import annotations

from dataclasses import dataclass
import functools
import json
import logging
import time
from typing import Any

import flax.nnx as nnx
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi_online_ppo.rl.chunk_buffer import ChunkRolloutBuffer
from openpi_online_ppo.rl.chunk_types import ChunkSample
from openpi_online_ppo.rl.pi0_fast_policy import Pi0FastRLPolicy
from openpi_online_ppo.rl.pi0_fast_rollout import Pi0FastChunkCollector
from openpi_online_ppo.rl.ppo_losses import compute_chunk_td_targets, normalize_advantages

logger = logging.getLogger(__name__)
timing_logger = logging.getLogger("openpi_online_ppo.timing")


def _state_fingerprint(state: nnx.State) -> float:
    leaves = jax.tree.leaves(state)
    acc = 0.0
    for leaf in leaves:
        arr = np.asarray(leaf)
        if arr.size == 0:
            continue
        acc += float(arr.reshape(-1)[0].astype(np.float64))
    return acc


@dataclass(frozen=True)
class Pi0FastOnlineRLConfig:
    rollout_batch_size: int = 512
    mini_batch_size: int = 64
    ppo_epochs: int = 4
    gamma: float = 0.99
    clip_eps: float = 0.2
    entropy_coef: float = 0.0
    buffer_capacity: int = 4096
    total_updates: int = 1000
    normalize_advantages: bool = True
    seed: int = 0
    explored_chunk_weight: float = 1.0
    non_explored_chunk_weight: float = 1.0


@struct.dataclass
class _ActorBatch:
    obs_t: dict[str, Any]
    action_tokens: jnp.ndarray
    action_token_mask: jnp.ndarray
    old_token_logprobs: jnp.ndarray
    chunk_advantages: jnp.ndarray
    chunk_weights: jnp.ndarray


@struct.dataclass
class _OnlineTrainState:
    step: jnp.ndarray
    params: nnx.State
    model_def: nnx.GraphDef[Any]
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    actor_opt_state: optax.OptState


class Pi0FastOnlineTrainer:
    """Online chunk-level PPO trainer with split actor/value optimization."""

    def __init__(
        self,
        *,
        cfg: Pi0FastOnlineRLConfig,
        policy: Pi0FastRLPolicy,
        collector: Pi0FastChunkCollector,
    ) -> None:
        self._cfg = cfg
        self._policy = policy
        self._collector = collector
        self._buffer = ChunkRolloutBuffer(capacity_chunks=cfg.buffer_capacity)
        self._global_chunks = 0
        self._global_updates = 0
        self._rng = jax.random.key(cfg.seed)
        self._actor_filter = nnx.All(
            self._policy.train_config.trainable_filter,
            nnx.Not(nnx_utils.PathRegex(".*value_head.*")),
        )
        self._train_state = self._init_train_state()
        self._actor_train_step = jax.jit(
            functools.partial(
                self._actor_step_impl,
                self._cfg,
                self._actor_filter,
            ),
            donate_argnums=(1,),
        )

    @property
    def global_chunks(self) -> int:
        return self._global_chunks

    @property
    def global_updates(self) -> int:
        return self._global_updates

    def initialize(self) -> None:
        self._collector.reset()

    def collect_until_ready(self, target_chunks: int) -> list[ChunkSample]:
        if target_chunks <= 0:
            return []
        total_t0 = time.perf_counter()
        collected: list[ChunkSample] = []
        while len(collected) < target_chunks:
            samples = self._collector.collect_chunk_batch()
            collected.extend(samples)
            self._buffer.extend(samples)
            self._global_chunks += len(samples)
        if not timing_logger.disabled:
            timing_logger.info(
                json.dumps(
                    {
                        "event": "trainer_collect_until_ready",
                        "target_chunks": int(target_chunks),
                        "collected_chunks": int(len(collected)),
                        "buffer_size_after": int(len(self._buffer)),
                        "duration_s": time.perf_counter() - total_t0,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        return collected

    def update_once(self) -> dict[str, float]:
        total_t0 = time.perf_counter()
        samples = self._buffer.as_list()
        if not samples:
            return {
                "buffer_size": 0.0,
                "actor_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            }

        compute_chunk_td_targets(samples, gamma=self._cfg.gamma)
        advantage_s = time.perf_counter() - total_t0
        norm_t0 = time.perf_counter()
        if self._cfg.normalize_advantages:
            normalize_advantages(samples)
        normalize_s = time.perf_counter() - norm_t0

        actor_metrics_all: list[dict[str, float]] = []
        actor_t0 = time.perf_counter()
        train_step_times: list[float] = []
        build_batch_times: list[float] = []
        for _ in range(self._cfg.ppo_epochs):
            for mini_batch in self._buffer.make_train_batches(self._cfg.mini_batch_size):
                build_t0 = time.perf_counter()
                batch = self._build_actor_batch(mini_batch)
                build_batch_times.append(time.perf_counter() - build_t0)
                self._rng, step_rng = jax.random.split(self._rng)
                step_t0 = time.perf_counter()
                self._train_state, metrics = self._actor_train_step(step_rng, self._train_state, batch)
                train_step_times.append(time.perf_counter() - step_t0)
                actor_metrics_all.append(_to_float_dict(metrics))
        actor_train_s = time.perf_counter() - actor_t0

        self._global_updates += 1
        sync_t0 = time.perf_counter()
        train_state_fingerprint = _state_fingerprint(self._train_state.params)
        self._policy.sync_params(self._train_state.model_def, self._train_state.params)
        sync_s = time.perf_counter() - sync_t0

        actor_mean = _mean_metric_dict(actor_metrics_all)
        out = {
            "buffer_size": float(len(samples)),
            "actor_loss": float(actor_mean.get("actor_loss", 0.0)),
            "entropy": float(actor_mean.get("entropy", 0.0)),
            "approx_kl": float(actor_mean.get("approx_kl", 0.0)),
            "clip_fraction": float(actor_mean.get("clip_fraction", 0.0)),
        }
        if not timing_logger.disabled:
            timing_logger.info(
                json.dumps(
                    {
                        "event": "trainer_update_once",
                        "buffer_size": int(len(samples)),
                        "mini_batch_size": int(self._cfg.mini_batch_size),
                        "ppo_epochs": int(self._cfg.ppo_epochs),
                        "compute_advantage_s": advantage_s,
                        "normalize_advantage_s": normalize_s,
                        "actor_train_s": actor_train_s,
                        "actor_train_num_steps": int(len(train_step_times)),
                        "actor_train_step_sum_s": float(np.sum(train_step_times)) if train_step_times else 0.0,
                        "actor_train_step_mean_s": float(np.mean(train_step_times)) if train_step_times else 0.0,
                        "actor_train_step_max_s": float(np.max(train_step_times)) if train_step_times else 0.0,
                        "build_actor_batch_sum_s": float(np.sum(build_batch_times)) if build_batch_times else 0.0,
                        "build_actor_batch_mean_s": float(np.mean(build_batch_times)) if build_batch_times else 0.0,
                        "build_actor_batch_max_s": float(np.max(build_batch_times)) if build_batch_times else 0.0,
                        "sync_params_s": sync_s,
                        "train_state_fingerprint": float(train_state_fingerprint),
                        "total_s": time.perf_counter() - total_t0,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        return out

    def train(self) -> None:
        self.initialize()
        warmup_chunks = max(1, int(self._cfg.buffer_capacity))
        refresh_chunks = max(1, int(self._cfg.buffer_capacity) // 4)
        logger.info(
            "online_rl_warmup buffer_capacity=%d refresh_chunks=%d ppo_epochs=%d",
            warmup_chunks,
            refresh_chunks,
            int(self._cfg.ppo_epochs),
        )
        self.collect_until_ready(warmup_chunks)
        while self._global_updates < self._cfg.total_updates:
            metrics = self.update_once()
            metrics["global_chunks"] = float(self._global_chunks)
            metrics["global_updates"] = float(self._global_updates)
            metrics["buffer_refresh_chunks"] = float(refresh_chunks)
            logger.info("online_rl_metrics=%s", metrics)
            if self._global_updates >= self._cfg.total_updates:
                break
            self.collect_until_ready(refresh_chunks)

    def _init_train_state(self) -> _OnlineTrainState:
        train_config = self._policy.train_config
        actor_tx = train_config.optimizer.create(train_config.lr_schedule.create(), weight_decay_mask=None)
        params = nnx.state(self._policy.model)
        return _OnlineTrainState(
            step=jnp.asarray(0, dtype=jnp.int32),
            params=params,
            model_def=nnx.graphdef(self._policy.model),
            actor_tx=actor_tx,
            actor_opt_state=actor_tx.init(params.filter(self._actor_filter)),
        )

    def _build_actor_batch(self, samples: list[ChunkSample]) -> _ActorBatch:
        obs_t = jax.tree.map(
            lambda *xs: jnp.asarray(np.stack(xs, axis=0)),
            *[sample.model_inputs_t for sample in samples],
        )
        return _ActorBatch(
            obs_t=obs_t,
            action_tokens=jnp.asarray(np.stack([sample.action_tokens for sample in samples], axis=0), dtype=jnp.int32),
            action_token_mask=jnp.asarray(
                np.stack([sample.action_token_mask for sample in samples], axis=0), dtype=jnp.bool_
            ),
            old_token_logprobs=jnp.asarray(
                np.stack([sample.old_token_logprobs for sample in samples], axis=0), dtype=jnp.float32
            ),
            chunk_advantages=jnp.asarray([float(sample.advantage or 0.0) for sample in samples], dtype=jnp.float32),
            chunk_weights=jnp.asarray(
                [
                    float(self._cfg.explored_chunk_weight)
                    if bool(sample.exploration_applied)
                    else float(self._cfg.non_explored_chunk_weight)
                    for sample in samples
                ],
                dtype=jnp.float32,
            ),
        )

    @staticmethod
    def _actor_step_impl(cfg: Pi0FastOnlineRLConfig, actor_filter, rng, state: _OnlineTrainState, batch: _ActorBatch):
        del rng
        model = nnx.merge(state.model_def, state.params)

        def loss_fn(model):
            observation = _model.Observation.from_dict(batch.obs_t)
            stats = model.recompute_action_logprobs(
                observation,
                batch.action_tokens,
                action_token_mask=batch.action_token_mask,
            )
            new_logprobs = stats["token_logprobs"]
            token_mask = batch.action_token_mask.astype(jnp.float32)
            chunk_weights = jnp.maximum(batch.chunk_weights, 0.0)
            token_weights = token_mask * chunk_weights[:, None]
            token_advantages = jnp.where(batch.action_token_mask, batch.chunk_advantages[:, None], 0.0)
            log_ratio = new_logprobs - batch.old_token_logprobs
            ratio = jnp.exp(log_ratio)
            clipped_ratio = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pg_loss_unclipped = -ratio * token_advantages
            pg_loss_clipped = -clipped_ratio * token_advantages
            actor_loss_per_token = jnp.maximum(pg_loss_unclipped, pg_loss_clipped)
            denom = jnp.maximum(jnp.sum(token_weights), 1.0)
            actor_loss = jnp.sum(actor_loss_per_token * token_weights) / denom
            entropy = jnp.sum(stats["token_entropy"] * token_weights) / denom
            total_loss = actor_loss - cfg.entropy_coef * entropy
            approx_kl = jnp.sum((batch.old_token_logprobs - new_logprobs) * token_weights) / denom
            clip_fraction = jnp.sum((jnp.abs(ratio - 1.0) > cfg.clip_eps) * token_weights) / denom
            return total_loss, {
                "actor_loss": actor_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            }

        diff_state = nnx.DiffState(0, actor_filter)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
        params = state.params.filter(actor_filter)
        updates, new_opt_state = state.actor_tx.update(grads, state.actor_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        next_state = _OnlineTrainState(
            step=state.step + 1,
            params=nnx.state(model),
            model_def=state.model_def,
            actor_tx=state.actor_tx,
            actor_opt_state=new_opt_state,
        )
        metrics = {
            **metrics,
            "loss": loss,
            "actor_grad_norm": optax.global_norm(grads),
        }
        return next_state, metrics


def _to_float_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


def _mean_metric_dict(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = set().union(*metrics_list)
    return {key: float(np.mean([m[key] for m in metrics_list if key in m])) for key in keys}
