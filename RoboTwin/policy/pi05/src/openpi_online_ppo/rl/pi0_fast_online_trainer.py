from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
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


@dataclass(frozen=True)
class Pi0FastOnlineRLConfig:
    rollout_batch_size: int = 512
    mini_batch_size: int = 64
    ppo_epochs: int = 4
    value_epochs: int = 1
    gamma: float = 0.99
    clip_eps: float = 0.2
    entropy_coef: float = 0.0
    buffer_capacity: int = 4096
    max_policy_lag: int = 4
    total_updates: int = 1000
    normalize_advantages: bool = True
    seed: int = 0


@struct.dataclass
class _ActorBatch:
    obs_t: dict[str, Any]
    action_tokens: jnp.ndarray
    action_token_mask: jnp.ndarray
    old_token_logprobs: jnp.ndarray
    chunk_advantages: jnp.ndarray


@struct.dataclass
class _ValueBatch:
    obs_t: dict[str, Any]
    value_targets: jnp.ndarray


@struct.dataclass
class _OnlineTrainState:
    step: jnp.ndarray
    params: nnx.State
    model_def: nnx.GraphDef[Any]
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    actor_opt_state: optax.OptState
    value_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    value_opt_state: optax.OptState


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
        if not getattr(self._policy.model.config, "use_value_head", False):
            raise ValueError("Pi0FastOnlineTrainer requires `Pi0FASTConfig.use_value_head=True`.")
        self._buffer = ChunkRolloutBuffer(capacity_chunks=cfg.buffer_capacity)
        self._policy_version = 0
        self._global_chunks = 0
        self._global_updates = 0
        self._rng = jax.random.key(cfg.seed)
        self._actor_filter = nnx.All(
            self._policy.train_config.trainable_filter,
            nnx.Not(nnx_utils.PathRegex(".*value_head.*")),
        )
        self._value_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*value_head.*"))
        self._train_state = self._init_train_state()
        self._actor_train_step = jax.jit(
            functools.partial(
                self._actor_step_impl,
                self._cfg,
                self._actor_filter,
            ),
            donate_argnums=(1,),
        )
        self._value_train_step = jax.jit(
            functools.partial(
                self._value_step_impl,
                self._value_filter,
            ),
            donate_argnums=(1,),
        )

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def global_chunks(self) -> int:
        return self._global_chunks

    @property
    def global_updates(self) -> int:
        return self._global_updates

    def initialize(self) -> None:
        self._collector.reset()

    def collect_until_ready(self) -> list[ChunkSample]:
        collected: list[ChunkSample] = []
        while len(collected) < self._cfg.rollout_batch_size:
            samples = self._collector.collect_chunk_batch(policy_version=self._policy_version)
            collected.extend(samples)
            self._buffer.extend(samples)
            self._global_chunks += len(samples)
        return collected

    def update_once(self) -> dict[str, float]:
        removed = self._buffer.drop_stale(self._policy_version, self._cfg.max_policy_lag)
        if removed:
            logger.info("Dropped %d stale chunk samples before update.", removed)

        samples = self._buffer.as_list()
        if not samples:
            return {
                "buffer_size": 0.0,
                "actor_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "value_loss": 0.0,
                "value_mae": 0.0,
            }

        compute_chunk_td_targets(samples, gamma=self._cfg.gamma)
        if self._cfg.normalize_advantages:
            normalize_advantages(samples)

        actor_metrics_all: list[dict[str, float]] = []
        value_metrics_all: list[dict[str, float]] = []

        for _ in range(self._cfg.ppo_epochs):
            for mini_batch in self._buffer.make_train_batches(self._cfg.mini_batch_size):
                batch = self._build_actor_batch(mini_batch)
                self._rng, step_rng = jax.random.split(self._rng)
                self._train_state, metrics = self._actor_train_step(step_rng, self._train_state, batch)
                actor_metrics_all.append(_to_float_dict(metrics))

        for _ in range(self._cfg.value_epochs):
            for mini_batch in self._buffer.make_train_batches(self._cfg.mini_batch_size):
                batch = self._build_value_batch(mini_batch)
                self._rng, step_rng = jax.random.split(self._rng)
                self._train_state, metrics = self._value_train_step(step_rng, self._train_state, batch)
                value_metrics_all.append(_to_float_dict(metrics))

        self._policy_version += 1
        self._global_updates += 1
        self._policy.sync_params(self._train_state.model_def, self._train_state.params)

        actor_mean = _mean_metric_dict(actor_metrics_all)
        value_mean = _mean_metric_dict(value_metrics_all)
        return {
            "buffer_size": float(len(samples)),
            "actor_loss": float(actor_mean.get("actor_loss", 0.0)),
            "entropy": float(actor_mean.get("entropy", 0.0)),
            "approx_kl": float(actor_mean.get("approx_kl", 0.0)),
            "clip_fraction": float(actor_mean.get("clip_fraction", 0.0)),
            "value_loss": float(value_mean.get("value_loss", 0.0)),
            "value_mae": float(value_mean.get("value_mae", 0.0)),
        }

    def train(self) -> None:
        self.initialize()
        while self._global_updates < self._cfg.total_updates:
            rollout_samples = self.collect_until_ready()
            rollout_reward = float(np.mean([sample.reward for sample in rollout_samples])) if rollout_samples else 0.0
            metrics = self.update_once()
            metrics["rollout_reward"] = rollout_reward
            metrics["global_chunks"] = float(self._global_chunks)
            metrics["global_updates"] = float(self._global_updates)
            logger.info("online_rl_metrics=%s", metrics)

    def _init_train_state(self) -> _OnlineTrainState:
        train_config = self._policy.train_config
        actor_tx = train_config.optimizer.create(train_config.lr_schedule.create(), weight_decay_mask=None)
        value_tx = train_config.optimizer.create(train_config.lr_schedule.create(), weight_decay_mask=None)
        params = nnx.state(self._policy.model)
        return _OnlineTrainState(
            step=jnp.asarray(0, dtype=jnp.int32),
            params=params,
            model_def=nnx.graphdef(self._policy.model),
            actor_tx=actor_tx,
            actor_opt_state=actor_tx.init(params.filter(self._actor_filter)),
            value_tx=value_tx,
            value_opt_state=value_tx.init(params.filter(self._value_filter)),
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
        )

    def _build_value_batch(self, samples: list[ChunkSample]) -> _ValueBatch:
        obs_t = jax.tree.map(
            lambda *xs: jnp.asarray(np.stack(xs, axis=0)),
            *[sample.model_inputs_t for sample in samples],
        )
        value_targets = jnp.asarray([float(sample.value_target or 0.0) for sample in samples], dtype=jnp.float32)
        return _ValueBatch(obs_t=obs_t, value_targets=value_targets)

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
            token_advantages = jnp.where(batch.action_token_mask, batch.chunk_advantages[:, None], 0.0)
            log_ratio = new_logprobs - batch.old_token_logprobs
            ratio = jnp.exp(log_ratio)
            clipped_ratio = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pg_loss_unclipped = -ratio * token_advantages
            pg_loss_clipped = -clipped_ratio * token_advantages
            actor_loss_per_token = jnp.maximum(pg_loss_unclipped, pg_loss_clipped)
            denom = jnp.maximum(jnp.sum(token_mask), 1.0)
            actor_loss = jnp.sum(actor_loss_per_token * token_mask) / denom
            entropy = jnp.sum(stats["token_entropy"] * token_mask) / denom
            total_loss = actor_loss - cfg.entropy_coef * entropy
            approx_kl = jnp.sum((batch.old_token_logprobs - new_logprobs) * token_mask) / denom
            clip_fraction = jnp.sum((jnp.abs(ratio - 1.0) > cfg.clip_eps) * token_mask) / denom
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
            value_tx=state.value_tx,
            value_opt_state=state.value_opt_state,
        )
        metrics = {
            **metrics,
            "loss": loss,
            "actor_grad_norm": optax.global_norm(grads),
        }
        return next_state, metrics

    @staticmethod
    def _value_step_impl(value_filter, rng, state: _OnlineTrainState, batch: _ValueBatch):
        del rng
        model = nnx.merge(state.model_def, state.params)

        def loss_fn(model):
            observation = _model.Observation.from_dict(batch.obs_t)
            logits = model.predict_value_logits(observation, stop_gradient=True)
            soft_target = model.project_values_to_bins(batch.value_targets)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            per_sample_loss = -jnp.sum(soft_target * log_probs, axis=-1)
            value_loss = jnp.mean(per_sample_loss)
            pred_value = model.predict_value(observation, stop_gradient=True)
            value_mae = jnp.mean(jnp.abs(pred_value - batch.value_targets))
            return value_loss, {
                "value_loss": value_loss,
                "value_mae": value_mae,
            }

        diff_state = nnx.DiffState(0, value_filter)
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
        params = state.params.filter(value_filter)
        updates, new_opt_state = state.value_tx.update(grads, state.value_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        next_state = _OnlineTrainState(
            step=state.step,
            params=nnx.state(model),
            model_def=state.model_def,
            actor_tx=state.actor_tx,
            actor_opt_state=state.actor_opt_state,
            value_tx=state.value_tx,
            value_opt_state=new_opt_state,
        )
        metrics = {
            **metrics,
            "loss": loss,
            "value_grad_norm": optax.global_norm(grads),
        }
        return next_state, metrics


def _to_float_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


def _mean_metric_dict(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = set().union(*metrics_list)
    return {key: float(np.mean([m[key] for m in metrics_list if key in m])) for key in keys}
