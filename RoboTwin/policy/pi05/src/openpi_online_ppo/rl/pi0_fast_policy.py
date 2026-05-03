from __future__ import annotations

from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any
from typing import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi_online_ppo.models import pi0_fast_rl as _pi0_fast
from openpi_online_ppo.rl.exploration import DCTNoiseFn, DimwiseDiagGaussianDCTPerturbNet, KeyframeExplorationConfig, maybe_apply_dct_exploration
import openpi.shared.download as _download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

logger = logging.getLogger(__name__)


def _extract_fast_output_tokenizer(
    output_transforms: Sequence[_transforms.DataTransformFn],
) -> tuple[Any, int, int, Sequence[_transforms.DataTransformFn]]:
    for i, transform in enumerate(output_transforms):
        if isinstance(transform, _transforms.ExtractFASTActions):
            return (
                transform.tokenizer,
                int(transform.action_horizon),
                int(transform.action_dim),
                output_transforms[i + 1 :],
            )
    raise ValueError("FAST RL policy requires an ExtractFASTActions transform in the output pipeline.")


class Pi0FastRLPolicy:
    """Additive RL wrapper around a trained `pi0_fast` policy.

    This wrapper intentionally does not modify the standard eval / inference
    policy interfaces. It only exposes trace-oriented methods required by the
    online PPO flow.
    """

    def __init__(
        self,
        model: _pi0_fast.Pi0FAST,
        *,
        train_config: _config.TrainConfig,
        rng: jax.Array,
        transforms: Sequence[_transforms.DataTransformFn],
        output_transforms: Sequence[_transforms.DataTransformFn],
        sample_kwargs: dict[str, Any] | None = None,
        exploration_config: KeyframeExplorationConfig | None = None,
        dct_noise_fn: DCTNoiseFn | None = None,
        keyframe_prob_fn: Callable[[dict[str, Any]], float] | None = None,
        perturb_net: Any | None = None,
    ) -> None:
        self._model = model
        self._train_config = train_config
        self._rng = rng
        self._input_transform = _transforms.compose(transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._exploration_cfg = exploration_config or KeyframeExplorationConfig()
        self._dct_noise_fn = dct_noise_fn
        self._keyframe_prob_fn = keyframe_prob_fn
        self._perturb_net = perturb_net
        self._fast_tokenizer, self._action_horizon, self._action_dim, post_extract_transforms = (
            _extract_fast_output_tokenizer(output_transforms)
        )
        self._post_extract_output_transform = _transforms.compose(post_extract_transforms)

    @staticmethod
    def _pad_token_sequence(tokens: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        if target_len <= 0:
            raise ValueError("target_len must be positive.")
        if tokens.shape[0] >= target_len:
            return tokens[:target_len], np.ones((target_len,), dtype=bool)
        padded = np.zeros((target_len,), dtype=np.int32)
        mask = np.zeros((target_len,), dtype=bool)
        padded[: tokens.shape[0]] = tokens
        mask[: tokens.shape[0]] = True
        return padded, mask

    def _prepare_observation(self, obs: dict[str, Any]) -> tuple[_model.Observation, dict[str, Any]]:
        inputs = jax.tree.map(lambda x: x, obs)
        transformed = self._input_transform(inputs)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
        return _model.Observation.from_dict(batched), transformed

    def _prepare_observations(self, obs_batch: list[dict[str, Any]]) -> tuple[_model.Observation, list[dict[str, Any]]]:
        if not obs_batch:
            raise ValueError("obs_batch must not be empty.")
        transformed_batch = []
        for obs in obs_batch:
            inputs = jax.tree.map(lambda x: x, obs)
            transformed_batch.append(self._input_transform(inputs))
        batched = jax.tree.map(
            lambda *xs: jnp.asarray(np.stack([np.asarray(x) for x in xs], axis=0)),
            *transformed_batch,
        )
        return _model.Observation.from_dict(batched), transformed_batch

    @staticmethod
    def _truncate_by_mask(tokens: np.ndarray, token_mask: np.ndarray | None) -> np.ndarray:
        toks = np.asarray(tokens, dtype=np.int32).reshape(-1)
        if token_mask is None:
            return toks
        mask = np.asarray(token_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != toks.shape[0]:
            raise ValueError(f"token/mask length mismatch: {toks.shape[0]} vs {mask.shape[0]}")
        valid = toks[mask]
        return valid if valid.size > 0 else toks[:0]

    def _should_use_rowwise_sigma_rollout(self) -> bool:
        return (
            isinstance(self._perturb_net, DimwiseDiagGaussianDCTPerturbNet)
            and bool(getattr(self._fast_tokenizer, "rowwise_bpe", False))
            and str(getattr(self._exploration_cfg, "perturb_backend", "")).lower() == "dct_sigma_network"
            and str(getattr(self._exploration_cfg, "mode", "always")).lower() != "never"
        )

    def _decode_step(
        self,
        *,
        last_logit: jnp.ndarray,
        cache: Any,
        token: int,
        step_idx: int,
        prefill_len: jnp.ndarray,
        prefill_size: int,
        prefix_start: jnp.ndarray,
        max_decoding_steps: int,
    ) -> tuple[jnp.ndarray, Any]:
        token_arr = jnp.asarray([[int(token)]], dtype=jnp.int32)
        token_embedding = self._model.PaliGemma.llm(token_arr, embed_only=True)
        positions = prefill_len[:, None] + int(step_idx) + 1
        mask = jnp.logical_and(
            jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
            jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
            < jnp.broadcast_to(prefill_size + int(step_idx) + 1, (prefix_start.shape[0], 1, 1)),
        )
        next_logit, next_cache, _ = self._model.PaliGemma.llm(
            embedded_prefix=token_embedding,
            mask=mask,
            positions=positions,
            decode=True,
            kv_cache=cache,
        )
        return next_logit, next_cache

    def _pick_valid_row_token(
        self,
        *,
        logit: np.ndarray,
        current_row_pg_tokens: list[int],
        row_char_width: int,
        temperature: float,
    ) -> int:
        logits = np.asarray(logit, dtype=np.float32).reshape(-1)
        token_ids: np.ndarray
        if temperature > 0.0:
            scaled = logits / max(float(temperature), 1e-6)
            scaled = scaled - float(np.max(scaled))
            probs = np.exp(scaled)
            probs = probs / np.maximum(float(np.sum(probs)), 1e-8)
            token_ids = np.random.default_rng().choice(np.arange(logits.shape[0]), size=min(256, logits.shape[0]), replace=False, p=probs)
        else:
            token_ids = np.argsort(logits)[::-1]
        for token_id in token_ids.tolist():
            token_i = int(token_id)
            if token_i == 1:
                continue
            cand_pg = np.asarray(current_row_pg_tokens + [token_i], dtype=np.int32)
            try:
                text = self._fast_tokenizer.decode_pg_tokens_to_text(cand_pg)
            except Exception:
                continue
            if len(text) <= int(row_char_width):
                return token_i
        return int(np.argmax(logits))

    def _sample_chunk_rowwise_sigma(
        self,
        obs: dict[str, Any],
        observation: _model.Observation,
        transformed: dict[str, Any],
        *,
        include_logprobs: bool,
        keyframe_prob: float,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        max_decoding_steps = int(self._sample_kwargs.get("max_decoding_steps", 256))
        temperature = float(self._sample_kwargs.get("temperature", 0.0))
        t0 = time.perf_counter()
        decode_state = self._model._prepare_decode_prefix(observation, max_decoding_steps)
        prefix_prepare_s = time.perf_counter() - t0
        last_logit = decode_state["last_logit"]
        kv_cache = decode_state["kv_cache"]
        prefill_size = int(decode_state["prefill_size"])
        prefill_len = decode_state["prefill_len"]
        prefix_start = decode_state["prefix_start"]

        all_raw_action_pg_tokens: list[int] = []
        all_exec_action_pg_tokens: list[int] = []
        raw_rows: list[np.ndarray] = []
        exec_rows: list[np.ndarray] = []
        step_idx = 0
        row_decode_s = 0.0
        sigma_sample_s = 0.0
        row_replay_s = 0.0
        row_token_counts: list[int] = []
        exec_row_token_counts: list[int] = []

        for token in self._fast_tokenizer.action_prefix_tokens().tolist():
            last_logit, kv_cache = self._decode_step(
                last_logit=last_logit,
                cache=kv_cache,
                token=int(token),
                step_idx=step_idx,
                prefill_len=prefill_len,
                prefill_size=prefill_size,
                prefix_start=prefix_start,
                max_decoding_steps=max_decoding_steps,
            )
            step_idx += 1

        row_count = self._fast_tokenizer.num_rows(action_horizon=self._action_horizon, action_dim=self._action_dim)
        row_char_width = self._fast_tokenizer.row_char_width(action_horizon=self._action_horizon, action_dim=self._action_dim)

        for _row_idx in range(row_count):
            row_start_logit = last_logit
            row_start_cache = kv_cache
            row_start_step = step_idx
            current_row_pg_tokens: list[int] = []
            row_decode_len = 0
            max_row_tokens = max(8, row_char_width * 4)
            row_t0 = time.perf_counter()
            while True:
                token = self._pick_valid_row_token(
                    logit=np.asarray(last_logit[0, 0, :], dtype=np.float32),
                    current_row_pg_tokens=current_row_pg_tokens,
                    row_char_width=row_char_width,
                    temperature=temperature,
                )
                current_row_pg_tokens.append(int(token))
                step_idx += 1
                last_logit, kv_cache = self._decode_step(
                    last_logit=last_logit,
                    cache=kv_cache,
                    token=int(token),
                    step_idx=step_idx - 1,
                    prefill_len=prefill_len,
                    prefill_size=prefill_size,
                    prefix_start=prefix_start,
                    max_decoding_steps=max_decoding_steps,
                )
                row_decode_len = len(self._fast_tokenizer.decode_pg_tokens_to_text(np.asarray(current_row_pg_tokens, dtype=np.int32)))
                if row_decode_len >= row_char_width or len(current_row_pg_tokens) >= max_row_tokens:
                    break
            row_decode_s += time.perf_counter() - row_t0
            row_token_counts.append(len(current_row_pg_tokens))

            raw_row = self._fast_tokenizer.decode_row_pg_tokens_to_coeffs(
                np.asarray(current_row_pg_tokens, dtype=np.int32),
                row_width=row_char_width,
            )
            raw_rows.append(raw_row.astype(np.float32, copy=False))
            all_raw_action_pg_tokens.extend(current_row_pg_tokens)

            row_matrix = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)
            if self._fast_tokenizer.rowwise_layout == "action_dim_major":
                row_matrix[:, _row_idx] = raw_row
            else:
                row_matrix[_row_idx, :] = raw_row
            sigma_t0 = time.perf_counter()
            delta_matrix = self._perturb_net.sample_delta_dct(
                dct_coeffs=row_matrix,
                action_noise_dims=getattr(self._exploration_cfg, "action_noise_indices", None),
            )
            sigma_sample_s += time.perf_counter() - sigma_t0
            exec_row = np.asarray(raw_row + (delta_matrix[:, _row_idx] if self._fast_tokenizer.rowwise_layout == "action_dim_major" else delta_matrix[_row_idx, :]), dtype=np.float32)
            exec_rows.append(exec_row)
            exec_row_pg_tokens = self._fast_tokenizer.encode_row_chars_to_pg_tokens(np.rint(exec_row).astype(np.int32))
            all_exec_action_pg_tokens.extend(exec_row_pg_tokens.tolist())
            exec_row_token_counts.append(len(exec_row_pg_tokens))

            last_logit = row_start_logit
            kv_cache = row_start_cache
            step_idx = row_start_step
            replay_t0 = time.perf_counter()
            for token in exec_row_pg_tokens.tolist():
                last_logit, kv_cache = self._decode_step(
                    last_logit=last_logit,
                    cache=kv_cache,
                    token=int(token),
                    step_idx=step_idx,
                    prefill_len=prefill_len,
                    prefill_size=prefill_size,
                    prefix_start=prefix_start,
                    max_decoding_steps=max_decoding_steps,
                )
                step_idx += 1
            row_replay_s += time.perf_counter() - replay_t0

        sampled_action_tokens = self._fast_tokenizer.format_action_tokens_as_output(
            np.asarray(all_raw_action_pg_tokens, dtype=np.int32),
            add_eos=True,
        )
        executed_action_tokens = self._fast_tokenizer.format_action_tokens_as_output(
            np.asarray(all_exec_action_pg_tokens, dtype=np.int32),
            add_eos=True,
        )
        sampled_action_token_mask = np.ones_like(sampled_action_tokens, dtype=bool)
        executed_action_token_mask = np.ones_like(executed_action_tokens, dtype=bool)

        sampled_dct_coeffs = (
            np.stack(raw_rows, axis=1) if self._fast_tokenizer.rowwise_layout == "action_dim_major" else np.stack(raw_rows, axis=0)
        ).astype(np.float32)
        explored_dct_coeffs = (
            np.stack(exec_rows, axis=1) if self._fast_tokenizer.rowwise_layout == "action_dim_major" else np.stack(exec_rows, axis=0)
        ).astype(np.float32)
        if include_logprobs:
            logprob_t0 = time.perf_counter()
            old_token_stats = self._model.recompute_action_logprobs(
                observation,
                jnp.asarray(executed_action_tokens, dtype=jnp.int32)[np.newaxis, ...],
                action_token_mask=jnp.asarray(executed_action_token_mask, dtype=jnp.bool_)[np.newaxis, ...],
            )
            old_token_logprobs = np.asarray(old_token_stats["token_logprobs"][0], dtype=np.float32)
            logprob_s = time.perf_counter() - logprob_t0
        else:
            old_token_logprobs = np.zeros(executed_action_tokens.shape, dtype=np.float32)
            logprob_s = 0.0
        decoded_actions = self._fast_tokenizer.decode_action_dct_coeffs(explored_dct_coeffs)
        outputs = self._post_extract_output_transform(
            {
                "state": np.asarray(transformed["state"]),
                "actions": np.asarray(decoded_actions, dtype=np.float32),
            }
        )
        total_s = time.perf_counter() - total_start
        logger.info(
            "rowwise_sigma_timing total=%.3fs prepare_prefix=%.3fs row_decode=%.3fs sigma_sample=%.3fs row_replay=%.3fs recompute_logprobs=%.3fs rows=%d raw_row_tokens_mean=%.1f exec_row_tokens_mean=%.1f keyframe_prob=%.4f",
            total_s,
            prefix_prepare_s,
            row_decode_s,
            sigma_sample_s,
            row_replay_s,
            logprob_s,
            row_count,
            float(np.mean(row_token_counts)) if row_token_counts else 0.0,
            float(np.mean(exec_row_token_counts)) if exec_row_token_counts else 0.0,
            float(keyframe_prob),
        )
        return {
            "action_chunk": np.asarray(outputs["actions"], dtype=np.float32),
            "action_tokens": executed_action_tokens,
            "action_token_mask": executed_action_token_mask,
            "old_token_logprobs": old_token_logprobs,
            "decoded_actions": np.asarray(decoded_actions, dtype=np.float32),
            "sampled_action_tokens": sampled_action_tokens,
            "sampled_action_token_mask": sampled_action_token_mask,
            "sampled_dct_coeffs": np.asarray(sampled_dct_coeffs, dtype=np.float32),
            "dct_coeffs": np.asarray(explored_dct_coeffs, dtype=np.float32),
            "keyframe_prob": float(keyframe_prob),
            "exploration_applied": True,
            "transformed_obs": transformed,
        }

    def sample_chunk(self, obs: dict[str, Any], *, include_logprobs: bool = True) -> dict[str, Any]:
        total_start = time.perf_counter()
        observation, transformed = self._prepare_observation(obs)
        keyframe_s = 0.0
        if str(self._exploration_cfg.mode) == "never":
            keyframe_prob = 0.0
        elif self._keyframe_prob_fn is not None:
            keyframe_t0 = time.perf_counter()
            keyframe_prob = float(self._keyframe_prob_fn(obs))
            keyframe_s = time.perf_counter() - keyframe_t0
        else:
            keyframe_t0 = time.perf_counter()
            keyframe_prob = self.predict_keyframe_prob(obs) if getattr(self._model, "keyframe_head", None) is not None else 0.0
            keyframe_s = time.perf_counter() - keyframe_t0
        use_rowwise_sigma = self._should_use_rowwise_sigma_rollout()
        if use_rowwise_sigma and str(self._exploration_cfg.mode).lower() == "conditional_keyframe":
            use_rowwise_sigma = float(keyframe_prob) >= float(self._exploration_cfg.keyframe_prob_threshold)
        if use_rowwise_sigma:
            out = self._sample_chunk_rowwise_sigma(
                obs,
                observation,
                transformed,
                include_logprobs=include_logprobs,
                keyframe_prob=keyframe_prob,
            )
            logger.info(
                "sample_chunk_timing mode=rowwise_sigma total=%.3fs keyframe=%.3fs keyframe_prob=%.4f explored=%s",
                time.perf_counter() - total_start,
                keyframe_s,
                float(keyframe_prob),
                bool(out.get("exploration_applied", False)),
            )
            return out
        sample_t0 = time.perf_counter()
        self._rng, sample_rng = jax.random.split(self._rng)
        trace = self._model.sample_actions_with_trace(sample_rng, observation, **self._sample_kwargs)
        sample_trace_s = time.perf_counter() - sample_t0

        sampled_action_tokens = np.asarray(trace["tokens"][0], dtype=np.int32)
        sampled_action_token_mask = np.asarray(trace["token_mask"][0], dtype=bool)
        sampled_valid_tokens = self._truncate_by_mask(sampled_action_tokens, sampled_action_token_mask)
        dct_coeffs = self._fast_tokenizer.extract_action_dct_coeffs(
            sampled_valid_tokens,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            relaxed_decoding=self._exploration_cfg.relaxed_fast_decoding,
        )
        explored_dct_coeffs, exploration_applied = maybe_apply_dct_exploration(
            dct_coeffs=dct_coeffs,
            cfg=self._exploration_cfg,
            keyframe_prob=keyframe_prob,
            noise_fn=self._dct_noise_fn,
            obs=obs,
            metadata={
                "sampled_action_tokens": sampled_action_tokens,
                "decode_dct_to_actions": self._fast_tokenizer.decode_action_dct_coeffs,
                "encode_actions_to_dct": self._fast_tokenizer.encode_action_dct_coeffs,
                "perturb_net": self._perturb_net,
            },
        )
        logprob_s = 0.0
        executed_action_tokens_action_only = self._fast_tokenizer.encode_action_dct_coeffs(explored_dct_coeffs)
        executed_action_tokens_unpadded = self._fast_tokenizer.format_action_tokens_as_output(
            executed_action_tokens_action_only,
            add_eos=True,
        )
        executed_action_tokens, executed_action_token_mask = self._pad_token_sequence(
            executed_action_tokens_unpadded,
            sampled_action_tokens.shape[0],
        )
        if include_logprobs:
            logprob_t0 = time.perf_counter()
            old_token_stats = self._model.recompute_action_logprobs(
                observation,
                jnp.asarray(executed_action_tokens, dtype=jnp.int32)[np.newaxis, ...],
                action_token_mask=jnp.asarray(executed_action_token_mask, dtype=jnp.bool_)[np.newaxis, ...],
            )
            old_token_logprobs = np.asarray(old_token_stats["token_logprobs"][0], dtype=np.float32)
            logprob_s = time.perf_counter() - logprob_t0
        else:
            old_token_logprobs = np.zeros(executed_action_tokens.shape, dtype=np.float32)
        decoded_actions = self._fast_tokenizer.decode_action_dct_coeffs(explored_dct_coeffs)
        decoded_actions = np.asarray(decoded_actions, dtype=np.float32)
        outputs = self._post_extract_output_transform(
            {
                "state": np.asarray(transformed["state"]),
                "actions": decoded_actions,
            }
        )
        action_chunk = np.asarray(outputs["actions"], dtype=np.float32)
        logger.info(
            "sample_chunk_timing mode=standard total=%.3fs keyframe=%.3fs sample_trace=%.3fs recompute_logprobs=%.3fs keyframe_prob=%.4f explored=%s",
            time.perf_counter() - total_start,
            keyframe_s,
            sample_trace_s,
            logprob_s,
            float(keyframe_prob),
            bool(exploration_applied),
        )
        return {
            "action_chunk": action_chunk,
            "action_tokens": executed_action_tokens,
            "action_token_mask": executed_action_token_mask,
            "old_token_logprobs": old_token_logprobs,
            "decoded_actions": np.asarray(decoded_actions, dtype=np.float32),
            "sampled_action_tokens": sampled_action_tokens,
            "sampled_action_token_mask": sampled_action_token_mask,
            "sampled_dct_coeffs": np.asarray(dct_coeffs, dtype=np.float32),
            "dct_coeffs": np.asarray(explored_dct_coeffs, dtype=np.float32),
            "keyframe_prob": float(keyframe_prob),
            "exploration_applied": bool(exploration_applied),
            "transformed_obs": transformed,
        }

    def predict_value(self, obs: dict[str, Any]) -> float:
        observation, _ = self._prepare_observation(obs)
        value = self._model.predict_value(observation)[0]
        return float(np.asarray(value))

    def predict_value_batch(self, obs_batch: list[dict[str, Any]]) -> np.ndarray:
        observation, _ = self._prepare_observations(obs_batch)
        values = self._model.predict_value(observation)
        return np.asarray(values, dtype=np.float32)

    def transform_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Apply policy input transforms and return unbatched model-ready observation dict."""
        _, transformed = self._prepare_observation(obs)
        return transformed

    def predict_keyframe_prob(self, obs: dict[str, Any]) -> float:
        observation, _ = self._prepare_observation(obs)
        prob = self._model.predict_keyframe_prob(observation, stop_gradient=True)[0]
        return float(np.asarray(prob))

    def predict_keyframe_prob_batch(self, obs_batch: list[dict[str, Any]]) -> np.ndarray:
        observation, _ = self._prepare_observations(obs_batch)
        probs = self._model.predict_keyframe_prob(observation, stop_gradient=True)
        return np.asarray(probs, dtype=np.float32)

    def predict_keyframe_class(self, obs: dict[str, Any]) -> int:
        observation, _ = self._prepare_observation(obs)
        klass = self._model.predict_keyframe_class(observation, stop_gradient=True)[0]
        return int(np.asarray(klass))

    def predict_keyframe_class_batch(self, obs_batch: list[dict[str, Any]]) -> np.ndarray:
        observation, _ = self._prepare_observations(obs_batch)
        klass = self._model.predict_keyframe_class(observation, stop_gradient=True)
        return np.asarray(klass, dtype=np.int32)

    def recompute_token_stats(
        self,
        obs: dict[str, Any],
        action_tokens: np.ndarray,
        action_token_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        observation, _ = self._prepare_observation(obs)
        stats = self._model.recompute_action_logprobs(
            observation,
            jnp.asarray(action_tokens, dtype=jnp.int32)[np.newaxis, ...],
            action_token_mask=jnp.asarray(action_token_mask, dtype=jnp.bool_)[np.newaxis, ...],
        )
        return {
            "token_logprobs": np.asarray(stats["token_logprobs"][0], dtype=np.float32),
            "token_entropy": np.asarray(stats["token_entropy"][0], dtype=np.float32),
            "action_token_mask": np.asarray(stats["action_token_mask"][0], dtype=bool),
        }

    @property
    def train_config(self) -> _config.TrainConfig:
        return self._train_config

    @property
    def model(self) -> _pi0_fast.Pi0FAST:
        return self._model

    def sync_model(self, model: _pi0_fast.Pi0FAST) -> None:
        """Replace the rollout/inference model with updated training parameters."""
        self._model = model

    def sync_params(self, graphdef: nnx.GraphDef[_pi0_fast.Pi0FAST], params: nnx.State) -> None:
        self._model = nnx.merge(graphdef, params)


def create_trained_pi0_fast_rl_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: _transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
    exploration_config: KeyframeExplorationConfig | None = None,
    dct_noise_fn: DCTNoiseFn | None = None,
    keyframe_prob_fn: Callable[[dict[str, Any]], float] | None = None,
    perturb_net: Any | None = None,
) -> Pi0FastRLPolicy:
    repack_transforms = repack_transforms or _transforms.Group()
    checkpoint_dir = _download.maybe_download(str(checkpoint_dir))
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(weight_path):
        raise TypeError("Pi0Fast RL policy currently expects the JAX checkpoint format, not a PyTorch checkpoint.")

    if not isinstance(train_config.model, _pi0_fast.Pi0FASTConfig):
        raise TypeError("Online PPO expects a Pi0FAST model config.")
    params = _model.restore_params(pathlib.Path(checkpoint_dir) / "params", dtype=jnp.bfloat16)
    model = train_config.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    partial_params = jax.tree.map(lambda x: x, params)
    state.replace_by_pure_dict(partial_params)
    model = nnx.merge(graphdef, state)
    if not isinstance(model, _pi0_fast.Pi0FAST):
        raise TypeError("create_trained_pi0_fast_rl_policy requires a Pi0FAST train config.")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id)

    input_transforms = [
        *repack_transforms.inputs,
        _transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
        *repack_transforms.outputs,
    ]

    return Pi0FastRLPolicy(
        model,
        train_config=train_config,
        rng=jax.random.key(0),
        transforms=input_transforms,
        output_transforms=output_transforms,
        sample_kwargs=sample_kwargs,
        exploration_config=exploration_config,
        dct_noise_fn=dct_noise_fn,
        keyframe_prob_fn=keyframe_prob_fn,
        perturb_net=perturb_net,
    )
