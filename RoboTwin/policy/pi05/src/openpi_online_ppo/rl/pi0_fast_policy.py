from __future__ import annotations

from collections.abc import Sequence
import json
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
from openpi_online_ppo.rl.exploration import DCTNoiseFn, KeyframeExplorationConfig, maybe_apply_dct_exploration
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

logger = logging.getLogger(__name__)
timing_logger = logging.getLogger("openpi_online_ppo.timing")


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


def _state_fingerprint(state: nnx.State) -> float:
    leaves = jax.tree.leaves(state)
    acc = 0.0
    for leaf in leaves:
        arr = np.asarray(leaf)
        if arr.size == 0:
            continue
        acc += float(arr.reshape(-1)[0].astype(np.float64))
    return acc


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
        self._action_prefix_tokens = self._fast_tokenizer.action_prefix_tokens()
        self._action_suffix_tokens = self._fast_tokenizer.action_suffix_tokens(add_eos=True)
        self._action_pg_token_ids = self._fast_tokenizer.action_pg_token_ids()
        self._post_extract_output_transform = _transforms.compose(post_extract_transforms)
        self._compiled_state_version = 0
        self._init_stateful_compiled_model_methods()

    def _init_stateful_compiled_model_methods(self) -> None:
        self._compiled_model_def, self._compiled_model_state = nnx.split(self._model)
        self._compiled_state_version += 1

        def _sample_with_trace_fn(
            state: nnx.State,
            rng: jax.Array,
            observation: _model.Observation,
            *,
            max_decoding_steps: int | jax.Array = 256,
            temperature: float = 0.0,
            selected_vocab_indices: jax.Array | None = None,
        ) -> dict[str, jax.Array]:
            model = nnx.merge(self._compiled_model_def, state)
            return _pi0_fast.Pi0FAST.sample_actions_with_trace(
                model,
                rng,
                observation,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                selected_vocab_indices=selected_vocab_indices,
            )

        def _recompute_logprobs_fn(
            state: nnx.State,
            observation: _model.Observation,
            action_tokens: jax.Array,
            *,
            action_token_mask: jax.Array | None = None,
        ) -> dict[str, jax.Array]:
            model = nnx.merge(self._compiled_model_def, state)
            return _pi0_fast.Pi0FAST.recompute_action_logprobs(
                model,
                observation,
                action_tokens,
                action_token_mask=action_token_mask,
            )

        self._sample_actions_with_trace = jax.jit(_sample_with_trace_fn, static_argnames=("max_decoding_steps", "temperature"))
        self._recompute_action_logprobs = jax.jit(_recompute_logprobs_fn)

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

    def sample_chunk(self, obs: dict[str, Any], *, include_logprobs: bool = True) -> dict[str, Any]:
        return self.sample_chunk_with_options(obs, include_logprobs=include_logprobs)

    def sample_chunk_with_options(
        self,
        obs: dict[str, Any],
        *,
        include_logprobs: bool = True,
        enable_exploration: bool = True,
        keyframe_prob_override: float | None = None,
        use_logprob_lookup: bool = False,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        prepare_obs_t0 = time.perf_counter()
        observation, transformed = self._prepare_observation(obs)
        prepare_observation_s = time.perf_counter() - prepare_obs_t0
        keyframe_s = 0.0
        if keyframe_prob_override is not None:
            keyframe_prob = float(keyframe_prob_override)
        elif str(self._exploration_cfg.mode) == "never":
            keyframe_prob = 0.0
        elif self._keyframe_prob_fn is not None:
            keyframe_t0 = time.perf_counter()
            keyframe_prob = float(self._keyframe_prob_fn(obs))
            keyframe_s = time.perf_counter() - keyframe_t0
        else:
            keyframe_t0 = time.perf_counter()
            keyframe_prob = self.predict_keyframe_prob(obs) if getattr(self._model, "keyframe_head", None) is not None else 0.0
            keyframe_s = time.perf_counter() - keyframe_t0
        sample_t0 = time.perf_counter()
        self._rng, sample_rng = jax.random.split(self._rng)
        trace = self._sample_actions_with_trace(
            self._compiled_model_state,
            sample_rng,
            observation,
            selected_vocab_indices=(
                self._action_pg_token_ids if bool(include_logprobs and use_logprob_lookup and enable_exploration) else None
            ),
            **self._sample_kwargs,
        )
        sample_trace_s = time.perf_counter() - sample_t0
        trace_timings = dict(trace.get("timings", {}))

        sampled_action_tokens = np.asarray(trace["tokens"][0], dtype=np.int32)
        sampled_action_token_mask = np.asarray(trace["token_mask"][0], dtype=bool)
        sampled_valid_tokens = self._truncate_by_mask(sampled_action_tokens, sampled_action_token_mask)
        dct_coeffs = self._fast_tokenizer.extract_action_dct_coeffs(
            sampled_valid_tokens,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            relaxed_decoding=self._exploration_cfg.relaxed_fast_decoding,
        )
        if enable_exploration:
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
        else:
            explored_dct_coeffs = np.asarray(dct_coeffs, dtype=np.float32)
            exploration_applied = False
        logprob_s = 0.0
        lookup_s = 0.0
        executed_action_tokens_action_only = self._fast_tokenizer.encode_action_dct_coeffs(explored_dct_coeffs)
        executed_action_tokens_unpadded = self._fast_tokenizer.format_action_tokens_as_output(
            executed_action_tokens_action_only,
            add_eos=True,
        )
        executed_action_tokens, executed_action_token_mask = self._pad_token_sequence(
            executed_action_tokens_unpadded,
            sampled_action_tokens.shape[0],
        )
        ppo_token_mask = self._build_action_body_mask(executed_action_token_mask)
        if include_logprobs:
            if bool(use_logprob_lookup and enable_exploration and "selected_token_logprobs" in trace):
                lookup_t0 = time.perf_counter()
                selected_logprobs = np.asarray(trace["selected_token_logprobs"][0], dtype=np.float32)
                old_token_logprobs = np.zeros(executed_action_tokens.shape, dtype=np.float32)
                body_start = int(self._action_prefix_tokens.shape[0])
                valid_len = int(np.sum(executed_action_token_mask.astype(np.int32)))
                body_end = max(body_start, valid_len - int(self._action_suffix_tokens.shape[0]))
                if body_end > body_start:
                    body_fast_tokens = self._fast_tokenizer.pg_tokens_to_fast_tokens(
                        executed_action_tokens[body_start:body_end]
                    )
                    gathered = np.take_along_axis(
                        selected_logprobs[body_start:body_end],
                        body_fast_tokens[:, None],
                        axis=-1,
                    )[:, 0]
                    old_token_logprobs[body_start:body_end] = gathered.astype(np.float32, copy=False)
                lookup_s = time.perf_counter() - lookup_t0
            elif not enable_exploration:
                old_token_logprobs = np.where(
                    ppo_token_mask,
                    np.asarray(trace["token_logprobs"][0], dtype=np.float32),
                    0.0,
                ).astype(np.float32)
            else:
                logprob_t0 = time.perf_counter()
                old_token_stats = self._recompute_action_logprobs(
                    self._compiled_model_state,
                    observation,
                    jnp.asarray(executed_action_tokens, dtype=jnp.int32)[np.newaxis, ...],
                    action_token_mask=jnp.asarray(ppo_token_mask, dtype=jnp.bool_)[np.newaxis, ...],
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
        if not timing_logger.disabled:
            timing_logger.info(
                json.dumps(
                    {
                        "event": "policy_sample_chunk",
                        "total_s": time.perf_counter() - total_start,
                        "prepare_observation_s": prepare_observation_s,
                        "keyframe_s": keyframe_s,
                        "sample_trace_s": sample_trace_s,
                        "prepare_decode_prefix_s": float(trace_timings.get("prepare_decode_prefix_s", 0.0)),
                        "decode_loop_s": float(trace_timings.get("decode_loop_s", 0.0)),
                        "recompute_logprobs_s": logprob_s,
                        "lookup_logprobs_s": lookup_s,
                        "lookup_used": bool(
                            use_logprob_lookup and enable_exploration and "selected_token_logprobs" in trace
                        ),
                        "lookup_table_bytes": int(
                            np.asarray(trace["selected_token_logprobs"]).nbytes
                            if "selected_token_logprobs" in trace
                            else 0
                        ),
                        "policy_state_version": int(self._compiled_state_version),
                        "policy_state_fingerprint": float(_state_fingerprint(self._compiled_model_state)),
                        "ppo_token_count": int(np.sum(ppo_token_mask.astype(np.int32))),
                        "keyframe_prob": float(keyframe_prob),
                        "exploration_applied": bool(exploration_applied),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        return {
            "action_chunk": action_chunk,
            "action_tokens": executed_action_tokens,
            "action_token_mask": ppo_token_mask,
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

    def _build_action_body_mask(self, full_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(full_mask, dtype=bool).reshape(-1)
        out = np.zeros_like(mask, dtype=bool)
        valid_len = int(np.sum(mask.astype(np.int32)))
        prefix_len = int(self._action_prefix_tokens.shape[0])
        suffix_len = int(self._action_suffix_tokens.shape[0])
        body_start = min(prefix_len, valid_len)
        body_end = max(body_start, valid_len - suffix_len)
        if body_end > body_start:
            out[body_start:body_end] = True
        return out

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
        self._init_stateful_compiled_model_methods()

    def sync_params(self, graphdef: nnx.GraphDef[_pi0_fast.Pi0FAST], params: nnx.State) -> None:
        self._model = nnx.merge(graphdef, params)
        self._compiled_model_state = params
        self._compiled_state_version += 1


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
