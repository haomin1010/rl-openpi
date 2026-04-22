from __future__ import annotations

from collections.abc import Sequence
import os
import pathlib
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
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


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
    ) -> None:
        self._model = model
        self._train_config = train_config
        self._rng = rng
        self._input_transform = _transforms.compose(transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._exploration_cfg = exploration_config or KeyframeExplorationConfig()
        self._dct_noise_fn = dct_noise_fn
        self._keyframe_prob_fn = keyframe_prob_fn
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
        observation, transformed = self._prepare_observation(obs)
        self._rng, sample_rng = jax.random.split(self._rng)
        trace = self._model.sample_actions_with_trace(sample_rng, observation, **self._sample_kwargs)

        sampled_action_tokens = np.asarray(trace["tokens"][0], dtype=np.int32)
        sampled_action_token_mask = np.asarray(trace["token_mask"][0], dtype=bool)
        sampled_valid_tokens = self._truncate_by_mask(sampled_action_tokens, sampled_action_token_mask)
        if str(self._exploration_cfg.mode) == "never":
            keyframe_prob = 0.0
        elif self._keyframe_prob_fn is not None:
            keyframe_prob = float(self._keyframe_prob_fn(obs))
        else:
            keyframe_prob = self.predict_keyframe_prob(obs) if getattr(self._model, "keyframe_head", None) is not None else 0.0
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
            },
        )
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
            old_token_stats = self._model.recompute_action_logprobs(
                observation,
                jnp.asarray(executed_action_tokens, dtype=jnp.int32)[np.newaxis, ...],
                action_token_mask=jnp.asarray(executed_action_token_mask, dtype=jnp.bool_)[np.newaxis, ...],
            )
            old_token_logprobs = np.asarray(old_token_stats["token_logprobs"][0], dtype=np.float32)
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

    def transform_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Apply policy input transforms and return unbatched model-ready observation dict."""
        _, transformed = self._prepare_observation(obs)
        return transformed

    def predict_keyframe_prob(self, obs: dict[str, Any]) -> float:
        observation, _ = self._prepare_observation(obs)
        prob = self._model.predict_keyframe_prob(observation, stop_gradient=True)[0]
        return float(np.asarray(prob))

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
    )
