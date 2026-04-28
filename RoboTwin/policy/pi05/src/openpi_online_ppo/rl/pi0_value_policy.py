from __future__ import annotations

from collections.abc import Sequence
import pathlib
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import weight_loaders as _weight_loaders
from openpi_online_ppo.models import pi0_aux as _pi0_aux
import openpi.shared.download as _download


class Pi0ValuePolicy:
    def __init__(
        self,
        model: _pi0_aux.Pi0Aux,
        *,
        train_config: _config.TrainConfig,
        transforms: Sequence[_transforms.DataTransformFn],
    ) -> None:
        self._model = model
        self._train_config = train_config
        self._input_transform = _transforms.compose(transforms)

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

    def transform_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        _, transformed = self._prepare_observation(obs)
        return transformed

    @property
    def model(self) -> _pi0_aux.Pi0Aux:
        return self._model

    @property
    def train_config(self) -> _config.TrainConfig:
        return self._train_config

    def predict_value(self, obs: dict[str, Any]) -> float:
        observation, _ = self._prepare_observation(obs)
        value = self._model.predict_value(observation, stop_gradient=True)[0]
        return float(np.asarray(value))

    def predict_value_batch(self, obs_batch: list[dict[str, Any]]) -> np.ndarray:
        observation, _ = self._prepare_observations(obs_batch)
        values = self._model.predict_value(observation, stop_gradient=True)
        return np.asarray(values, dtype=np.float32)

    def predict_keyframe_prob(self, obs: dict[str, Any]) -> float:
        observation, _ = self._prepare_observation(obs)
        prob = self._model.predict_keyframe_prob(observation, stop_gradient=True)[0]
        return float(np.asarray(prob))

    def predict_keyframe_prob_batch(self, obs_batch: list[dict[str, Any]]) -> np.ndarray:
        observation, _ = self._prepare_observations(obs_batch)
        probs = self._model.predict_keyframe_prob(observation, stop_gradient=True)
        return np.asarray(probs, dtype=np.float32)

    def sync_model(self, model: _pi0_aux.Pi0Aux) -> None:
        self._model = model


def _load_params_with_aux_init(
    train_config: _config.TrainConfig,
    ref_params: dict[str, Any],
    checkpoint_dir: pathlib.Path | str | None,
) -> dict[str, Any]:
    if checkpoint_dir:
        checkpoint_dir = _download.maybe_download(str(checkpoint_dir))
        raw_params = _model.restore_params(pathlib.Path(checkpoint_dir) / "params", restore_type=np.ndarray)
        return _weight_loaders._merge_params(raw_params, ref_params, missing_regex=".*")

    loader = train_config.weight_loader
    if isinstance(loader, _weight_loaders.CheckpointWeightLoader):
        raw_params = _model.restore_params(_download.maybe_download(loader.params_path), restore_type=np.ndarray)
        return _weight_loaders._merge_params(raw_params, ref_params, missing_regex=".*")
    return loader.load(ref_params)


def create_pi0_value_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str | None,
    *,
    repack_transforms: _transforms.Group | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
) -> Pi0ValuePolicy:
    repack_transforms = repack_transforms or _transforms.Group()
    model = train_config.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    params = _load_params_with_aux_init(train_config, state.to_pure_dict(), checkpoint_dir)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    if not isinstance(model, _pi0_aux.Pi0Aux):
        raise TypeError("create_pi0_value_policy requires a Pi0Aux train config.")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if checkpoint_dir and data_config.asset_id is not None:
            norm_stats = _checkpoints.load_norm_stats(pathlib.Path(_download.maybe_download(str(checkpoint_dir))) / "assets", data_config.asset_id)
        else:
            norm_stats = data_config.norm_stats

    transforms = [
        *repack_transforms.inputs,
        _transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
    ]
    if norm_stats is not None:
        transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms.extend(data_config.model_transforms.inputs)

    return Pi0ValuePolicy(
        model,
        train_config=train_config,
        transforms=transforms,
    )
