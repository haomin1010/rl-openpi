#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import math
import multiprocessing
import os
import pathlib
import pickle
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config as _base_pi0
from openpi.models import pi0_fast as _base_pi0_fast
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as train_config
from openpi_client import msgpack_numpy
from openpi_online_ppo.data.local_lerobot_loader import ensure_local_hf_cache
from openpi_online_ppo.models import pi0_aux as _rl_pi0
from openpi_online_ppo.models import pi0_fast_rl as _rl_pi0_fast
from openpi_online_ppo.rl.pi0_value_policy import create_pi0_value_policy
from openpi_online_ppo.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy

log = logging.getLogger("value_mc_ws")


def _collate_tree(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(_: int) -> None:
    # Avoid aggressive GPU memory preallocation in worker processes.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _as_numpy_static(x: Any) -> np.ndarray:
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _unflatten_obs_t_row_static(row: dict[str, Any]) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for k, v in row.items():
        if not k.startswith("obs_t."):
            continue
        parts = k.split(".")[1:]
        node = obs
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = _as_numpy_static(v)
    return obs


def _row_to_worker_payload(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, str):
            normalized[k] = v
        else:
            arr = _as_numpy_static(v)
            # Aux value/keyframe training consumes single-frame LeRobot rows, while
            # the shared SFT transforms expect action chunks shaped [T, D]. Promote
            # a single action vector to a length-1 chunk so transforms like
            # AlohaInputs/DeltaActions can still run without broadcasting errors.
            if k in {"action", "actions"} and arr.ndim == 1:
                arr = arr[np.newaxis, ...]
            normalized[k] = arr
    return normalized


def _build_input_transform(
    cfg: train_config.TrainConfig,
    *,
    tasks: dict[int, str] | None,
    checkpoint_dir: str | None,
) -> _transforms.DataTransformFn:
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = data_config.norm_stats
    if checkpoint_dir and data_config.asset_id is not None:
        ckpt_path = pathlib.Path(checkpoint_dir)
        assets_dir = ckpt_path / "assets"
        if assets_dir.exists():
            norm_stats = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)

    transforms: list[_transforms.DataTransformFn] = []
    if data_config.prompt_from_task:
        if tasks is None:
            raise ValueError("Prompt-from-task transform requires dataset task metadata.")
        transforms.append(_transforms.PromptFromLeRobotTask(tasks))
    transforms.extend(data_config.repack_transforms.inputs)
    transforms.extend(data_config.data_transforms.inputs)
    transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms.extend(data_config.model_transforms.inputs)
    return _transforms.compose(transforms)


def _row_to_transformed_obs(
    row: dict[str, Any],
    input_transform: _transforms.DataTransformFn,
) -> dict[str, Any]:
    if any(k.startswith("obs_t.") for k in row.keys()):
        return _unflatten_obs_t_row_static(row)
    obs = dict(input_transform(_row_to_worker_payload(row)))
    # Observation.from_dict consumes observation-side fields only.
    obs.pop("actions", None)
    return obs


def _to_rl_train_config(cfg: train_config.TrainConfig) -> train_config.TrainConfig:
    if isinstance(cfg.model, _base_pi0_fast.Pi0FASTConfig):
        base = cfg.model
        rl_model = _rl_pi0_fast.Pi0FASTConfig(
            dtype=base.dtype,
            paligemma_variant=base.paligemma_variant,
            action_dim=base.action_dim,
            action_horizon=base.action_horizon,
            max_token_len=base.max_token_len,
            fast_model_tokenizer=base.fast_model_tokenizer,
            fast_model_tokenizer_kwargs=base.fast_model_tokenizer_kwargs,
            use_value_head=True,
            use_keyframe_head=True,
            keyframe_num_bins=max(2, int(base.action_horizon) // 2),
        )
        return dataclasses.replace(cfg, model=rl_model)
    if isinstance(cfg.model, _base_pi0.Pi0Config):
        base = cfg.model
        rl_model = _rl_pi0.Pi0AuxConfig(
            dtype=base.dtype,
            paligemma_variant=base.paligemma_variant,
            action_expert_variant=base.action_expert_variant,
            action_dim=base.action_dim,
            action_horizon=base.action_horizon,
            max_token_len=base.max_token_len,
            pi05=base.pi05,
            discrete_state_input=base.discrete_state_input,
            use_value_head=True,
            use_keyframe_head=True,
            keyframe_num_bins=max(2, int(base.action_horizon) // 2),
        )
        return dataclasses.replace(cfg, model=rl_model)
    raise TypeError(f"Config `{cfg.name}` is not a supported Pi0FAST/Pi0/Pi05 config.")


class _LerobotTargetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_ds: Any,
        input_transform: _transforms.DataTransformFn,
        targets: np.ndarray,
    ) -> None:
        self._base_ds = base_ds
        self._input_transform = input_transform
        self._targets = np.asarray(targets)

    def __len__(self) -> int:
        return len(self._base_ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._base_ds[int(idx)]
        row = {k: item[k] for k in item.keys()}
        obs = _row_to_transformed_obs(row, self._input_transform)
        return {
            "obs": obs,
            "target": np.asarray(self._targets[int(idx)]),
        }


class _LerobotKeyframeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_ds: Any,
        input_transform: _transforms.DataTransformFn,
        *,
        ann_eps: dict[int, list[int]],
        num_bins: int,
    ) -> None:
        self._base_ds = base_ds
        self._input_transform = input_transform
        self._ann_eps = ann_eps
        self._num_bins = int(num_bins)

    def __len__(self) -> int:
        return len(self._base_ds)

    def _nearest_distance_bin(self, frame_idx: int, keyframes: list[int]) -> int:
        if not keyframes:
            return self._num_bins - 1
        nearest = min(abs(int(frame_idx) - int(kf)) for kf in keyframes)
        return int(min(max(0, nearest), self._num_bins - 1))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._base_ds[int(idx)]
        row = {k: item[k] for k in item.keys()}
        ep = int(np.asarray(row["episode_index"]).item())
        fi = int(np.asarray(row["frame_index"]).item())
        label = self._nearest_distance_bin(fi, self._ann_eps.get(ep, []))
        obs = _row_to_transformed_obs(row, self._input_transform)
        return {
            "obs": obs,
            "target": np.asarray(label, dtype=np.int32),
        }


class ValueMCService:
    def __init__(
        self,
        *,
        policy_config: str,
        policy_path: str,
        mc_epochs: int,
        mc_batch_size: int,
        mc_gamma: float,
        keyframe_epochs: int,
        keyframe_batch_size: int,
        keyframe_chunk_size: int,
        num_workers: int,
        persistent_workers: bool,
        value_target_mode: str,
        value_c_fail_coef: float,
        value_length_scale_quantile: float,
        state_ckpt_dir: str | None,
        init_state_file: str | None,
    ) -> None:
        cfg = _to_rl_train_config(train_config.get_config(policy_config))
        log.info("loading value/keyframe policy config=%s checkpoint=%s", policy_config, policy_path or "<config-default>")
        if isinstance(cfg.model, _rl_pi0_fast.Pi0FASTConfig):
            if not policy_path:
                raise ValueError("Pi0FAST value service requires --policy.path.")
            policy = create_trained_pi0_fast_rl_policy(cfg, policy_path)
        elif isinstance(cfg.model, _rl_pi0.Pi0AuxConfig):
            policy = create_pi0_value_policy(cfg, policy_path or None)
        else:
            raise TypeError(f"Unsupported model type for value service: {type(cfg.model).__name__}")
        log.info("RL policy loaded; initializing value/keyframe service state")
        self._model = policy.model
        self._train_config = cfg
        self._checkpoint_dir = str(pathlib.Path(policy_path).resolve()) if policy_path else None
        # Freeze backbone; train only lightweight aux input + task heads.
        self._value_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*(value_head|aux_cls_embed).*"))
        self._keyframe_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*(keyframe_head|aux_cls_embed).*"))
        self._tx = cfg.optimizer.create(cfg.lr_schedule.create(), weight_decay_mask=None)
        model_state = nnx.state(self._model)
        self._value_opt_state = self._tx.init(model_state.filter(self._value_filter))
        self._keyframe_opt_state = self._tx.init(model_state.filter(self._keyframe_filter))
        self._mc_epochs = max(1, int(mc_epochs))
        self._mc_batch_size = max(1, int(mc_batch_size))
        self._mc_gamma = float(mc_gamma)
        self._keyframe_epochs = max(1, int(keyframe_epochs))
        self._keyframe_batch_size = max(1, int(keyframe_batch_size))
        self._keyframe_chunk_size = max(2, int(keyframe_chunk_size))
        self._num_workers = max(0, int(num_workers))
        self._persistent_workers = bool(persistent_workers) and self._num_workers > 0
        self._value_target_mode = str(value_target_mode)
        self._value_c_fail_coef = float(value_c_fail_coef)
        self._value_length_scale_quantile = float(value_length_scale_quantile)
        self._state_ckpt_dir = pathlib.Path(state_ckpt_dir).resolve() if state_ckpt_dir else None

        if init_state_file:
            self.load_state_from_file(init_state_file, load_optimizer_state=True)

    @staticmethod
    def _resolve_success_bool(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value) != 0
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"success", "succeeded", "true", "1", "yes"}:
                return True
            if v in {"failure", "failed", "false", "0", "no"}:
                return False
        return bool(value)

    @staticmethod
    def _compute_global_length_scale(ep_to_indices: dict[int, list[int]]) -> float:
        lengths = [len(rows) for rows in ep_to_indices.values()]
        if not lengths:
            raise ValueError("No episode lengths available to compute global length scale.")
        lengths_np = np.asarray(lengths, dtype=np.float32)
        if np.any(lengths_np <= 0):
            raise ValueError("Invalid non-positive episode lengths in dataset.")
        return float(np.max(lengths_np))

    def predict(self, transformed_obs: dict[str, Any]) -> float:
        observation = _model.Observation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed_obs)
        )
        value = self._model.predict_value(observation)[0]
        return float(np.asarray(value))

    def predict_keyframe(self, transformed_obs: dict[str, Any]) -> float:
        observation = _model.Observation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed_obs)
        )
        prob = self._model.predict_keyframe_prob(observation, stop_gradient=True)[0]
        return float(np.asarray(prob))

    def sync_params_from_file(self, params_file: str) -> None:
        path = pathlib.Path(params_file)
        with path.open("rb") as f:
            pure = pickle.load(f)
        state = nnx.state(self._model)
        state.replace_by_pure_dict(pure)
        nnx.update(self._model, state)

    @staticmethod
    def _dump_pickle_atomic(path: pathlib.Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def save_state_to_file(self, state_file: str, *, include_optimizer_state: bool = True) -> str:
        out = pathlib.Path(state_file).resolve()
        payload: dict[str, Any] = {
            "format": "openpi_value_mc_state_v1",
            "params_pure": nnx.state(self._model).to_pure_dict(),
        }
        if include_optimizer_state:
            payload["value_opt_state"] = self._value_opt_state
            payload["keyframe_opt_state"] = self._keyframe_opt_state
        self._dump_pickle_atomic(out, payload)
        log.info("saved value/keyframe state: %s", out)
        return str(out)

    def load_state_from_file(self, state_file: str, *, load_optimizer_state: bool = True) -> None:
        path = pathlib.Path(state_file).resolve()
        with path.open("rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and "params_pure" in payload:
            pure = payload["params_pure"]
            value_opt_state = payload.get("value_opt_state")
            keyframe_opt_state = payload.get("keyframe_opt_state")
        else:
            # Backward-compatible: raw pure params dict.
            pure = payload
            value_opt_state = None
            keyframe_opt_state = None

        state = nnx.state(self._model)
        state.replace_by_pure_dict(pure)
        nnx.update(self._model, state)

        if load_optimizer_state:
            if value_opt_state is not None:
                self._value_opt_state = value_opt_state
            if keyframe_opt_state is not None:
                self._keyframe_opt_state = keyframe_opt_state
        log.info("loaded value/keyframe state: %s", path)

    def _autosave_after_train(self, stage: str) -> str | None:
        if self._state_ckpt_dir is None:
            return None
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        latest = self._state_ckpt_dir / "latest.pkl"
        tagged = self._state_ckpt_dir / f"{stage}_{now}.pkl"
        latest_path = self.save_state_to_file(str(latest), include_optimizer_state=True)
        self.save_state_to_file(str(tagged), include_optimizer_state=True)
        return latest_path

    @staticmethod
    def _as_float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        return {str(k): float(np.asarray(v)) for k, v in metrics.items()}

    @staticmethod
    def _should_log_step_timing(step_i: int, total_steps: int) -> bool:
        if step_i <= 5:
            return True
        if step_i == total_steps:
            return True
        return step_i % 20 == 0

    @staticmethod
    def _log_step_metrics(stage: str, epoch: int, total_epochs: int, step_i: int, total_steps: int, metrics: dict[str, float]) -> None:
        metric_parts = []
        for key in (
            "keyframe_loss",
            "keyframe_acc",
            "keyframe_distance_mae",
            "keyframe_grad_norm",
            "value_loss",
            "value_mae",
            "value_grad_norm",
        ):
            if key in metrics:
                metric_parts.append(f"{key}={metrics[key]:.6f}")
        if not metric_parts:
            return
        log.info(
            "%s metrics epoch=%d/%d step=%d/%d %s",
            stage,
            epoch,
            total_epochs,
            step_i,
            total_steps,
            " ".join(metric_parts),
        )

    def _train_one_batch(self, obs_batch: dict[str, jnp.ndarray], targets: jnp.ndarray) -> dict[str, float]:
        model = self._model

        def loss_fn(m):
            observation = _model.Observation.from_dict(obs_batch)
            logits = m.predict_value_logits(observation, stop_gradient=False)
            soft_target = m.project_values_to_bins(targets)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            per_sample_loss = -jnp.sum(soft_target * log_probs, axis=-1)
            value_loss = jnp.mean(per_sample_loss)
            centers = m.value_bin_centers()
            probs = jax.nn.softmax(logits, axis=-1)
            pred_value = jnp.sum(probs * centers[None, :], axis=-1)
            value_mae = jnp.mean(jnp.abs(pred_value - targets))
            return value_loss, {"value_loss": value_loss, "value_mae": value_mae}

        diff_state = nnx.DiffState(0, self._value_filter)
        (_, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
        params = nnx.state(model).filter(self._value_filter)
        updates, self._value_opt_state = self._tx.update(grads, self._value_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        metrics = {
            **metrics,
            "value_grad_norm": optax.global_norm(grads),
        }
        return self._as_float_metrics(metrics)

    def _train_one_keyframe_batch(self, obs_batch: dict[str, jnp.ndarray], labels: jnp.ndarray) -> dict[str, float]:
        model = self._model

        def loss_fn(m):
            observation = _model.Observation.from_dict(obs_batch)
            logits = m.predict_keyframe_logits(observation, stop_gradient=False)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            per_sample_loss = -jnp.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0]
            keyframe_loss = jnp.mean(per_sample_loss)
            pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            acc = jnp.mean((pred == labels).astype(jnp.float32))
            distance_mae = jnp.mean(jnp.abs(pred - labels).astype(jnp.float32))
            return keyframe_loss, {
                "keyframe_loss": keyframe_loss,
                "keyframe_acc": acc,
                "keyframe_distance_mae": distance_mae,
            }

        diff_state = nnx.DiffState(0, self._keyframe_filter)
        (_, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
        params = nnx.state(model).filter(self._keyframe_filter)
        updates, self._keyframe_opt_state = self._tx.update(grads, self._keyframe_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        metrics = {
            **metrics,
            "keyframe_grad_norm": optax.global_norm(grads),
        }
        return self._as_float_metrics(metrics)

    def train_mc_from_file(self, dataset_file: str) -> dict[str, float]:
        path = pathlib.Path(dataset_file)
        log.info("train_mc_from_file start dataset_file=%s", path)
        with path.open("rb") as f:
            dataset = pickle.load(f)
        if not isinstance(dataset, list) or len(dataset) == 0:
            log.info("train_mc_from_file empty dataset.")
            return {"value_loss": 0.0, "value_mae": 0.0, "value_grad_norm": 0.0, "num_samples": 0.0}

        obs_list = [x["obs_t"] for x in dataset]
        ret_arr = np.asarray([float(x["return"]) for x in dataset], dtype=np.float32)
        idx = np.arange(ret_arr.shape[0], dtype=np.int32)

        all_metrics: list[dict[str, float]] = []
        rng = np.random.default_rng()
        total_steps = int(np.ceil(idx.shape[0] / self._mc_batch_size))
        log.info(
            "train_mc_from_file dataset_size=%d epochs=%d batch_size=%d steps_per_epoch=%d",
            idx.shape[0],
            self._mc_epochs,
            self._mc_batch_size,
            total_steps,
        )
        for epoch in range(self._mc_epochs):
            rng.shuffle(idx)
            for step_i, i in enumerate(range(0, idx.shape[0], self._mc_batch_size), start=1):
                j = idx[i : i + self._mc_batch_size]
                batch_obs = jax.tree.map(
                    lambda *xs: jnp.asarray(np.stack(xs, axis=0)),
                    *[obs_list[int(k)] for k in j.tolist()],
                )
                batch_ret = jnp.asarray(ret_arr[j], dtype=jnp.float32)
                metrics = self._train_one_batch(batch_obs, batch_ret)
                all_metrics.append(metrics)
                if step_i == 1 or step_i % 20 == 0 or step_i == total_steps:
                    log.info("train_mc_from_file epoch=%d/%d step=%d/%d", epoch + 1, self._mc_epochs, step_i, total_steps)
                    self._log_step_metrics("train_mc_from_file", epoch + 1, self._mc_epochs, step_i, total_steps, metrics)

        mean = {}
        for k in all_metrics[0]:
            mean[k] = float(np.mean([m[k] for m in all_metrics]))
        mean["num_samples"] = float(len(dataset))
        saved_to = self._autosave_after_train("train_mc_from_file")
        if saved_to:
            mean["saved_state_file"] = saved_to
        log.info("train_mc_from_file done metrics=%s", mean)
        return mean

    @staticmethod
    def _batch_to_model_inputs(batch_obs: dict[str, Any]) -> dict[str, jnp.ndarray]:
        return jax.tree.map(lambda x: jnp.asarray(x), batch_obs)

    @staticmethod
    def _load_value_rows_cache(cache_path: pathlib.Path, *, expected_rows: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, dict[int, str], np.ndarray | None
    ] | None:
        if not cache_path.exists():
            return None
        ep: list[int] = []
        fi: list[int] = []
        task_index: list[int] = []
        task_name_by_idx: dict[int, str] = {}
        has_mc = False
        mc_values: list[float] = []
        with cache_path.open("r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep.append(int(rec["episode_index"]))
                fi.append(int(rec["frame_index"]))
                ti = rec.get("task_index")
                task_index.append(int(ti) if ti is not None else -1)
                task_text = rec.get("task")
                if ti is None and task_text is not None:
                    # Keep the same semantics as the online scan fallback:
                    # map dataset row index -> task string when task_index is unavailable.
                    task_name_by_idx[int(row_idx)] = str(task_text)
                if rec.get("mc_return") is not None:
                    has_mc = True
                    mc_values.append(float(rec["mc_return"]))
                else:
                    mc_values.append(0.0)
        if len(ep) != expected_rows:
            log.warning(
                "value rows cache length mismatch: cache=%d expected=%d path=%s; fallback to online scan",
                len(ep),
                expected_rows,
                cache_path,
            )
            return None
        mc_arr = np.asarray(mc_values, dtype=np.float32) if has_mc else None
        return (
            np.asarray(ep, dtype=np.int32),
            np.asarray(fi, dtype=np.int32),
            np.asarray(task_index, dtype=np.int32),
            task_name_by_idx,
            mc_arr,
        )

    def train_mc_from_lerobot(self, *, dataset_root: str, repo_id: str) -> dict[str, float]:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        log.info("train_mc_from_lerobot start dataset_root=%s repo_id=%s", dataset_root, repo_id)
        ensure_local_hf_cache()
        ds = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        num_rows = len(ds)
        if num_rows == 0:
            log.info("train_mc_from_lerobot empty dataset.")
            return {"value_loss": 0.0, "value_mae": 0.0, "value_grad_norm": 0.0, "num_samples": 0.0}

        # Metadata-only pre-pass: prefer cache file generated at merge time.
        ep_arr = np.zeros((num_rows,), dtype=np.int32)
        fi_arr = np.zeros((num_rows,), dtype=np.int32)
        task_index_arr = np.full((num_rows,), -1, dtype=np.int32)
        task_name_by_idx: dict[int, str] = {}
        has_mc_return = False
        mc_return_arr: np.ndarray | None = None
        cache_path = pathlib.Path(ds.root) / "meta" / "value_train_rows.jsonl"
        cache_loaded = self._load_value_rows_cache(cache_path, expected_rows=num_rows)
        if cache_loaded is not None:
            ep_arr, fi_arr, task_index_arr, task_name_by_idx, mc_return_arr = cache_loaded
            has_mc_return = mc_return_arr is not None
            log.info("train_mc_from_lerobot loaded metadata cache: %s", cache_path)
        else:
            for i in range(num_rows):
                item = ds[i]
                ep_arr[i] = int(np.asarray(item["episode_index"]).item())
                fi_arr[i] = int(np.asarray(item["frame_index"]).item())
                if "task_index" in item:
                    task_index_arr[i] = int(np.asarray(item["task_index"]).item())
                else:
                    task_name_by_idx[i] = str(item.get("task", ""))
                if "mc_return" in item:
                    has_mc_return = True
                    if mc_return_arr is None:
                        mc_return_arr = np.zeros((num_rows,), dtype=np.float32)
                    ret = item["mc_return"]
                    if hasattr(ret, "numpy"):
                        ret = ret.numpy()
                    ret = np.asarray(ret, dtype=np.float32).reshape(-1)
                    mc_return_arr[i] = float(ret[0])

        target_arr = np.zeros((num_rows,), dtype=np.float32)
        ep_to_indices: dict[int, list[int]] = {}
        for i in range(num_rows):
            ep_to_indices.setdefault(int(ep_arr[i]), []).append(i)

        if self._value_target_mode == "evorl_normalized":
            outcome_path = pathlib.Path(ds.root) / "meta" / "online_episode_outcomes.jsonl"
            if not outcome_path.exists():
                raise RuntimeError(
                    f"EvoRL-style target needs episode outcomes file: {outcome_path}. "
                    "Please enable env-side outcome logging."
                )
            outcome_by_ep: dict[int, bool] = {}
            with outcome_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    outcome_by_ep[int(rec["episode_index"])] = self._resolve_success_bool(rec.get("success", False))

            c_fail_coef = float(self._value_c_fail_coef)
            if c_fail_coef < 0:
                raise ValueError("'value_c_fail_coef' must be non-negative.")
            clip_min = float(getattr(self._train_config.model, "value_bin_min", -1.0))
            clip_max = float(getattr(self._train_config.model, "value_bin_max", 0.0))
            global_scale = self._compute_global_length_scale(ep_to_indices)

            log.info(
                "train_mc_from_lerobot EvoRL-target mode: episodes=%d global_scale=%.3f c_fail_coef=%.4f clip=[%.3f, %.3f] "
                "(treating whole dataset as one task; value_length_scale_quantile=%.3f ignored)",
                len(ep_to_indices),
                global_scale,
                c_fail_coef,
                clip_min,
                clip_max,
                self._value_length_scale_quantile,
            )
            for ep in sorted(ep_to_indices.keys()):
                rows_ep = sorted(ep_to_indices[ep], key=lambda idx: int(fi_arr[idx]))
                success = bool(outcome_by_ep.get(ep, False))
                c_fail = float(global_scale)
                ep_len = len(rows_ep)
                for pos, ds_idx in enumerate(rows_ep):
                    remaining_steps = float(ep_len - int(pos) - 1)
                    g = -float(remaining_steps)
                    if not success:
                        g -= (c_fail_coef**remaining_steps) * c_fail
                    denom = float(global_scale) + c_fail
                    target = float(np.clip(g / denom, clip_min, clip_max))
                    target_arr[int(ds_idx)] = target
        elif has_mc_return:
            log.info("train_mc_from_lerobot using mc_return field.")
            if mc_return_arr is None:
                raise RuntimeError("Detected mc_return fields but failed to build mc_return array.")
            target_arr = mc_return_arr.astype(np.float32, copy=False)
        else:
            log.info("train_mc_from_lerobot mc_return missing, building returns from outcome jsonl.")
            outcome_path = pathlib.Path(ds.root) / "meta" / "online_episode_outcomes.jsonl"
            if not outcome_path.exists():
                raise RuntimeError(
                    f"`mc_return` not found in dataset and outcomes file missing: {outcome_path}. "
                    "Please enable env-side outcome logging or provide mc_return labels."
                )
            outcome_by_ep: dict[int, bool] = {}
            with outcome_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    outcome_by_ep[int(rec["episode_index"])] = bool(rec["success"])

            gamma = float(self._mc_gamma)
            for ep in sorted(ep_to_indices.keys()):
                rows_ep = sorted(ep_to_indices[ep], key=lambda idx: int(fi_arr[idx]))
                success = outcome_by_ep.get(ep, False)
                final_reward = 0.0 if success else -100.0
                returns = [0.0] * len(rows_ep)
                if rows_ep:
                    returns[-1] = final_reward
                    for idx in range(len(rows_ep) - 2, -1, -1):
                        returns[idx] = -1.0 + gamma * returns[idx + 1]
                for ds_idx, ret in zip(rows_ep, returns):
                    target_arr[int(ds_idx)] = float(ret)

        tasks = getattr(getattr(ds, "meta", None), "tasks", None)
        input_transform = _build_input_transform(
            self._train_config,
            tasks=tasks,
            checkpoint_dir=self._checkpoint_dir,
        )
        dataset = _LerobotTargetDataset(ds, input_transform, target_arr)
        all_metrics: list[dict[str, float]] = []
        total_steps = int(np.ceil(float(num_rows) / float(self._mc_batch_size)))
        log.info(
            "train_mc_from_lerobot prepared_samples=%d epochs=%d batch_size=%d steps_per_epoch=%d",
            num_rows,
            self._mc_epochs,
            self._mc_batch_size,
            total_steps,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._mc_batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
            multiprocessing_context=(multiprocessing.get_context("spawn") if self._num_workers > 0 else None),
            worker_init_fn=_worker_init_fn if self._num_workers > 0 else None,
            collate_fn=_collate_tree,
            drop_last=False,
        )
        for epoch in range(self._mc_epochs):
            data_iter = iter(loader)
            for step_i in range(1, total_steps + 1):
                batch = next(data_iter)
                batch_obs = self._batch_to_model_inputs(batch["obs"])
                batch_ret = jnp.asarray(batch["target"], dtype=jnp.float32)
                metrics = self._train_one_batch(batch_obs, batch_ret)
                all_metrics.append(metrics)
                if self._should_log_step_timing(step_i, total_steps):
                    log.info("train_mc_from_lerobot epoch=%d/%d step=%d/%d", epoch + 1, self._mc_epochs, step_i, total_steps)
                    self._log_step_metrics("train_mc_from_lerobot", epoch + 1, self._mc_epochs, step_i, total_steps, metrics)

        mean = {}
        for k in all_metrics[0]:
            mean[k] = float(np.mean([m[k] for m in all_metrics]))
        mean["num_samples"] = float(num_rows)
        saved_to = self._autosave_after_train("train_mc_from_lerobot")
        if saved_to:
            mean["saved_state_file"] = saved_to
        log.info("train_mc_from_lerobot done metrics=%s", mean)
        return mean

    def train_keyframe_from_lerobot(
        self,
        *,
        dataset_root: str,
        repo_id: str,
        annotations_json: str,
    ) -> dict[str, float]:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        log.info(
            "train_keyframe_from_lerobot start dataset_root=%s repo_id=%s annotations_json=%s",
            dataset_root,
            repo_id,
            annotations_json,
        )
        ann = json.loads(pathlib.Path(annotations_json).read_text(encoding="utf-8"))
        ann_eps = {int(k): sorted(set(int(x) for x in v)) for k, v in ann.get("episodes", {}).items()}

        ensure_local_hf_cache()
        ds = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        if len(ds) == 0:
            log.info("train_keyframe_from_lerobot empty dataset.")
            return {
                "keyframe_loss": 0.0,
                "keyframe_acc": 0.0,
                "keyframe_distance_mae": 0.0,
                "keyframe_grad_norm": 0.0,
                "num_samples": 0.0,
            }

        half_chunk = max(1, int(self._keyframe_chunk_size // 2))
        num_bins = int(getattr(self._train_config.model, "keyframe_num_bins", 0) or max(2, half_chunk))

        num_rows = len(ds)
        tasks = getattr(getattr(ds, "meta", None), "tasks", None)
        input_transform = _build_input_transform(
            self._train_config,
            tasks=tasks,
            checkpoint_dir=self._checkpoint_dir,
        )
        dataset = _LerobotKeyframeDataset(
            ds,
            input_transform,
            ann_eps=ann_eps,
            num_bins=num_bins,
        )
        all_metrics: list[dict[str, float]] = []
        total_steps = int(np.ceil(float(num_rows) / float(self._keyframe_batch_size)))
        log.info(
            (
                "train_keyframe_from_lerobot prepared_samples=%d "
                "chunk_size=%d bins=%d "
                "epochs=%d batch_size=%d steps_per_epoch=%d"
            ),
            num_rows,
            self._keyframe_chunk_size,
            num_bins,
            self._keyframe_epochs,
            self._keyframe_batch_size,
            total_steps,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._keyframe_batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
            multiprocessing_context=(multiprocessing.get_context("spawn") if self._num_workers > 0 else None),
            worker_init_fn=_worker_init_fn if self._num_workers > 0 else None,
            collate_fn=_collate_tree,
            drop_last=False,
        )
        for epoch in range(self._keyframe_epochs):
            data_iter = iter(loader)
            for step_i in range(1, total_steps + 1):
                batch = next(data_iter)
                batch_obs = self._batch_to_model_inputs(batch["obs"])
                batch_label = jnp.asarray(batch["target"], dtype=jnp.int32)
                metrics = self._train_one_keyframe_batch(batch_obs, batch_label)
                all_metrics.append(metrics)
                if self._should_log_step_timing(step_i, total_steps):
                    log.info(
                        "train_keyframe_from_lerobot epoch=%d/%d step=%d/%d",
                        epoch + 1,
                        self._keyframe_epochs,
                        step_i,
                        total_steps,
                    )
                    self._log_step_metrics(
                        "train_keyframe_from_lerobot",
                        epoch + 1,
                        self._keyframe_epochs,
                        step_i,
                        total_steps,
                        metrics,
                    )

        mean = {}
        for k in all_metrics[0]:
            mean[k] = float(np.mean([m[k] for m in all_metrics]))
        mean["num_samples"] = float(num_rows)
        saved_to = self._autosave_after_train("train_keyframe_from_lerobot")
        if saved_to:
            mean["saved_state_file"] = saved_to
        log.info("train_keyframe_from_lerobot done metrics=%s", mean)
        return mean


class ValueMCWebsocketServer:
    def __init__(self, *, host: str, port: int, service: ValueMCService):
        self._host = host
        self._port = int(port)
        self._service = service

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"server": "value_mc_ws"}))
        log.info("client connected: %s", websocket.remote_address)
        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                if not isinstance(msg, dict):
                    raise TypeError(f"Expected dict payload, got {type(msg)}")
                cmd = msg.get("cmd")
                if cmd == "predict":
                    value = self._service.predict(dict(msg["observation"]))
                    resp = {"value": value}
                elif cmd == "predict_keyframe":
                    keyframe_prob = self._service.predict_keyframe(dict(msg["observation"]))
                    resp = {"keyframe_prob": keyframe_prob}
                elif cmd == "sync_params_from_file":
                    self._service.sync_params_from_file(str(msg["params_file"]))
                    resp = {"ok": True}
                elif cmd == "save_state_to_file":
                    out = self._service.save_state_to_file(
                        str(msg["state_file"]),
                        include_optimizer_state=bool(msg.get("include_optimizer_state", True)),
                    )
                    resp = {"ok": True, "state_file": out}
                elif cmd == "load_state_from_file":
                    self._service.load_state_from_file(
                        str(msg["state_file"]),
                        load_optimizer_state=bool(msg.get("load_optimizer_state", True)),
                    )
                    resp = {"ok": True}
                elif cmd == "train_mc_from_file":
                    metrics = self._service.train_mc_from_file(str(msg["dataset_file"]))
                    resp = {"metrics": metrics}
                elif cmd == "train_mc_from_lerobot":
                    metrics = self._service.train_mc_from_lerobot(
                        dataset_root=str(msg["dataset_root"]),
                        repo_id=str(msg["repo_id"]),
                    )
                    resp = {"metrics": metrics}
                elif cmd == "train_keyframe_from_lerobot":
                    metrics = self._service.train_keyframe_from_lerobot(
                        dataset_root=str(msg["dataset_root"]),
                        repo_id=str(msg["repo_id"]),
                        annotations_json=str(msg["annotations_json"]),
                    )
                    resp = {"metrics": metrics}
                else:
                    raise ValueError(f"Unknown command: {cmd}")
                await websocket.send(packer.pack(resp))
            except websockets.ConnectionClosed:
                log.info("client disconnected: %s", websocket.remote_address)
                return
            except Exception as exc:  # noqa: BLE001
                log.exception("value ws handler error")
                await websocket.send(str(exc))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Error string included in previous frame.",
                )
                return

    async def run(self) -> None:
        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
        ):
            log.info("value ws listening on ws://%s:%d", self._host, self._port)
            await asyncio.Future()


def _parse_args() -> argparse.Namespace:
    def _parse_bool(v: str) -> bool:
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")

    p = argparse.ArgumentParser(description="Standalone value MC websocket service.")
    p.add_argument(
        "--policy.path",
        dest="policy_path",
        type=str,
        default="",
        help="Optional for Pi0/Pi05 base configs; required for Pi0FAST checkpoints.",
    )
    p.add_argument("--policy.config", dest="policy_config", type=str, default="pi0_fast_aloha_robotwin_ppo")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8877)
    p.add_argument("--mc_epochs", type=int, default=1)
    p.add_argument("--mc_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers for value/keyframe training.")
    p.add_argument(
        "--persistent_workers",
        type=_parse_bool,
        default=True,
        help="Keep DataLoader workers alive across epochs (effective only when num_workers > 0).",
    )
    p.add_argument("--mc_gamma", type=float, default=0.99)
    p.add_argument("--value_target_mode", type=str, default="evorl_normalized", choices=("evorl_normalized", "legacy"))
    p.add_argument("--value_c_fail_coef", type=float, default=1.0)
    p.add_argument(
        "--value_length_scale_quantile",
        type=float,
        default=1.0,
        help="Deprecated in evorl_normalized mode; kept only for CLI compatibility.",
    )
    p.add_argument("--keyframe_epochs", type=int, default=3)
    p.add_argument("--keyframe_batch_size", type=int, default=64)
    p.add_argument("--keyframe_chunk_size", type=int, default=32)
    p.add_argument(
        "--state_ckpt_dir",
        type=str,
        default="/tmp/openpi_value_mc_ws_ckpt",
        help="Auto-save directory after each train request; empty string disables auto-save.",
    )
    p.add_argument(
        "--init_state_file",
        type=str,
        default=None,
        help="Optional checkpoint file to load at service startup.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    service = ValueMCService(
        policy_config=args.policy_config,
        policy_path=args.policy_path,
        mc_epochs=args.mc_epochs,
        mc_batch_size=args.mc_batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        mc_gamma=args.mc_gamma,
        keyframe_epochs=args.keyframe_epochs,
        keyframe_batch_size=args.keyframe_batch_size,
        keyframe_chunk_size=args.keyframe_chunk_size,
        value_target_mode=args.value_target_mode,
        value_c_fail_coef=args.value_c_fail_coef,
        value_length_scale_quantile=args.value_length_scale_quantile,
        state_ckpt_dir=(args.state_ckpt_dir.strip() if args.state_ckpt_dir else None),
        init_state_file=args.init_state_file,
    )
    server = ValueMCWebsocketServer(host=args.host, port=args.port, service=service)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
