#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import math
import pathlib
import pickle
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames

from openpi.models import model as _model
from openpi.models import pi0_fast as _base_pi0_fast
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as train_config
from openpi_client import msgpack_numpy
from openpi_online_ppo.models import pi0_fast_rl as _rl_pi0_fast
from openpi_online_ppo.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy

log = logging.getLogger("value_mc_ws")


def _to_rl_train_config(cfg: train_config.TrainConfig) -> train_config.TrainConfig:
    if not isinstance(cfg.model, _base_pi0_fast.Pi0FASTConfig):
        raise TypeError(f"Config `{cfg.name}` is not a Pi0FAST config.")
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
        value_target_mode: str,
        value_c_fail_coef: float,
        value_length_scale_quantile: float,
        state_ckpt_dir: str | None,
        init_state_file: str | None,
    ) -> None:
        cfg = _to_rl_train_config(train_config.get_config(policy_config))
        policy = create_trained_pi0_fast_rl_policy(cfg, policy_path)
        self._policy_wrapper = policy
        self._model_def = nnx.graphdef(policy.model)
        self._params = nnx.state(policy.model)
        self._train_config = cfg
        self._value_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*value_head.*"))
        self._keyframe_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*keyframe_head.*"))
        self._tx = cfg.optimizer.create(cfg.lr_schedule.create(), weight_decay_mask=None)
        self._value_opt_state = self._tx.init(self._params.filter(self._value_filter))
        self._keyframe_opt_state = self._tx.init(self._params.filter(self._keyframe_filter))
        self._mc_epochs = max(1, int(mc_epochs))
        self._mc_batch_size = max(1, int(mc_batch_size))
        self._mc_gamma = float(mc_gamma)
        self._keyframe_epochs = max(1, int(keyframe_epochs))
        self._keyframe_batch_size = max(1, int(keyframe_batch_size))
        self._keyframe_chunk_size = max(2, int(keyframe_chunk_size))
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
    def _compute_task_length_scales(task_lengths: dict[int, list[int]], quantile: float) -> dict[int, float]:
        if not 0.0 < float(quantile) <= 1.0:
            raise ValueError("'value_length_scale_quantile' must be within (0, 1].")
        task_scales: dict[int, float] = {}
        for task_index, lengths in task_lengths.items():
            if not lengths:
                raise ValueError(f"No episode lengths collected for task_index={task_index}.")
            lengths_np = np.asarray(lengths, dtype=np.float32)
            if np.any(lengths_np <= 0):
                raise ValueError(f"Invalid non-positive episode lengths for task_index={task_index}.")
            task_scale = float(np.quantile(lengths_np, quantile))
            if task_scale <= 0:
                raise ValueError(f"Computed non-positive task scale {task_scale} for task_index={task_index}.")
            task_scales[task_index] = task_scale
        return task_scales

    def _merge_model(self):
        return nnx.merge(self._model_def, self._params)

    def predict(self, transformed_obs: dict[str, Any]) -> float:
        observation = _model.Observation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed_obs)
        )
        model = self._merge_model()
        value = model.predict_value(observation)[0]
        return float(np.asarray(value))

    def predict_keyframe(self, transformed_obs: dict[str, Any]) -> float:
        observation = _model.Observation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed_obs)
        )
        model = self._merge_model()
        prob = model.predict_keyframe_prob(observation, stop_gradient=True)[0]
        return float(np.asarray(prob))

    def sync_params_from_file(self, params_file: str) -> None:
        path = pathlib.Path(params_file)
        with path.open("rb") as f:
            pure = pickle.load(f)
        model = self._merge_model()
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(pure)
        model = nnx.merge(graphdef, state)
        self._params = nnx.state(model)

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
            "params_pure": self._params.to_pure_dict(),
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

        model = self._merge_model()
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(pure)
        model = nnx.merge(graphdef, state)
        self._params = nnx.state(model)

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

    def _train_one_batch(self, obs_batch: dict[str, jnp.ndarray], targets: jnp.ndarray) -> dict[str, float]:
        model = self._merge_model()

        def loss_fn(m):
            observation = _model.Observation.from_dict(obs_batch)
            logits = m.predict_value_logits(observation, stop_gradient=True)
            soft_target = m.project_values_to_bins(targets)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            per_sample_loss = -jnp.sum(soft_target * log_probs, axis=-1)
            value_loss = jnp.mean(per_sample_loss)
            pred_value = m.predict_value(observation, stop_gradient=True)
            value_mae = jnp.mean(jnp.abs(pred_value - targets))
            return value_loss, {"value_loss": value_loss, "value_mae": value_mae}

        diff_state = nnx.DiffState(0, self._value_filter)
        (_, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
        params = self._params.filter(self._value_filter)
        updates, self._value_opt_state = self._tx.update(grads, self._value_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        self._params = nnx.state(model)
        metrics = {
            **metrics,
            "value_grad_norm": optax.global_norm(grads),
        }
        return self._as_float_metrics(metrics)

    def _train_one_keyframe_batch(self, obs_batch: dict[str, jnp.ndarray], labels: jnp.ndarray) -> dict[str, float]:
        model = self._merge_model()

        def loss_fn(m):
            observation = _model.Observation.from_dict(obs_batch)
            logits = m.predict_keyframe_logits(observation, stop_gradient=True)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            per_sample_loss = -jnp.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0]
            keyframe_loss = jnp.mean(per_sample_loss)
            pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            acc = jnp.mean((pred == labels).astype(jnp.float32))
            return keyframe_loss, {"keyframe_loss": keyframe_loss, "keyframe_acc": acc}

        diff_state = nnx.DiffState(0, self._keyframe_filter)
        (_, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
        params = self._params.filter(self._keyframe_filter)
        updates, self._keyframe_opt_state = self._tx.update(grads, self._keyframe_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        self._params = nnx.state(model)
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
                all_metrics.append(self._train_one_batch(batch_obs, batch_ret))
                if step_i == 1 or step_i % 20 == 0 or step_i == total_steps:
                    log.info("train_mc_from_file epoch=%d/%d step=%d/%d", epoch + 1, self._mc_epochs, step_i, total_steps)

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
    def _as_numpy(x: Any) -> np.ndarray:
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x)

    @staticmethod
    def _unflatten_obs_t_row(row: dict[str, Any]) -> dict[str, Any]:
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
            arr = ValueMCService._as_numpy(v)
            node[parts[-1]] = arr
        return obs

    def _row_to_model_inputs(self, row: dict[str, Any]) -> dict[str, Any]:
        if any(k.startswith("obs_t.") for k in row.keys()):
            return self._unflatten_obs_t_row(row)

        required = (
            "observation.state",
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
        if all(k in row for k in required):
            prompt = row.get("task", "")
            if not isinstance(prompt, str):
                prompt = str(prompt)
            env_obs = {
                "state": self._as_numpy(row["observation.state"]).astype(np.float32),
                "images": {
                    "cam_high": self._as_numpy(row["observation.images.cam_high"]),
                    "cam_left_wrist": self._as_numpy(row["observation.images.cam_left_wrist"]),
                    "cam_right_wrist": self._as_numpy(row["observation.images.cam_right_wrist"]),
                },
                "prompt": prompt,
            }
            return self._policy_wrapper.transform_observation(env_obs)

        raise RuntimeError(f"Unsupported row format; cannot build model inputs from keys: {sorted(row.keys())}")

    def train_mc_from_lerobot(self, *, dataset_root: str, repo_id: str) -> dict[str, float]:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        log.info("train_mc_from_lerobot start dataset_root=%s repo_id=%s", dataset_root, repo_id)
        ds = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        if len(ds) == 0:
            log.info("train_mc_from_lerobot empty dataset.")
            return {"value_loss": 0.0, "value_mae": 0.0, "value_grad_norm": 0.0, "num_samples": 0.0}

        rows: list[dict[str, Any]] = []
        has_mc_return = False
        for i in range(len(ds)):
            item = ds[i]
            row = {k: item[k] for k in item.keys()}
            if "mc_return" in row:
                has_mc_return = True
            rows.append(row)

        obs_list: list[dict[str, Any]] = []
        ret_list: list[float] = []
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

            ep_rows: dict[int, list[tuple[int, dict[str, Any]]]] = {}
            task_name_to_index: dict[str, int] = {}
            ep_task_index: dict[int, int] = {}
            task_lengths: dict[int, list[int]] = {}
            for row in rows:
                ep = int(np.asarray(row["episode_index"]).item())
                fi = int(np.asarray(row["frame_index"]).item())
                ep_rows.setdefault(ep, []).append((fi, row))
            for ep in sorted(ep_rows.keys()):
                rows_ep = sorted(ep_rows[ep], key=lambda x: x[0])
                ep_len = len(rows_ep)
                if rows_ep and "task_index" in rows_ep[0][1]:
                    task_index = int(np.asarray(rows_ep[0][1]["task_index"]).item())
                else:
                    task_name = ""
                    if rows_ep and "task" in rows_ep[0][1]:
                        task_name = str(rows_ep[0][1]["task"])
                    if task_name not in task_name_to_index:
                        task_name_to_index[task_name] = len(task_name_to_index)
                    task_index = task_name_to_index[task_name]
                ep_task_index[ep] = task_index
                task_lengths.setdefault(task_index, []).append(ep_len)

            task_scales = self._compute_task_length_scales(task_lengths, self._value_length_scale_quantile)
            c_fail_coef = float(self._value_c_fail_coef)
            if c_fail_coef < 0:
                raise ValueError("'value_c_fail_coef' must be non-negative.")
            clip_min = float(getattr(self._train_config.model, "value_bin_min", -1.0))
            clip_max = float(getattr(self._train_config.model, "value_bin_max", 0.0))

            log.info(
                "train_mc_from_lerobot EvoRL-target mode: episodes=%d tasks=%d c_fail_coef=%.4f q=%.3f clip=[%.3f, %.3f]",
                len(ep_rows),
                len(task_scales),
                c_fail_coef,
                self._value_length_scale_quantile,
                clip_min,
                clip_max,
            )
            for ep in sorted(ep_rows.keys()):
                rows_ep = sorted(ep_rows[ep], key=lambda x: x[0])
                success = bool(outcome_by_ep.get(ep, False))
                task_index = ep_task_index[ep]
                task_scale = float(task_scales[task_index])
                c_fail = float(task_scale)
                ep_len = len(rows_ep)
                for fi, row in rows_ep:
                    remaining_steps = ep_len - int(fi) - 1
                    remaining_steps = min(float(remaining_steps), float(task_scale))
                    g = -float(remaining_steps)
                    if not success:
                        g -= (c_fail_coef**remaining_steps) * c_fail
                    denom = float(task_scale) + c_fail
                    target = float(np.clip(g / denom, clip_min, clip_max))
                    obs_list.append(self._row_to_model_inputs(row))
                    ret_list.append(target)
        elif has_mc_return:
            log.info("train_mc_from_lerobot using mc_return field.")
            for row in rows:
                obs_list.append(self._row_to_model_inputs(row))
                ret = row["mc_return"]
                if hasattr(ret, "numpy"):
                    ret = ret.numpy()
                ret = np.asarray(ret, dtype=np.float32).reshape(-1)
                ret_list.append(float(ret[0]))
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

            ep_rows: dict[int, list[tuple[int, dict[str, Any]]]] = {}
            for row in rows:
                ep = int(np.asarray(row["episode_index"]).item())
                fi = int(np.asarray(row["frame_index"]).item())
                ep_rows.setdefault(ep, []).append((fi, row))

            gamma = float(self._mc_gamma)
            for ep in sorted(ep_rows.keys()):
                rows_ep = sorted(ep_rows[ep], key=lambda x: x[0])
                success = outcome_by_ep.get(ep, False)
                final_reward = 0.0 if success else -100.0
                returns = [0.0] * len(rows_ep)
                if rows_ep:
                    returns[-1] = final_reward
                    for idx in range(len(rows_ep) - 2, -1, -1):
                        returns[idx] = -1.0 + gamma * returns[idx + 1]
                for (_, row), ret in zip(rows_ep, returns):
                    obs_list.append(self._row_to_model_inputs(row))
                    ret_list.append(float(ret))

        ret_arr = np.asarray(ret_list, dtype=np.float32)
        idx = np.arange(ret_arr.shape[0], dtype=np.int32)
        all_metrics: list[dict[str, float]] = []
        rng = np.random.default_rng()
        total_steps = int(np.ceil(idx.shape[0] / self._mc_batch_size))
        log.info(
            "train_mc_from_lerobot prepared_samples=%d epochs=%d batch_size=%d steps_per_epoch=%d",
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
                all_metrics.append(self._train_one_batch(batch_obs, batch_ret))
                if step_i == 1 or step_i % 20 == 0 or step_i == total_steps:
                    log.info("train_mc_from_lerobot epoch=%d/%d step=%d/%d", epoch + 1, self._mc_epochs, step_i, total_steps)

        mean = {}
        for k in all_metrics[0]:
            mean[k] = float(np.mean([m[k] for m in all_metrics]))
        mean["num_samples"] = float(len(obs_list))
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

        ds = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        if len(ds) == 0:
            log.info("train_keyframe_from_lerobot empty dataset.")
            return {"keyframe_loss": 0.0, "keyframe_acc": 0.0, "keyframe_grad_norm": 0.0, "num_samples": 0.0}

        obs_list: list[dict[str, Any]] = []
        labels_list: list[int] = []
        half_chunk = max(1, int(self._keyframe_chunk_size // 2))
        num_bins = int(getattr(self._train_config.model, "keyframe_num_bins", 0) or max(2, half_chunk))

        def _nearest_distance_bin(frame_idx: int, keyframes: list[int]) -> int:
            if not keyframes:
                return num_bins - 1
            nearest = min(abs(int(frame_idx) - int(kf)) for kf in keyframes)
            return int(min(max(0, nearest), num_bins - 1))

        for i in range(len(ds)):
            item = ds[i]
            row = {k: item[k] for k in item.keys()}
            ep = int(np.asarray(row["episode_index"]).item())
            fi = int(np.asarray(row["frame_index"]).item())
            obs_list.append(self._row_to_model_inputs(row))
            labels_list.append(_nearest_distance_bin(fi, ann_eps.get(ep, [])))

        labels_arr = np.asarray(labels_list, dtype=np.int32)
        idx = np.arange(labels_arr.shape[0], dtype=np.int32)
        all_metrics: list[dict[str, float]] = []
        rng = np.random.default_rng()
        key_ratio = float(np.mean(labels_arr == 0)) if labels_arr.size > 0 else 0.0
        total_steps = int(np.ceil(idx.shape[0] / self._keyframe_batch_size))
        log.info(
            (
                "train_keyframe_from_lerobot prepared_samples=%d "
                "key_bin_ratio=%.4f chunk_size=%d bins=%d "
                "epochs=%d batch_size=%d steps_per_epoch=%d"
            ),
            idx.shape[0],
            key_ratio,
            self._keyframe_chunk_size,
            num_bins,
            self._keyframe_epochs,
            self._keyframe_batch_size,
            total_steps,
        )
        for epoch in range(self._keyframe_epochs):
            rng.shuffle(idx)
            for step_i, i in enumerate(range(0, idx.shape[0], self._keyframe_batch_size), start=1):
                j = idx[i : i + self._keyframe_batch_size]
                batch_obs = jax.tree.map(
                    lambda *xs: jnp.asarray(np.stack(xs, axis=0)),
                    *[obs_list[int(k)] for k in j.tolist()],
                )
                batch_label = jnp.asarray(labels_arr[j], dtype=jnp.int32)
                all_metrics.append(self._train_one_keyframe_batch(batch_obs, batch_label))
                if step_i == 1 or step_i % 20 == 0 or step_i == total_steps:
                    log.info(
                        "train_keyframe_from_lerobot epoch=%d/%d step=%d/%d",
                        epoch + 1,
                        self._keyframe_epochs,
                        step_i,
                        total_steps,
                    )

        mean = {}
        for k in all_metrics[0]:
            mean[k] = float(np.mean([m[k] for m in all_metrics]))
        mean["num_samples"] = float(len(obs_list))
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
    p = argparse.ArgumentParser(description="Standalone value MC websocket service.")
    p.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    p.add_argument("--policy.config", dest="policy_config", type=str, default="pi0_fast_aloha_robotwin_ppo")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8877)
    p.add_argument("--mc_epochs", type=int, default=1)
    p.add_argument("--mc_batch_size", type=int, default=32)
    p.add_argument("--mc_gamma", type=float, default=0.99)
    p.add_argument("--value_target_mode", type=str, default="evorl_normalized", choices=("evorl_normalized", "legacy"))
    p.add_argument("--value_c_fail_coef", type=float, default=0.995)
    p.add_argument("--value_length_scale_quantile", type=float, default=0.95)
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
