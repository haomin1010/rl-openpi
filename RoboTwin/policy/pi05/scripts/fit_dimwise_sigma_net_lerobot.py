#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.tokenizer import FASTTokenizer
from openpi_online_ppo.data.local_lerobot_loader import ensure_local_hf_cache


def _build_ortho_dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float32)[:, None]
    t = np.arange(n, dtype=np.float32)[None, :]
    mat = np.sqrt(2.0 / n) * np.cos(np.pi * (t + 0.5) * k / n)
    mat[0, :] = np.sqrt(1.0 / n)
    return mat.astype(np.float32)


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _extract_vec(item: dict[str, Any], key: str) -> np.ndarray:
    if key not in item:
        raise KeyError(f"Missing key `{key}` in sample.")
    return _to_np(item[key]).astype(np.float32).reshape(-1)


def _parse_dim_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"all", "none"}:
        return None
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return tuple(out) if out else None


def _default_action_noise_dims(action_dim: int) -> tuple[int, ...] | None:
    if int(action_dim) == 14:
        return tuple(range(14))
    return None


def _parse_annotation_start_frames(annotations: dict[str, Any]) -> dict[int, list[int]]:
    episodes_raw = annotations.get("episodes", {})
    if not isinstance(episodes_raw, dict):
        raise ValueError("annotations_json must contain an `episodes` object.")

    parsed: dict[int, list[int]] = {}
    for ep_key, spans_raw in episodes_raw.items():
        ep = int(ep_key)
        if spans_raw is None:
            parsed[ep] = []
            continue
        if not isinstance(spans_raw, list):
            raise ValueError(f"Episode {ep} annotations must be a list.")

        start_frames: set[int] = set()
        for item in spans_raw:
            if isinstance(item, dict):
                if "start" not in item or "end" not in item:
                    raise ValueError(f"Episode {ep} phase item missing start/end: {item!r}")
                start = int(item["start"])
                end = int(item["end"])
                if end < start:
                    raise ValueError(f"Episode {ep} phase range invalid: [{start}, {end}]")
                # Align with the old keyframe behavior: a labeled frame t supervises
                # the action chunk that starts at frame t+1.
                for frame_idx in range(start, end + 1):
                    start_frames.add(int(frame_idx) + 1)
            else:
                # Backward-compatible old format: list[int] of keyframes.
                start_frames.add(int(item) + 1)
        parsed[ep] = sorted(start_frames)
    return parsed


def _init_mlp_params(
    *,
    rng: jax.Array,
    in_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    out_dim: int,
) -> list[dict[str, jnp.ndarray]]:
    dims = [int(in_dim)] + [int(hidden_dim)] * int(hidden_depth) + [int(out_dim)]
    params: list[dict[str, jnp.ndarray]] = []
    key = rng
    for i in range(len(dims) - 1):
        key, sub = jax.random.split(key)
        fan_in = float(dims[i])
        w = jax.random.normal(sub, (dims[i], dims[i + 1]), dtype=jnp.float32) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros((dims[i + 1],), dtype=jnp.float32)
        params.append({"w": w, "b": b})
    return params


def _mlp_forward(params: list[dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for i, layer in enumerate(params):
        h = h @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            h = jax.nn.gelu(h)
    return h


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-dimension diagonal Gaussian sigma net in DCT space.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--action_key", type=str, default="action")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--action_noise_dims", type=str, default="auto")
    p.add_argument(
        "--noise_dct_keep_k",
        type=int,
        default=0,
        help="If > 0, only the first K DCT coefficients per action dimension are allowed to receive noise.",
    )
    p.add_argument("--fast_tokenizer_path", type=str, default="physical-intelligence/fast")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--hidden_depth", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_chunks", type=int, default=50000)
    p.add_argument("--sigma_min", type=float, default=1e-4)
    p.add_argument("--sigma_max", type=float, default=10.0)
    p.add_argument("--action_delta_limit", type=float, default=0.03)
    p.add_argument("--constraint_margin", type=float, default=0.95)
    p.add_argument("--constraint_coef", type=float, default=50.0)
    p.add_argument(
        "--reward_scale_ratio",
        type=float,
        default=1.0,
        help="Reward saturation scale as a ratio of action_delta_limit * constraint_margin.",
    )
    p.add_argument(
        "--penalty_power",
        type=float,
        default=4.0,
        help="Power used in the normalized soft constraint penalty. Must be >= 2.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(int(args.seed))

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ensure_local_hf_cache()
    ann = json.loads(pathlib.Path(args.annotations_json).read_text(encoding="utf-8"))
    ann_eps = _parse_annotation_start_frames(ann)

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")

    ep_frames: dict[int, list[tuple[int, np.ndarray]]] = {}
    for i in range(len(ds)):
        item = ds[i]
        ep = int(_to_np(item["episode_index"]).item())
        fi = int(_to_np(item["frame_index"]).item())
        act = _extract_vec(item, args.action_key)
        ep_frames.setdefault(ep, []).append((fi, act))
    for ep in ep_frames:
        ep_frames[ep].sort(key=lambda x: x[0])

    action_dim = int(ep_frames[next(iter(ep_frames))][0][1].shape[0])
    raw_noise_dims = str(args.action_noise_dims).strip().lower()
    if raw_noise_dims == "auto":
        action_noise_dims = _default_action_noise_dims(action_dim)
    else:
        action_noise_dims = _parse_dim_list(args.action_noise_dims)
    allowed_dims = list(range(action_dim)) if action_noise_dims is None else [int(x) for x in action_noise_dims if 0 <= int(x) < action_dim]

    chunk_size = int(args.chunk_size)
    chunks: list[np.ndarray] = []
    for ep, keyframes in ann_eps.items():
        frames = ep_frames.get(ep, [])
        if not frames:
            continue
        frame_to_pos = {fi: idx for idx, (fi, _) in enumerate(frames)}
        actions_ep = np.stack([a for _, a in frames], axis=0)
        for kf in keyframes:
            start_fi = kf + 1
            if start_fi not in frame_to_pos:
                continue
            s = frame_to_pos[start_fi]
            e = s + chunk_size
            if e > actions_ep.shape[0]:
                continue
            fis = [frames[j][0] for j in range(s, e)]
            if any(fis[j + 1] != fis[j] + 1 for j in range(len(fis) - 1)):
                continue
            chunks.append(actions_ep[s:e].astype(np.float32))

    if not chunks:
        raise RuntimeError("No valid key chunks found from annotations.")
    np.random.shuffle(chunks)
    if int(args.max_chunks) > 0:
        chunks = chunks[: int(args.max_chunks)]
    chunk_arr = np.stack(chunks, axis=0).astype(np.float32)  # [N,T,A]

    fast_tokenizer = FASTTokenizer(max_len=256, fast_tokenizer_path=args.fast_tokenizer_path)
    dct_scale = float(getattr(fast_tokenizer._fast_tokenizer, "scale", 1.0))
    dct_mat = _build_ortho_dct_matrix(chunk_size)
    dct_chunks = (np.einsum("tk,nta->nka", dct_mat, chunk_arr) * dct_scale).astype(np.float32)

    dimwise_inputs: list[np.ndarray] = []
    dimwise_dims: list[int] = []
    for chunk_idx in range(chunk_arr.shape[0]):
        for dim_idx in allowed_dims:
            onehot = np.zeros((action_dim,), dtype=np.float32)
            onehot[int(dim_idx)] = 1.0
            x = np.concatenate([dct_chunks[chunk_idx, :, int(dim_idx)], onehot], axis=0)
            dimwise_inputs.append(x.astype(np.float32))
            dimwise_dims.append(int(dim_idx))
    x_arr = np.stack(dimwise_inputs, axis=0).astype(np.float32)
    dim_arr = np.asarray(dimwise_dims, dtype=np.int32)
    chunk_idx_arr = np.repeat(np.arange(chunk_arr.shape[0], dtype=np.int32), len(allowed_dims))

    in_dim = int(x_arr.shape[1])
    out_dim = int(chunk_size)
    x_mean = np.mean(x_arr, axis=0, keepdims=True).astype(np.float32)
    x_std = np.std(x_arr, axis=0, keepdims=True).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

    params = _init_mlp_params(
        rng=jax.random.key(int(args.seed)),
        in_dim=in_dim,
        hidden_dim=int(args.hidden_dim),
        hidden_depth=int(args.hidden_depth),
        out_dim=out_dim,
    )
    tx = optax.adamw(float(args.lr), weight_decay=float(args.weight_decay))
    opt_state = tx.init(params)

    x_mean_j = jnp.asarray(x_mean, dtype=jnp.float32)
    x_std_j = jnp.asarray(x_std, dtype=jnp.float32)
    dct_scale_j = jnp.asarray(float(dct_scale), dtype=jnp.float32)
    dct_mat_inv = jnp.asarray(dct_mat.T, dtype=jnp.float32)
    sigma_min = float(args.sigma_min)
    sigma_max = float(args.sigma_max)
    delta_limit = float(args.action_delta_limit)
    margin = float(args.constraint_margin)
    constraint_coef = float(args.constraint_coef)
    penalty_power = max(2.0, float(args.penalty_power))
    threshold = delta_limit * margin
    if threshold <= 0:
        raise ValueError("--action_delta_limit * --constraint_margin must be > 0")
    reward_scale = max(1e-6, threshold * float(args.reward_scale_ratio))
    noise_dct_keep_k = int(args.noise_dct_keep_k)
    if noise_dct_keep_k < 0:
        raise ValueError("--noise_dct_keep_k must be >= 0")
    effective_keep_k = chunk_size if noise_dct_keep_k <= 0 else min(chunk_size, noise_dct_keep_k)
    noise_mask_j = jnp.zeros((chunk_size,), dtype=jnp.float32).at[:effective_keep_k].set(1.0)

    @jax.jit
    def train_step(
        params_in: list[dict[str, jnp.ndarray]],
        opt_state_in: optax.OptState,
        batch_x: jnp.ndarray,
        eps_noise: jnp.ndarray,
    ):
        def loss_fn(p):
            x_norm = (batch_x - x_mean_j) / (x_std_j + 1e-6)
            log_sigma = _mlp_forward(p, x_norm)
            sigma = jax.nn.softplus(log_sigma) + sigma_min
            if sigma_max > 0:
                sigma = jnp.clip(sigma, sigma_min, sigma_max)
            sigma = sigma * noise_mask_j
            delta_dct = sigma * eps_noise
            delta_action = (dct_mat_inv @ (delta_dct / dct_scale_j)[..., None])[..., 0]
            per_step_abs = jnp.abs(delta_action)
            violation = jax.nn.relu(per_step_abs - threshold)
            penalty = jnp.mean((violation / threshold) ** penalty_power)
            reward = jnp.mean(1.0 - jnp.exp(-per_step_abs / reward_scale))
            loss = -reward + constraint_coef * penalty
            return loss, {
                "loss": loss,
                "reward": reward,
                "penalty": penalty,
                "sigma_mean": jnp.mean(sigma),
                "delta_abs_mean": jnp.mean(per_step_abs),
                "delta_abs_max": jnp.max(per_step_abs),
                "delta_abs_p95": jnp.quantile(per_step_abs.reshape(-1), 0.95),
                "delta_abs_p99": jnp.quantile(per_step_abs.reshape(-1), 0.99),
                "exceed_rate": jnp.mean(per_step_abs > threshold),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_in)
        updates, opt_state_out = tx.update(grads, opt_state_in, params_in)
        params_out = optax.apply_updates(params_in, updates)
        return params_out, opt_state_out, metrics

    idx_all = np.arange(x_arr.shape[0], dtype=np.int64)
    rng = np.random.default_rng(int(args.seed))
    for epoch in range(int(args.epochs)):
        rng.shuffle(idx_all)
        metric_acc: dict[str, list[float]] = {}
        for i in range(0, idx_all.shape[0], int(args.batch_size)):
            bi = idx_all[i : i + int(args.batch_size)]
            batch_x = x_arr[bi]
            eps_noise = rng.normal(0.0, 1.0, size=(len(bi), chunk_size)).astype(np.float32)
            params, opt_state, metrics = train_step(
                params,
                opt_state,
                jnp.asarray(batch_x, dtype=jnp.float32),
                jnp.asarray(eps_noise, dtype=jnp.float32),
            )
            for key, value in metrics.items():
                metric_acc.setdefault(str(key), []).append(float(np.asarray(value)))
        mean_metrics = {k: float(np.mean(v)) for k, v in metric_acc.items()}
        print(f"[epoch {epoch + 1}/{args.epochs}] {json.dumps(mean_metrics, ensure_ascii=True, sort_keys=True)}")

    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    layers_payload: list[dict[str, np.ndarray]] = []
    for layer in params:
        layers_payload.append(
            {
                "w": np.asarray(layer["w"], dtype=np.float32),
                "b": np.asarray(layer["b"], dtype=np.float32),
            }
        )
    payload = {
        "model_type": "dimwise_diag_gaussian_dct_v1",
        "meta": {
            "in_dim": int(in_dim),
            "hidden_dim": int(args.hidden_dim),
            "hidden_depth": int(args.hidden_depth),
            "out_dim": int(out_dim),
            "chunk_size": int(chunk_size),
            "action_dim": int(action_dim),
            "dct_scale": float(dct_scale),
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "action_delta_limit": float(args.action_delta_limit),
            "action_noise_dims": [int(x) for x in allowed_dims],
            "noise_dct_keep_k": int(effective_keep_k),
            "target_construction": "dimwise_diag_gaussian_constraint",
        },
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "layers": layers_payload,
        "chunk_index": chunk_idx_arr.astype(np.int32),
        "dim_index": dim_arr.astype(np.int32),
    }
    with out.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dimwise sigma perturb net checkpoint: {out}")


if __name__ == "__main__":
    main()
