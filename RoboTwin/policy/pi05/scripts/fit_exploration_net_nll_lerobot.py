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


def _init_mlp_params(
    *,
    rng: jax.Array,
    in_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    out_dim: int,
) -> list[dict[str, jnp.ndarray]]:
    dims = [int(in_dim)] + [int(hidden_dim)] * int(hidden_depth) + [int(out_dim) + 1]
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
    p = argparse.ArgumentParser(description="Train JAX direction+radius perturb net with fixed-variance NLL.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--action_key", type=str, default="action")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--dct_k", type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--hidden_depth", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_chunks", type=int, default=50000)
    p.add_argument("--ar_rho", type=float, default=0.92)
    p.add_argument("--ar_sigma", type=float, default=0.03)
    p.add_argument("--ar_clip", type=float, default=0.2)
    p.add_argument("--radius_min", type=float, default=0.0)
    p.add_argument("--radius_max", type=float, default=0.2)
    p.add_argument("--nll_sigma", type=float, default=0.05)
    p.add_argument("--smooth_coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(int(args.seed))

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ensure_local_hf_cache()
    ann = json.loads(pathlib.Path(args.annotations_json).read_text(encoding="utf-8"))
    ann_eps: dict[int, list[int]] = {int(k): sorted(set(int(x) for x in v)) for k, v in ann.get("episodes", {}).items()}

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
    chunk_size = int(args.chunk_size)
    dct_k = int(args.dct_k)
    if not (1 <= dct_k <= chunk_size):
        raise ValueError(f"dct_k must be in [1, chunk_size], got {dct_k} vs {chunk_size}")

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
    n_samples = chunk_arr.shape[0]

    dct_mat = _build_ortho_dct_matrix(chunk_size)
    dct_chunks = np.einsum("tk,nta->nka", dct_mat, chunk_arr)  # [N,T,A]
    dct_firstk = dct_chunks[:, :dct_k, :].reshape(n_samples, -1).astype(np.float32)

    latent_dim = int(args.latent_dim)
    in_dim = dct_firstk.shape[1] + 1 + latent_dim
    out_dim = dct_firstk.shape[1]

    x_mean = np.zeros((1, in_dim), dtype=np.float32)
    x_std = np.ones((1, in_dim), dtype=np.float32)

    params = _init_mlp_params(
        rng=jax.random.key(int(args.seed)),
        in_dim=in_dim,
        hidden_dim=int(args.hidden_dim),
        hidden_depth=int(args.hidden_depth),
        out_dim=out_dim,
    )
    tx = optax.adamw(float(args.lr), weight_decay=float(args.weight_decay))
    opt_state = tx.init(params)
    dct_mat_t = jnp.asarray(dct_mat, dtype=jnp.float32)

    nll_sigma = float(args.nll_sigma)
    if nll_sigma <= 0:
        raise ValueError("--nll_sigma must be > 0")
    ar_rho = float(args.ar_rho)
    ar_sigma = float(args.ar_sigma)
    ar_clip = float(args.ar_clip)
    smooth_coef = float(args.smooth_coef)
    radius_min = float(args.radius_min)
    radius_max = float(args.radius_max)

    def _build_ar1_noise(batch_size: int, seed: int) -> np.ndarray:
        eps = np.random.default_rng(seed).normal(0.0, ar_sigma, size=(batch_size, chunk_size, action_dim)).astype(np.float32)
        noise = np.zeros_like(eps, dtype=np.float32)
        noise[:, 0, :] = eps[:, 0, :]
        alpha = float(np.sqrt(max(1e-8, 1.0 - ar_rho * ar_rho)))
        for t in range(1, chunk_size):
            noise[:, t, :] = ar_rho * noise[:, t - 1, :] + alpha * eps[:, t, :]
        if ar_clip > 0:
            noise = np.clip(noise, -ar_clip, ar_clip).astype(np.float32)
        return noise

    @jax.jit
    def train_step(
        params_in: list[dict[str, jnp.ndarray]],
        opt_state_in: optax.OptState,
        key_chunk: jnp.ndarray,
        key_dct_firstk: jnp.ndarray,
        target_chunk: jnp.ndarray,
        z: jnp.ndarray,
    ):
        def loss_fn(p):
            key_prob = jnp.ones((key_chunk.shape[0], 1), dtype=jnp.float32)
            x = jnp.concatenate([key_dct_firstk, key_prob, z], axis=-1)
            x = (x - jnp.asarray(x_mean, dtype=jnp.float32)) / (jnp.asarray(x_std, dtype=jnp.float32) + 1e-6)
            out = _mlp_forward(p, x)
            dir_raw = out[:, :out_dim]
            radius_logit = out[:, out_dim : out_dim + 1]
            direction = dir_raw / (jnp.linalg.norm(dir_raw, axis=-1, keepdims=True) + 1e-6)
            radius = radius_min + (radius_max - radius_min) * jax.nn.sigmoid(radius_logit)
            delta_flat = direction * radius

            delta_dct = jnp.zeros((key_chunk.shape[0], chunk_size, action_dim), dtype=jnp.float32)
            delta_dct = delta_dct.at[:, :dct_k, :].set(delta_flat.reshape(key_chunk.shape[0], dct_k, action_dim))
            delta_action = jnp.einsum("kt,bka->bta", dct_mat_t.T, delta_dct)
            pred_chunk = key_chunk + delta_action

            nll = jnp.mean((pred_chunk - target_chunk) ** 2) / (2.0 * nll_sigma * nll_sigma)
            if smooth_coef > 0:
                acc = pred_chunk[:, 2:, :] - 2.0 * pred_chunk[:, 1:-1, :] + pred_chunk[:, :-2, :]
                smooth = jnp.mean(acc * acc)
            else:
                smooth = jnp.asarray(0.0, dtype=jnp.float32)
            return nll + smooth_coef * smooth

        loss, grads = jax.value_and_grad(loss_fn)(params_in)
        updates, opt_state_out = tx.update(grads, opt_state_in, params_in)
        params_out = optax.apply_updates(params_in, updates)
        return params_out, opt_state_out, loss

    idx_all = np.arange(n_samples, dtype=np.int64)
    seed_seq = np.random.default_rng(int(args.seed))
    for epoch in range(int(args.epochs)):
        np.random.shuffle(idx_all)
        losses: list[float] = []
        for i in range(0, n_samples, int(args.batch_size)):
            bi = idx_all[i : i + int(args.batch_size)]
            bsz = len(bi)
            key_chunk_np = chunk_arr[bi]
            key_dct_np = dct_firstk[bi]
            noise_np = _build_ar1_noise(bsz, int(seed_seq.integers(0, 2**31 - 1)))
            target_np = key_chunk_np + noise_np
            z_np = seed_seq.normal(0.0, 1.0, size=(bsz, latent_dim)).astype(np.float32)
            params, opt_state, loss = train_step(
                params,
                opt_state,
                jnp.asarray(key_chunk_np, dtype=jnp.float32),
                jnp.asarray(key_dct_np, dtype=jnp.float32),
                jnp.asarray(target_np, dtype=jnp.float32),
                jnp.asarray(z_np, dtype=jnp.float32),
            )
            losses.append(float(np.asarray(loss)))
        print(f"[epoch {epoch + 1}/{args.epochs}] loss={float(np.mean(losses)):.6f}")

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
        "model_type": "direction_radius_mlp_jax_v1",
        "meta": {
            "in_dim": int(in_dim),
            "hidden_dim": int(args.hidden_dim),
            "hidden_depth": int(args.hidden_depth),
            "out_dim": int(out_dim),
            "chunk_size": int(chunk_size),
            "action_dim": int(action_dim),
            "dct_k": int(dct_k),
            "radius_min": float(radius_min),
            "radius_max": float(radius_max),
            "latent_dim": int(latent_dim),
        },
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "layers": layers_payload,
    }
    with out.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved JAX perturb net checkpoint: {out}")


if __name__ == "__main__":
    main()
