#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from openpi.models.tokenizer import FASTTokenizer
from openpi_online_ppo.data.local_lerobot_loader import ensure_local_hf_cache
from openpi_online_ppo.rl.exploration import DimwiseDiagGaussianDCTPerturbNet


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
    if s == "" or s.lower() in {"all", "none", "auto"}:
        return None
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return tuple(out) if out else None


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
                for frame_idx in range(start, end + 1):
                    start_frames.add(int(frame_idx) + 1)
            else:
                start_frames.add(int(item) + 1)
        parsed[ep] = sorted(start_frames)
    return parsed


def _safe_quantiles(x: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    if x.size == 0:
        return {f"p{int(q * 100):02d}": 0.0 for q in quantiles}
    vals = np.quantile(x.astype(np.float64), quantiles)
    return {f"p{int(q * 100):02d}": float(v) for q, v in zip(quantiles, vals)}


def _histogram_ratios(x: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        out: dict[str, float] = {}
        for i in range(len(edges) - 1):
            out[f"[{edges[i]:.6f},{edges[i + 1]:.6f})"] = 0.0
        out[f">={edges[-1]:.6f}"] = 0.0
        return out
    hist, _ = np.histogram(x.astype(np.float64), bins=edges)
    total = max(1, int(x.size))
    out = {f"[{edges[i]:.6f},{edges[i + 1]:.6f})": float(hist[i] / total) for i in range(len(hist))}
    out[f">={edges[-1]:.6f}"] = float(np.mean(x.astype(np.float64) >= float(edges[-1])))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a trained dimwise sigma DCT perturbation network on LeRobot chunks.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    p.add_argument("--net", type=str, required=True, help="Path to dimwise_sigma_dct.pkl")
    p.add_argument("--action_key", type=str, default="action")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--fast_tokenizer_path", type=str, default="physical-intelligence/fast")
    p.add_argument("--action_noise_dims", type=str, default="auto")
    p.add_argument("--max_chunks", type=int, default=0)
    p.add_argument("--num_samples_per_chunk", type=int, default=64)
    p.add_argument("--threshold", type=float, default=-1.0, help="Override action delta threshold. Default uses net.action_delta_limit.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    net = DimwiseDiagGaussianDCTPerturbNet.load(args.net)
    chunk_size = int(args.chunk_size)
    if chunk_size != net.chunk_size:
        raise ValueError(f"--chunk_size={chunk_size} does not match net.chunk_size={net.chunk_size}")

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
    if action_dim != net.action_dim:
        raise ValueError(f"Dataset action_dim={action_dim} does not match net.action_dim={net.action_dim}")

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
    if int(args.max_chunks) > 0:
        chunks = chunks[: int(args.max_chunks)]
    chunk_arr = np.stack(chunks, axis=0).astype(np.float32)

    raw_noise_dims = str(args.action_noise_dims).strip().lower()
    if raw_noise_dims == "auto":
        allowed_dims = net._allowed_dims(action_dim, None)
    else:
        allowed_dims = net._allowed_dims(action_dim, _parse_dim_list(args.action_noise_dims))

    fast_tokenizer = FASTTokenizer(max_len=256, fast_tokenizer_path=args.fast_tokenizer_path)
    dct_scale = float(getattr(fast_tokenizer._fast_tokenizer, "scale", 1.0))
    if abs(dct_scale - float(net.dct_scale)) > 1e-6:
        print(
            json.dumps(
                {
                    "warning": "tokenizer_dct_scale_mismatch",
                    "tokenizer_dct_scale": float(dct_scale),
                    "net_dct_scale": float(net.dct_scale),
                },
                ensure_ascii=True,
            )
        )
    dct_mat = _build_ortho_dct_matrix(chunk_size)
    dct_chunks = (np.einsum("tk,nta->nka", dct_mat, chunk_arr) * dct_scale).astype(np.float32)

    threshold = float(args.threshold) if float(args.threshold) > 0 else float(net.action_delta_limit)
    if threshold <= 0:
        raise ValueError("Threshold must be > 0, either via --threshold or checkpoint action_delta_limit.")

    num_samples = int(args.num_samples_per_chunk)
    if num_samples <= 0:
        raise ValueError("--num_samples_per_chunk must be > 0")

    pre_abs_all: list[np.ndarray] = []
    post_abs_all: list[np.ndarray] = []
    pre_dim_abs_all: list[np.ndarray] = []
    post_dim_abs_all: list[np.ndarray] = []
    pre_max_per_sample: list[float] = []
    post_max_per_sample: list[float] = []
    rescale_factors: list[float] = []
    rescaled_count = 0

    for chunk_idx in range(chunk_arr.shape[0]):
        coeffs = dct_chunks[chunk_idx]
        for _ in range(num_samples):
            delta_dct = np.zeros_like(coeffs, dtype=np.float32)
            for dim_idx in allowed_dims:
                dim_onehot = np.zeros((action_dim,), dtype=np.float32)
                dim_onehot[int(dim_idx)] = 1.0
                x = np.concatenate([coeffs[:, int(dim_idx)], dim_onehot], axis=0)
                sigma = net._sigma_from_logits(net._mlp(x))
                eps = rng.normal(0.0, 1.0, size=(chunk_size,)).astype(np.float32)
                delta_dct[:, int(dim_idx)] = sigma * eps

            delta_action_pre = dct_mat.T @ (delta_dct / max(float(net.dct_scale), 1e-6))
            pre_abs = np.abs(delta_action_pre).astype(np.float32)
            pre_abs_all.append(pre_abs.reshape(-1))
            pre_dim_abs_all.append(pre_abs)

            pre_max = float(np.max(pre_abs)) if pre_abs.size > 0 else 0.0
            pre_max_per_sample.append(pre_max)

            scale = 1.0
            delta_dct_post = delta_dct
            if pre_max > threshold:
                scale = float(threshold / pre_max)
                delta_dct_post = (delta_dct * scale).astype(np.float32)
                rescaled_count += 1
            rescale_factors.append(scale)

            delta_action_post = dct_mat.T @ (delta_dct_post / max(float(net.dct_scale), 1e-6))
            post_abs = np.abs(delta_action_post).astype(np.float32)
            post_abs_all.append(post_abs.reshape(-1))
            post_dim_abs_all.append(post_abs)
            post_max_per_sample.append(float(np.max(post_abs)) if post_abs.size > 0 else 0.0)

    pre_abs_flat = np.concatenate(pre_abs_all, axis=0) if pre_abs_all else np.zeros((0,), dtype=np.float32)
    post_abs_flat = np.concatenate(post_abs_all, axis=0) if post_abs_all else np.zeros((0,), dtype=np.float32)
    pre_dim_abs = np.stack(pre_dim_abs_all, axis=0) if pre_dim_abs_all else np.zeros((0, chunk_size, action_dim), dtype=np.float32)
    post_dim_abs = np.stack(post_dim_abs_all, axis=0) if post_dim_abs_all else np.zeros((0, chunk_size, action_dim), dtype=np.float32)

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    hist_edges = np.asarray(
        [
            0.0,
            threshold / 6.0,
            threshold / 3.0,
            threshold / 2.0,
            2.0 * threshold / 3.0,
            5.0 * threshold / 6.0,
            threshold,
        ],
        dtype=np.float64,
    )
    summary = {
        "net": str(pathlib.Path(args.net).resolve()),
        "dataset_root": str(pathlib.Path(args.dataset_root).resolve()),
        "repo_id": str(args.repo_id),
        "chunk_size": int(chunk_size),
        "action_dim": int(action_dim),
        "allowed_dims": [int(x) for x in allowed_dims],
        "num_chunks": int(chunk_arr.shape[0]),
        "num_samples_per_chunk": int(num_samples),
        "num_total_draws": int(len(pre_max_per_sample)),
        "threshold": float(threshold),
        "rescale_trigger_rate": float(rescaled_count / max(1, len(pre_max_per_sample))),
        "rescale_factor_mean": float(np.mean(rescale_factors)) if rescale_factors else 1.0,
        "rescale_factor_min": float(np.min(rescale_factors)) if rescale_factors else 1.0,
        "pre_clip": {
            "mean_abs": float(np.mean(pre_abs_flat)) if pre_abs_flat.size > 0 else 0.0,
            "exceed_rate": float(np.mean(pre_abs_flat > threshold)) if pre_abs_flat.size > 0 else 0.0,
            "near_limit_rate_90": float(np.mean(pre_abs_flat >= 0.9 * threshold)) if pre_abs_flat.size > 0 else 0.0,
            "near_limit_rate_95": float(np.mean(pre_abs_flat >= 0.95 * threshold)) if pre_abs_flat.size > 0 else 0.0,
            "sample_max_mean": float(np.mean(pre_max_per_sample)) if pre_max_per_sample else 0.0,
            "sample_max_exceed_rate": float(np.mean(np.asarray(pre_max_per_sample) > threshold)) if pre_max_per_sample else 0.0,
            **_safe_quantiles(pre_abs_flat, quantiles),
            **{f"sample_max_{k}": v for k, v in _safe_quantiles(np.asarray(pre_max_per_sample, dtype=np.float32), [0.5, 0.9, 0.95, 0.99]).items()},
            "histogram": _histogram_ratios(pre_abs_flat, hist_edges),
        },
        "post_clip": {
            "mean_abs": float(np.mean(post_abs_flat)) if post_abs_flat.size > 0 else 0.0,
            "exceed_rate": float(np.mean(post_abs_flat > threshold)) if post_abs_flat.size > 0 else 0.0,
            "near_limit_rate_90": float(np.mean(post_abs_flat >= 0.9 * threshold)) if post_abs_flat.size > 0 else 0.0,
            "near_limit_rate_95": float(np.mean(post_abs_flat >= 0.95 * threshold)) if post_abs_flat.size > 0 else 0.0,
            "sample_max_mean": float(np.mean(post_max_per_sample)) if post_max_per_sample else 0.0,
            "sample_max_exceed_rate": float(np.mean(np.asarray(post_max_per_sample) > threshold)) if post_max_per_sample else 0.0,
            **_safe_quantiles(post_abs_flat, quantiles),
            **{f"sample_max_{k}": v for k, v in _safe_quantiles(np.asarray(post_max_per_sample, dtype=np.float32), [0.5, 0.9, 0.95, 0.99]).items()},
            "histogram": _histogram_ratios(post_abs_flat, hist_edges),
        },
    }

    per_dim = []
    for dim_idx in range(action_dim):
        pre_dim_flat = pre_dim_abs[:, :, dim_idx].reshape(-1) if pre_dim_abs.size > 0 else np.zeros((0,), dtype=np.float32)
        post_dim_flat = post_dim_abs[:, :, dim_idx].reshape(-1) if post_dim_abs.size > 0 else np.zeros((0,), dtype=np.float32)
        per_dim.append(
            {
                "dim": int(dim_idx),
                "enabled": bool(dim_idx in allowed_dims),
                "pre_clip_mean_abs": float(np.mean(pre_dim_flat)) if pre_dim_flat.size > 0 else 0.0,
                "pre_clip_exceed_rate": float(np.mean(pre_dim_flat > threshold)) if pre_dim_flat.size > 0 else 0.0,
                "pre_clip_p95": float(np.quantile(pre_dim_flat, 0.95)) if pre_dim_flat.size > 0 else 0.0,
                "pre_clip_p99": float(np.quantile(pre_dim_flat, 0.99)) if pre_dim_flat.size > 0 else 0.0,
                "post_clip_mean_abs": float(np.mean(post_dim_flat)) if post_dim_flat.size > 0 else 0.0,
                "post_clip_exceed_rate": float(np.mean(post_dim_flat > threshold)) if post_dim_flat.size > 0 else 0.0,
                "post_clip_p95": float(np.quantile(post_dim_flat, 0.95)) if post_dim_flat.size > 0 else 0.0,
                "post_clip_p99": float(np.quantile(post_dim_flat, 0.99)) if post_dim_flat.size > 0 else 0.0,
            }
        )
    summary["per_dim"] = per_dim

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if args.out_json:
        out = pathlib.Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved analysis summary to {out}")


if __name__ == "__main__":
    main()
