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


def _progress(msg: str) -> None:
    print(msg, flush=True)


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


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("Expected at least one integer in --lowfreq_ks")
    return out


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
                start = int(item["start"])
                end = int(item["end"])
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare smoothness when sigma-net noise is restricted to low-frequency DCT coefficients.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    p.add_argument("--net", type=str, required=True)
    p.add_argument("--action_key", type=str, default="action")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--fast_tokenizer_path", type=str, default="physical-intelligence/fast")
    p.add_argument("--action_noise_dims", type=str, default="auto")
    p.add_argument("--max_chunks", type=int, default=0)
    p.add_argument("--num_samples_per_chunk", type=int, default=32)
    p.add_argument("--threshold", type=float, default=-1.0)
    p.add_argument("--lowfreq_ks", type=str, default="1,2,4,8,16,32")
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
    dct_mat = _build_ortho_dct_matrix(chunk_size)
    dct_chunks = (np.einsum("tk,nta->nka", dct_mat, chunk_arr) * dct_scale).astype(np.float32)

    threshold = float(args.threshold) if float(args.threshold) > 0 else float(net.action_delta_limit)
    if threshold <= 0:
        raise ValueError("Threshold must be > 0, either via --threshold or checkpoint action_delta_limit.")
    num_samples = int(args.num_samples_per_chunk)
    if num_samples <= 0:
        raise ValueError("--num_samples_per_chunk must be > 0")

    ks = sorted(set(max(1, min(chunk_size, int(k))) for k in _parse_int_list(args.lowfreq_ks)))
    variants = [("all", chunk_size)] + [(f"first_{k}", k) for k in ks]
    _progress(
        f"analysis_start num_chunks={int(chunk_arr.shape[0])} "
        f"num_samples_per_chunk={num_samples} variants={[name for name, _ in variants]}"
    )

    quantiles = [0.5, 0.9, 0.95, 0.99]
    results: list[dict[str, Any]] = []

    for variant_name, keep_k in variants:
        _progress(f"variant_start name={variant_name} keep_lowfreq_k={int(keep_k)}")
        abs_flat_all: list[np.ndarray] = []
        sample_max_all: list[float] = []
        vel_l2_all: list[float] = []
        acc_l2_all: list[float] = []
        acc_abs_max_all: list[float] = []
        rescale_count = 0

        for chunk_idx in range(chunk_arr.shape[0]):
            if chunk_idx == 0 or (chunk_idx + 1) % 100 == 0 or (chunk_idx + 1) == chunk_arr.shape[0]:
                _progress(
                    f"variant_progress name={variant_name} "
                    f"chunk={chunk_idx + 1}/{int(chunk_arr.shape[0])}"
                )
            coeffs = dct_chunks[chunk_idx]
            for _ in range(num_samples):
                delta_dct = np.zeros_like(coeffs, dtype=np.float32)
                for dim_idx in allowed_dims:
                    dim_onehot = np.zeros((action_dim,), dtype=np.float32)
                    dim_onehot[int(dim_idx)] = 1.0
                    x = np.concatenate([coeffs[:, int(dim_idx)], dim_onehot], axis=0)
                    sigma = net._sigma_from_logits(net._mlp(x))
                    eps = rng.normal(0.0, 1.0, size=(chunk_size,)).astype(np.float32)
                    sigma_masked = np.zeros_like(sigma, dtype=np.float32)
                    sigma_masked[:keep_k] = sigma[:keep_k]
                    delta_dct[:, int(dim_idx)] = sigma_masked * eps

                delta_action = dct_mat.T @ (delta_dct / max(float(net.dct_scale), 1e-6))
                max_abs = float(np.max(np.abs(delta_action))) if delta_action.size > 0 else 0.0
                if max_abs > threshold:
                    delta_action = delta_action * float(threshold / max_abs)
                    rescale_count += 1

                abs_flat = np.abs(delta_action).reshape(-1).astype(np.float32)
                abs_flat_all.append(abs_flat)
                sample_max_all.append(float(np.max(abs_flat)) if abs_flat.size > 0 else 0.0)

                vel = np.diff(delta_action, axis=0)
                acc = delta_action[2:, :] - 2.0 * delta_action[1:-1, :] + delta_action[:-2, :]
                vel_l2_all.append(float(np.linalg.norm(vel, ord=2) / max(1, vel.size ** 0.5)) if vel.size > 0 else 0.0)
                acc_l2_all.append(float(np.linalg.norm(acc, ord=2) / max(1, acc.size ** 0.5)) if acc.size > 0 else 0.0)
                acc_abs_max_all.append(float(np.max(np.abs(acc))) if acc.size > 0 else 0.0)

        abs_flat_np = np.concatenate(abs_flat_all, axis=0) if abs_flat_all else np.zeros((0,), dtype=np.float32)
        sample_max_np = np.asarray(sample_max_all, dtype=np.float32)
        vel_l2_np = np.asarray(vel_l2_all, dtype=np.float32)
        acc_l2_np = np.asarray(acc_l2_all, dtype=np.float32)
        acc_abs_max_np = np.asarray(acc_abs_max_all, dtype=np.float32)

        results.append(
            {
                "variant": variant_name,
                "keep_lowfreq_k": int(keep_k),
                "num_draws": int(len(sample_max_all)),
                "rescale_trigger_rate": float(rescale_count / max(1, len(sample_max_all))),
                "delta_abs_mean": float(np.mean(abs_flat_np)) if abs_flat_np.size > 0 else 0.0,
                "delta_abs_exceed_rate": float(np.mean(abs_flat_np > threshold)) if abs_flat_np.size > 0 else 0.0,
                "delta_abs_near_limit_90": float(np.mean(abs_flat_np >= 0.9 * threshold)) if abs_flat_np.size > 0 else 0.0,
                "delta_abs_near_limit_95": float(np.mean(abs_flat_np >= 0.95 * threshold)) if abs_flat_np.size > 0 else 0.0,
                **_safe_quantiles(abs_flat_np, quantiles),
                **{f"sample_max_{k}": v for k, v in _safe_quantiles(sample_max_np, [0.5, 0.9, 0.95, 0.99]).items()},
                "vel_l2_mean": float(np.mean(vel_l2_np)) if vel_l2_np.size > 0 else 0.0,
                "vel_l2_p95": float(np.quantile(vel_l2_np, 0.95)) if vel_l2_np.size > 0 else 0.0,
                "acc_l2_mean": float(np.mean(acc_l2_np)) if acc_l2_np.size > 0 else 0.0,
                "acc_l2_p95": float(np.quantile(acc_l2_np, 0.95)) if acc_l2_np.size > 0 else 0.0,
                "acc_abs_max_mean": float(np.mean(acc_abs_max_np)) if acc_abs_max_np.size > 0 else 0.0,
                "acc_abs_max_p95": float(np.quantile(acc_abs_max_np, 0.95)) if acc_abs_max_np.size > 0 else 0.0,
            }
        )
        _progress(
            f"variant_done name={variant_name} "
            f"delta_abs_mean={results[-1]['delta_abs_mean']:.6f} "
            f"acc_l2_mean={results[-1]['acc_l2_mean']:.6f} "
            f"acc_abs_max_mean={results[-1]['acc_abs_max_mean']:.6f}"
        )

    summary = {
        "net": str(pathlib.Path(args.net).resolve()),
        "dataset_root": str(pathlib.Path(args.dataset_root).resolve()),
        "repo_id": str(args.repo_id),
        "threshold": float(threshold),
        "chunk_size": int(chunk_size),
        "action_dim": int(action_dim),
        "allowed_dims": [int(x) for x in allowed_dims],
        "num_chunks": int(chunk_arr.shape[0]),
        "num_samples_per_chunk": int(num_samples),
        "variants": results,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if args.out_json:
        out = pathlib.Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved low-frequency mask analysis to {out}")


if __name__ == "__main__":
    main()
