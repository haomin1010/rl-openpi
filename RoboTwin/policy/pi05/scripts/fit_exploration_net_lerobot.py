#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def _uniform_direction(size: int) -> np.ndarray:
    v = np.random.normal(0.0, 1.0, size=(size,)).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _formula_delta(flat_action: np.ndarray, *, kp: float, rmin: float, rmax: float) -> np.ndarray:
    direction = _uniform_direction(flat_action.shape[0])
    radius = float(rmin + (rmax - rmin) * np.clip(kp, 0.0, 1.0))
    return direction * radius


def _extract_vec(item: dict, key: str) -> np.ndarray:
    if key not in item:
        raise KeyError(f"Key `{key}` not found in dataset sample.")
    arr = item[key]
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _make_nested_obs(dotted_key: str, value) -> dict:
    parts = dotted_key.split(".")
    root = {}
    node = root
    for p in parts[:-1]:
        nxt = {}
        node[p] = nxt
        node = nxt
    node[parts[-1]] = value
    return root


def main() -> None:
    p = argparse.ArgumentParser(description="Fit a first-version linear action perturb network from LeRobot dataset.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--obs_key", type=str, default="observation.state")
    p.add_argument("--action_key", type=str, default="action")
    p.add_argument("--obs_dim", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=10000)
    p.add_argument("--radius_min", type=float, default=0.0)
    p.add_argument("--radius_max", type=float, default=0.15)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--keyframe_source", type=str, default="all", choices=("all", "annotations", "model"))
    p.add_argument("--keyframe_annotations_json", type=str, default=None)
    p.add_argument("--keyframe_model_ckpt", type=str, default=None)
    p.add_argument("--keyframe_model_threshold", type=float, default=0.5)
    p.add_argument("--keyframe_chunk_size", type=int, default=32)
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi_online_ppo.rl.exploration import LinearKeyframeNet

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    total_n = len(ds)
    if total_n <= 0:
        raise RuntimeError("Dataset is empty.")

    selected_indices = np.arange(total_n, dtype=np.int32)
    if args.keyframe_source != "all":
        kf_map: dict[int, set[int]] = {}
        if args.keyframe_source == "annotations":
            if not args.keyframe_annotations_json:
                raise ValueError("--keyframe_annotations_json is required when keyframe_source=annotations")
            ann = json.loads(pathlib.Path(args.keyframe_annotations_json).read_text(encoding="utf-8"))
            kf_map = {int(k): set(int(x) for x in v) for k, v in ann.get("episodes", {}).items()}
        elif args.keyframe_source == "model":
            if not args.keyframe_model_ckpt:
                raise ValueError("--keyframe_model_ckpt is required when keyframe_source=model")
            model = LinearKeyframeNet.load(args.keyframe_model_ckpt)
            for i in range(total_n):
                item = ds[i]
                ep = int(item["episode_index"])
                model_obs_key = str(model.obs_key)
                key_for_item = model_obs_key if model_obs_key in item else str(args.obs_key)
                obs = _make_nested_obs(model_obs_key, item[key_for_item])
                prob = model.predict_prob(obs)
                if prob >= float(args.keyframe_model_threshold):
                    fi = int(item["frame_index"])
                    if ep not in kf_map:
                        kf_map[ep] = set()
                    kf_map[ep].add(fi)

        keep = []
        c = int(max(1, args.keyframe_chunk_size))
        for i in range(total_n):
            item = ds[i]
            ep = int(item["episode_index"])
            fi = int(item["frame_index"])
            kfs = kf_map.get(ep, set())
            if any((kf - c) <= fi <= (kf + c) for kf in kfs):
                keep.append(i)
        selected_indices = np.asarray(keep, dtype=np.int32)

    if selected_indices.size == 0:
        raise RuntimeError("No frames selected for exploration-net fitting.")
    np.random.shuffle(selected_indices)
    n = min(int(selected_indices.size), int(args.max_samples))
    selected_indices = selected_indices[:n]
    if n <= 0:
        raise RuntimeError("No samples available.")

    # Determine dimensions from first sample.
    first = ds[0]
    first_action = _extract_vec(first, args.action_key)
    action_dim = int(first_action.shape[0])
    obs_dim = int(args.obs_dim)
    latent_dim = int(args.latent_dim)
    in_dim = action_dim + obs_dim + 1 + latent_dim
    out_dim = action_dim

    xs = np.zeros((n, in_dim), dtype=np.float32)
    ys = np.zeros((n, out_dim), dtype=np.float32)

    for i, ds_idx in enumerate(selected_indices.tolist()):
        item = ds[int(ds_idx)]
        action = _extract_vec(item, args.action_key)
        obs = _extract_vec(item, args.obs_key)
        obs_pad = np.zeros((obs_dim,), dtype=np.float32)
        obs_pad[: min(obs_dim, obs.shape[0])] = obs[: min(obs_dim, obs.shape[0])]
        kp = np.random.uniform(0.0, 1.0)
        z = np.random.normal(0.0, 1.0, size=(latent_dim,)).astype(np.float32)
        x = np.concatenate([action, obs_pad, np.asarray([kp], dtype=np.float32), z], axis=0)
        y = _formula_delta(action, kp=kp, rmin=float(args.radius_min), rmax=float(args.radius_max))
        xs[i] = x
        ys[i] = y

    x_mean = xs.mean(axis=0, keepdims=True)
    x_std = xs.std(axis=0, keepdims=True) + 1e-6
    xsn = (xs - x_mean) / x_std

    # Ridge regression: W = (X^T X + λI)^-1 X^T Y
    xtx = xsn.T @ xsn
    xty = xsn.T @ ys
    reg = np.eye(xtx.shape[0], dtype=np.float32) * float(args.ridge)
    w = np.linalg.solve(xtx + reg, xty).astype(np.float32)
    b = np.zeros((out_dim,), dtype=np.float32)

    out_path = pathlib.Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        in_dim=np.asarray([in_dim], dtype=np.int32),
        out_dim=np.asarray([out_dim], dtype=np.int32),
        w=w,
        b=b,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        obs_dim=np.asarray([obs_dim], dtype=np.int32),
        latent_dim=np.asarray([latent_dim], dtype=np.int32),
    )
    print(f"Saved exploration net checkpoint: {out_path}")


if __name__ == "__main__":
    main()
