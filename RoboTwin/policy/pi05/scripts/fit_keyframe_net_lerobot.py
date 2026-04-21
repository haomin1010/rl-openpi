#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    p = argparse.ArgumentParser(description="Fit a small linear keyframe classifier from human annotations.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--obs_key", type=str, default="observation.state")
    p.add_argument("--obs_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=1e-4)
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ann = json.loads(pathlib.Path(args.annotations_json).read_text(encoding="utf-8"))
    ann_eps = {int(k): set(int(x) for x in v) for k, v in ann.get("episodes", {}).items()}

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    n = len(ds)
    if n <= 0:
        raise RuntimeError("Dataset is empty.")

    obs_dim = int(args.obs_dim)
    x = np.zeros((n, obs_dim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        item = ds[i]
        ep = int(item["episode_index"])
        fi = int(item["frame_index"])
        arr = item[args.obs_key]
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        x[i, : min(obs_dim, arr.shape[0])] = arr[: min(obs_dim, arr.shape[0])]
        y[i] = 1.0 if fi in ann_eps.get(ep, set()) else 0.0

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    xs = (x - x_mean) / x_std

    w = np.zeros((obs_dim,), dtype=np.float32)
    b = np.zeros((1,), dtype=np.float32)
    lr = float(args.lr)
    l2 = float(args.l2)
    for _ in range(int(args.epochs)):
        logits = xs @ w + b[0]
        p_hat = _sigmoid(logits)
        err = (p_hat - y).astype(np.float32)
        grad_w = (xs.T @ err) / float(n) + l2 * w
        grad_b = np.asarray([np.mean(err)], dtype=np.float32)
        w -= lr * grad_w
        b -= lr * grad_b

    out = pathlib.Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        in_dim=np.asarray([obs_dim], dtype=np.int32),
        w=w.astype(np.float32),
        b=b.astype(np.float32),
        obs_key=np.asarray(args.obs_key),
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
    )
    print(f"Saved keyframe net: {out}")


if __name__ == "__main__":
    main()

