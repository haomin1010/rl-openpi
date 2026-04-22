#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from collections import defaultdict
from typing import Any

import numpy as np

from openpi_online_ppo.data.local_lerobot_loader import ensure_local_hf_cache


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple LeRobot round datasets into one cumulative corpus.")
    p.add_argument("--src_dataset_roots", type=str, nargs="+", required=True)
    p.add_argument("--src_repo_ids", type=str, nargs="+", required=True)
    p.add_argument("--out_dataset_root", type=str, required=True)
    p.add_argument("--out_repo_id", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_dataset_rows(ds) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i in range(len(ds)):
        item = ds[i]
        row = {k: item[k] for k in item.keys()}
        ep = int(np.asarray(row["episode_index"]).item())
        grouped[ep].append(row)
    for ep in grouped:
        grouped[ep].sort(key=lambda x: int(np.asarray(x["frame_index"]).item()))
    return grouped


def _read_outcomes(dataset_root: pathlib.Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "meta" / "online_episode_outcomes.jsonl"
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["episode_index"])] = rec
    return out


def _feature_frame(row: dict[str, Any], feature_keys: list[str]) -> dict[str, Any]:
    frame = {k: row[k] for k in feature_keys if k in row}
    if "task" in row:
        frame["task"] = str(row["task"])
    return frame


def main() -> None:
    args = _parse_args()
    if len(args.src_dataset_roots) != len(args.src_repo_ids):
        raise ValueError("--src_dataset_roots and --src_repo_ids must have the same length.")

    ensure_local_hf_cache()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    src_roots = [pathlib.Path(x).resolve() for x in args.src_dataset_roots]
    out_root = pathlib.Path(args.out_dataset_root).resolve()
    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)

    first_info = _load_json(src_roots[0] / "meta" / "info.json")
    features = dict(first_info["features"])
    fps = int(first_info["fps"])
    robot_type = str(first_info["robot_type"])

    merged = LeRobotDataset.create(
        repo_id=str(args.out_repo_id),
        root=out_root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=False,
    )

    out_outcome_path = out_root / "meta" / "online_episode_outcomes.jsonl"
    out_outcome_path.parent.mkdir(parents=True, exist_ok=True)
    feature_keys = list(features.keys())
    merged_episode_index = 0

    for src_root, src_repo_id in zip(src_roots, args.src_repo_ids, strict=True):
        ds = LeRobotDataset(repo_id=src_repo_id, root=src_root)
        ep_rows = _group_dataset_rows(ds)
        outcomes = _read_outcomes(src_root)

        for src_ep in sorted(ep_rows.keys()):
            rows = ep_rows[src_ep]
            for row in rows:
                merged.add_frame(_feature_frame(row, feature_keys))
            merged.save_episode()

            outcome = outcomes.get(src_ep)
            if outcome is not None:
                rec = dict(outcome)
                rec["episode_index"] = int(merged_episode_index)
                with out_outcome_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            merged_episode_index += 1

    print(
        json.dumps(
            {
                "out_dataset_root": str(out_root),
                "out_repo_id": str(args.out_repo_id),
                "merged_episodes": int(merged_episode_index),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
