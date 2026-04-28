#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge multiple LeRobot round datasets into one cumulative corpus by using the first dataset as base."
    )
    p.add_argument("--src_dataset_roots", type=str, nargs="+", required=True)
    p.add_argument("--src_repo_ids", type=str, nargs="+", required=True)
    p.add_argument("--out_dataset_root", type=str, required=True)
    p.add_argument("--out_repo_id", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_success_outcome(*, episode_index: int, episode_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "success": True,
        "num_frames": int(episode_row["length"]),
        "step_lim": None,
        "seed": None,
        "prompt": str(episode_row["tasks"][0]) if episode_row.get("tasks") else None,
    }


def _is_joint_named_14d(spec: dict[str, Any]) -> bool:
    shape = tuple(spec.get("shape", ()))
    names = spec.get("names", [])
    if shape != (14,) and shape != [14]:
        return False
    if not names or not isinstance(names[0], list):
        return False
    first_name = str(names[0][0]) if names[0] else ""
    return first_name == "left_waist"


def _canonicalize_ee_delta_features(features: dict[str, Any], *, repo_id: str, root: pathlib.Path) -> dict[str, Any]:
    repo_hint = str(repo_id).lower()
    root_hint = str(root).lower()
    is_ee_delta = "ee_delta" in repo_hint or "ee_delta" in root_hint
    if not is_ee_delta:
        return copy.deepcopy(features)

    out = copy.deepcopy(features)
    state_spec = dict(out.get("observation.state", {}))
    action_spec = dict(out.get("action", {}))
    if _is_joint_named_14d(state_spec):
        state_spec["names"] = [[
            "left_x",
            "left_y",
            "left_z",
            "left_rotvec_x",
            "left_rotvec_y",
            "left_rotvec_z",
            "left_gripper",
            "right_x",
            "right_y",
            "right_z",
            "right_rotvec_x",
            "right_rotvec_y",
            "right_rotvec_z",
            "right_gripper",
        ]]
        out["observation.state"] = state_spec
    if _is_joint_named_14d(action_spec):
        action_spec["names"] = [[
            "left_dx",
            "left_dy",
            "left_dz",
            "left_drotvec_x",
            "left_drotvec_y",
            "left_drotvec_z",
            "left_gripper_target",
            "right_dx",
            "right_dy",
            "right_dz",
            "right_drotvec_x",
            "right_drotvec_y",
            "right_drotvec_z",
            "right_gripper_target",
        ]]
        out["action"] = action_spec
    return out


def _feature_signature(features: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in features.items():
        new_spec = copy.deepcopy(spec)
        if "shape" in new_spec:
            new_spec["shape"] = tuple(new_spec["shape"])
        out[key] = new_spec
    return out


def _episode_parquet_paths(dataset_root: pathlib.Path) -> dict[int, pathlib.Path]:
    out: dict[int, pathlib.Path] = {}
    for path in sorted(dataset_root.glob("data/chunk-*/*.parquet")):
        stem = path.stem
        if not stem.startswith("episode_"):
            continue
        ep_idx = int(stem.split("_")[-1])
        out[ep_idx] = path
    return out


def _task_rows_to_maps(rows: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, int]]:
    by_index = {int(row["task_index"]): str(row["task"]) for row in rows}
    by_task = {str(row["task"]): int(row["task_index"]) for row in rows}
    return by_index, by_task


def _build_value_train_rows(dataset_root: pathlib.Path, task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_by_index, _ = _task_rows_to_maps(task_rows)
    parquet_paths = _episode_parquet_paths(dataset_root)
    records: list[dict[str, Any]] = []
    for ep_idx in sorted(parquet_paths):
        table = pq.read_table(parquet_paths[ep_idx])
        names = set(table.schema.names)
        required = {"episode_index", "frame_index"}
        if not required.issubset(names):
            raise ValueError(
                f"Parquet missing required columns {sorted(required)}: {parquet_paths[ep_idx]}"
            )
        ep_col = table.column("episode_index").to_numpy(zero_copy_only=False)
        fi_col = table.column("frame_index").to_numpy(zero_copy_only=False)
        index_col = table.column("index").to_numpy(zero_copy_only=False) if "index" in names else None
        ti_col = table.column("task_index").to_numpy(zero_copy_only=False) if "task_index" in names else None
        mc_col = table.column("mc_return").to_numpy(zero_copy_only=False) if "mc_return" in names else None

        for i in range(table.num_rows):
            ti = int(ti_col[i]) if ti_col is not None else None
            rec = {
                "episode_index": int(ep_col[i]),
                "frame_index": int(fi_col[i]),
                "task_index": ti,
                "task": task_by_index.get(ti) if ti is not None else None,
                "mc_return": (float(mc_col[i]) if mc_col is not None else None),
            }
            if index_col is not None:
                rec["index"] = int(index_col[i])
            records.append(rec)
    if records and "index" in records[0]:
        records.sort(key=lambda r: int(r["index"]))
    return records


def _remap_stats_record(
    record: dict[str, Any],
    *,
    new_episode_index: int,
    new_task_index: int,
    new_index_start: int,
) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out["episode_index"] = int(new_episode_index)
    stats = out.get("stats", {})

    if "episode_index" in stats:
        stats["episode_index"]["min"] = [int(new_episode_index)]
        stats["episode_index"]["max"] = [int(new_episode_index)]
        stats["episode_index"]["mean"] = [float(new_episode_index)]
        stats["episode_index"]["std"] = [0.0]

    if "task_index" in stats:
        stats["task_index"]["min"] = [int(new_task_index)]
        stats["task_index"]["max"] = [int(new_task_index)]
        stats["task_index"]["mean"] = [float(new_task_index)]
        stats["task_index"]["std"] = [0.0]

    if "index" in stats:
        old_min = int(stats["index"]["min"][0])
        old_max = int(stats["index"]["max"][0])
        delta = int(new_index_start - old_min)
        stats["index"]["min"] = [old_min + delta]
        stats["index"]["max"] = [old_max + delta]
        stats["index"]["mean"] = [float(stats["index"]["mean"][0] + delta)]

    return out


def _rewrite_episode_table(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    *,
    new_episode_index: int,
    new_task_index: int,
    new_index_start: int,
) -> int:
    table = pq.read_table(src_path)
    num_rows = table.num_rows
    if num_rows <= 0:
        raise ValueError(f"Empty parquet episode file: {src_path}")

    arrays: list[pa.Array | pa.ChunkedArray] = []
    for field in table.schema:
        name = field.name
        if name == "episode_index":
            arr = pa.array(np.full((num_rows,), new_episode_index, dtype=np.int64), type=pa.int64())
        elif name == "task_index":
            arr = pa.array(np.full((num_rows,), new_task_index, dtype=np.int64), type=pa.int64())
        elif name == "index":
            arr = pa.array(
                np.arange(new_index_start, new_index_start + num_rows, dtype=np.int64),
                type=pa.int64(),
            )
        else:
            arr = table[name].combine_chunks()
        arrays.append(arr)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    new_table = pa.Table.from_arrays(arrays, schema=table.schema)
    pq.write_table(new_table, dst_path)
    return num_rows


def main() -> None:
    args = _parse_args()
    if len(args.src_dataset_roots) != len(args.src_repo_ids):
        raise ValueError("--src_dataset_roots and --src_repo_ids must have the same length.")
    if len(args.src_dataset_roots) < 1:
        raise ValueError("At least one source dataset is required.")

    src_roots = [pathlib.Path(x).resolve() for x in args.src_dataset_roots]
    out_root = pathlib.Path(args.out_dataset_root).resolve()

    base_root = src_roots[0]
    base_repo_id = str(args.src_repo_ids[0])

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    if out_root.exists():
        raise FileExistsError(f"Output dataset root already exists: {out_root}")

    shutil.copytree(base_root, out_root)

    out_info = _load_json(out_root / "meta" / "info.json")
    base_features = _canonicalize_ee_delta_features(dict(out_info["features"]), repo_id=base_repo_id, root=base_root)
    out_info["features"] = base_features
    out_info["repo_id"] = str(args.out_repo_id)

    out_episodes = _load_jsonl(out_root / "meta" / "episodes.jsonl")
    out_episode_stats = _load_jsonl(out_root / "meta" / "episodes_stats.jsonl")
    out_tasks = _load_jsonl(out_root / "meta" / "tasks.jsonl")
    out_outcomes = _load_jsonl(out_root / "meta" / "online_episode_outcomes.jsonl")
    out_outcomes_by_ep = {int(row["episode_index"]): row for row in out_outcomes}

    for ep_row in out_episodes:
        ep_idx = int(ep_row["episode_index"])
        if ep_idx not in out_outcomes_by_ep:
            out_outcomes_by_ep[ep_idx] = _default_success_outcome(episode_index=ep_idx, episode_row=ep_row)

    _, task_to_out_index = _task_rows_to_maps(out_tasks)
    next_task_index = len(out_tasks)
    next_episode_index = len(out_episodes)
    next_global_index = int(out_info.get("total_frames", 0))

    for src_root, src_repo_id in zip(src_roots[1:], args.src_repo_ids[1:], strict=True):
        src_info = _load_json(src_root / "meta" / "info.json")
        src_features = _canonicalize_ee_delta_features(
            dict(src_info["features"]),
            repo_id=str(src_repo_id),
            root=src_root,
        )
        if _feature_signature(src_features) != _feature_signature(base_features):
            raise ValueError(f"Feature spec mismatch for source dataset: root={src_root} repo_id={src_repo_id}")
        if int(src_info["fps"]) != int(out_info["fps"]):
            raise ValueError(f"FPS mismatch: base={out_info['fps']} src={src_info['fps']} root={src_root}")
        if str(src_info["robot_type"]) != str(out_info["robot_type"]):
            raise ValueError(
                f"Robot type mismatch: base={out_info['robot_type']} src={src_info['robot_type']} root={src_root}"
            )

        src_episodes = _load_jsonl(src_root / "meta" / "episodes.jsonl")
        src_episode_stats = _load_jsonl(src_root / "meta" / "episodes_stats.jsonl")
        src_tasks = _load_jsonl(src_root / "meta" / "tasks.jsonl")
        src_outcomes = _load_jsonl(src_root / "meta" / "online_episode_outcomes.jsonl")
        src_task_by_index, _ = _task_rows_to_maps(src_tasks)
        src_parquets = _episode_parquet_paths(src_root)

        src_stats_by_ep = {int(row["episode_index"]): row for row in src_episode_stats}
        src_outcomes_by_ep = {int(row["episode_index"]): row for row in src_outcomes}

        for ep_row in src_episodes:
            src_ep = int(ep_row["episode_index"])
            src_task = str(ep_row["tasks"][0])
            if src_task not in task_to_out_index:
                task_to_out_index[src_task] = next_task_index
                out_tasks.append({"task_index": int(next_task_index), "task": src_task})
                next_task_index += 1
            new_task_index = int(task_to_out_index[src_task])
            new_episode_index = int(next_episode_index)

            src_parquet = src_parquets.get(src_ep)
            if src_parquet is None:
                raise FileNotFoundError(f"Missing parquet file for episode_index={src_ep} in {src_root}")
            dst_parquet = out_root / "data" / "chunk-000" / f"episode_{new_episode_index:06d}.parquet"
            frame_count = _rewrite_episode_table(
                src_parquet,
                dst_parquet,
                new_episode_index=new_episode_index,
                new_task_index=new_task_index,
                new_index_start=next_global_index,
            )

            out_episodes.append(
                {
                    "episode_index": new_episode_index,
                    "tasks": [src_task],
                    "length": int(ep_row["length"]),
                }
            )

            src_stats = src_stats_by_ep.get(src_ep)
            if src_stats is not None:
                out_episode_stats.append(
                    _remap_stats_record(
                        src_stats,
                        new_episode_index=new_episode_index,
                        new_task_index=new_task_index,
                        new_index_start=next_global_index,
                    )
                )

            src_outcome = src_outcomes_by_ep.get(src_ep)
            if src_outcome is None:
                new_outcome = _default_success_outcome(episode_index=new_episode_index, episode_row=ep_row)
            else:
                new_outcome = dict(src_outcome)
                new_outcome["episode_index"] = new_episode_index
            out_outcomes_by_ep[new_episode_index] = new_outcome

            next_episode_index += 1
            next_global_index += frame_count

    out_info["total_episodes"] = int(len(out_episodes))
    out_info["total_frames"] = int(next_global_index)
    out_info["total_tasks"] = int(len(out_tasks))
    out_info["splits"] = {"train": f"0:{len(out_episodes)}"}

    _write_json(out_root / "meta" / "info.json", out_info)
    _write_jsonl(out_root / "meta" / "episodes.jsonl", out_episodes)
    _write_jsonl(out_root / "meta" / "episodes_stats.jsonl", out_episode_stats)
    _write_jsonl(out_root / "meta" / "tasks.jsonl", out_tasks)
    out_outcomes = [out_outcomes_by_ep[i] for i in sorted(out_outcomes_by_ep)]
    if out_outcomes:
        _write_jsonl(out_root / "meta" / "online_episode_outcomes.jsonl", out_outcomes)
    value_rows = _build_value_train_rows(out_root, out_tasks)
    _write_jsonl(out_root / "meta" / "value_train_rows.jsonl", value_rows)

    print(
        json.dumps(
            {
                "out_dataset_root": str(out_root),
                "out_repo_id": str(args.out_repo_id),
                "merged_episodes": int(len(out_episodes)),
                "merged_frames": int(next_global_index),
                "value_train_rows": int(len(value_rows)),
                "merge_mode": "copy_base_then_append",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
