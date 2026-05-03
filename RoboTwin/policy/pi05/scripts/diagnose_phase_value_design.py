#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose phase-conditioned value target design.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--value_target_mode", type=str, default="evorl_normalized", choices=("evorl_normalized", "legacy"))
    p.add_argument("--mc_gamma", type=float, default=0.99)
    p.add_argument("--value_c_fail_coef", type=float, default=1.0)
    p.add_argument("--progress_bins", type=int, default=10)
    return p.parse_args()


def _resolve_success_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"success", "succeeded", "true", "1", "yes"}:
            return True
        if v in {"failure", "failed", "false", "0", "no"}:
            return False
    return bool(value)


def _load_annotations(path: pathlib.Path) -> dict[int, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    episodes_raw = raw.get("episodes", {})
    if not isinstance(episodes_raw, dict):
        raise ValueError("annotations_json must contain an `episodes` object.")
    parsed: dict[int, list[dict[str, Any]]] = {}
    for ep_key, spans_raw in episodes_raw.items():
        ep = int(ep_key)
        if spans_raw is None:
            parsed[ep] = []
            continue
        if not isinstance(spans_raw, list):
            raise ValueError(f"Episode {ep} annotations must be a list.")
        spans: list[dict[str, Any]] = []
        prev_end = -1
        for item in spans_raw:
            if not isinstance(item, dict):
                raise ValueError(f"Episode {ep} phase item must be object, got {type(item).__name__}")
            name = str(item.get("name", "")).strip()
            start = int(item.get("start"))
            end = int(item.get("end"))
            if not name:
                raise ValueError(f"Episode {ep} has empty phase name.")
            if end < start:
                raise ValueError(f"Episode {ep} phase `{name}` invalid range [{start}, {end}]")
            if start <= prev_end:
                raise ValueError(f"Episode {ep} phase `{name}` overlaps previous range.")
            spans.append({"name": name, "start": start, "end": end})
            prev_end = end
        parsed[ep] = spans
    return parsed


def _load_outcomes(path: pathlib.Path) -> dict[int, bool]:
    outcomes: dict[int, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            outcomes[int(rec["episode_index"])] = _resolve_success_bool(rec.get("success", False))
    return outcomes


def _compute_phase_target(
    *,
    pos: int,
    phase_len: int,
    phase_success: bool,
    value_target_mode: str,
    mc_gamma: float,
    value_c_fail_coef: float,
    clip_min: float = -1.0,
    clip_max: float = 0.0,
) -> float:
    phase_scale = float(max(1, phase_len - 1))
    if value_target_mode == "evorl_normalized":
        remaining_steps = float(phase_len - pos - 1)
        g = -remaining_steps
        if not phase_success:
            c_fail = float(phase_scale)
            g -= (float(value_c_fail_coef) ** remaining_steps) * c_fail
            denom = float(phase_scale) + c_fail
        else:
            denom = float(phase_scale)
        return float(np.clip(g / max(1.0, denom), clip_min, clip_max))

    final_reward = 0.0 if phase_success else -100.0
    returns = [0.0] * int(phase_len)
    returns[-1] = final_reward
    for idx in range(phase_len - 2, -1, -1):
        returns[idx] = -1.0 + float(mc_gamma) * returns[idx + 1]
    return float(returns[pos])


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _safe_std(xs: list[float]) -> float:
    return float(np.std(xs)) if xs else float("nan")


def _to_progress_bin(progress: float, num_bins: int) -> int:
    if num_bins <= 1:
        return 0
    clipped = float(np.clip(progress, 0.0, 1.0))
    return min(int(math.floor(clipped * num_bins)), num_bins - 1)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def main() -> None:
    args = _parse_args()
    dataset_root = pathlib.Path(args.dataset_root).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations(pathlib.Path(args.annotations_json).resolve())
    outcomes = _load_outcomes(dataset_root / "meta" / "online_episode_outcomes.jsonl")

    rows: list[dict[str, Any]] = []
    for ep in sorted(annotations.keys()):
        spans = annotations[ep]
        if not spans:
            continue
        episode_success = bool(outcomes.get(ep, False))
        for span_idx, span in enumerate(spans):
            phase_name = str(span["name"])
            start = int(span["start"])
            end = int(span["end"])
            phase_len = end - start + 1
            phase_success = bool(episode_success or span_idx < len(spans) - 1)
            for pos, frame_idx in enumerate(range(start, end + 1)):
                progress = 1.0 if phase_len <= 1 else float(pos) / float(phase_len - 1)
                target = _compute_phase_target(
                    pos=pos,
                    phase_len=phase_len,
                    phase_success=phase_success,
                    value_target_mode=args.value_target_mode,
                    mc_gamma=float(args.mc_gamma),
                    value_c_fail_coef=float(args.value_c_fail_coef),
                )
                rows.append(
                    {
                        "episode_index": ep,
                        "phase_index": int(span_idx),
                        "phase_name": phase_name,
                        "frame_index": int(frame_idx),
                        "phase_start": start,
                        "phase_end": end,
                        "phase_len": phase_len,
                        "is_last_phase": int(span_idx == len(spans) - 1),
                        "episode_success": int(episode_success),
                        "phase_success": int(phase_success),
                        "progress": float(progress),
                        "progress_bin": int(_to_progress_bin(progress, int(args.progress_bins))),
                        "target": float(target),
                    }
                )

    if not rows:
        raise RuntimeError("No phase samples constructed.")

    target_arr = np.asarray([r["target"] for r in rows], dtype=np.float32)
    progress_arr = np.asarray([r["progress"] for r in rows], dtype=np.float32)
    phase_names = [str(r["phase_name"]) for r in rows]

    # Baseline 1: global mean.
    global_mean = float(np.mean(target_arr))
    pred_global = np.full_like(target_arr, global_mean)

    # Baseline 2: per-phase-name mean.
    phase_mean: dict[str, float] = {}
    phase_to_targets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        phase_to_targets[str(r["phase_name"])].append(float(r["target"]))
    for name, vals in phase_to_targets.items():
        phase_mean[name] = float(np.mean(np.asarray(vals, dtype=np.float32)))
    pred_phase = np.asarray([phase_mean[str(name)] for name in phase_names], dtype=np.float32)

    # Baseline 3: per-(phase_name, progress_bin) mean. This is an oracle-ish test for whether
    # target design becomes easy once progress is explicitly available.
    phase_bin_mean: dict[tuple[str, int], float] = {}
    phase_bin_targets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (str(r["phase_name"]), int(r["progress_bin"]))
        phase_bin_targets[key].append(float(r["target"]))
    for key, vals in phase_bin_targets.items():
        phase_bin_mean[key] = float(np.mean(np.asarray(vals, dtype=np.float32)))
    pred_phase_bin = np.asarray(
        [phase_bin_mean[(str(r["phase_name"]), int(r["progress_bin"]))] for r in rows],
        dtype=np.float32,
    )

    # Per-phase diagnostics.
    per_phase_summary: list[dict[str, Any]] = []
    per_phase_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        per_phase_rows[str(r["phase_name"])].append(r)
    for phase_name in sorted(per_phase_rows.keys()):
        rs = per_phase_rows[phase_name]
        lens = [int(r["phase_len"]) for r in rs if float(r["progress"]) == 0.0]
        span_count = len([r for r in rs if float(r["progress"]) == 0.0])
        failed_span_count = len([r for r in rs if float(r["progress"]) == 0.0 and int(r["phase_success"]) == 0])
        target_std_by_bin: list[float] = []
        for bin_idx in range(int(args.progress_bins)):
            vals = [float(r["target"]) for r in rs if int(r["progress_bin"]) == bin_idx]
            if len(vals) >= 2:
                target_std_by_bin.append(float(np.std(np.asarray(vals, dtype=np.float32))))
        per_phase_summary.append(
            {
                "phase_name": phase_name,
                "num_samples": int(len(rs)),
                "num_spans": int(span_count),
                "failed_spans": int(failed_span_count),
                "failed_span_ratio": float(failed_span_count / max(1, span_count)),
                "phase_len_mean": _safe_mean([float(x) for x in lens]),
                "phase_len_std": _safe_std([float(x) for x in lens]),
                "phase_len_min": int(min(lens)) if lens else None,
                "phase_len_max": int(max(lens)) if lens else None,
                "target_mean": float(np.mean([float(r["target"]) for r in rs])),
                "target_std": float(np.std([float(r["target"]) for r in rs])),
                "within_progress_bin_target_std_mean": _safe_mean(target_std_by_bin),
                "within_progress_bin_target_std_max": float(max(target_std_by_bin)) if target_std_by_bin else float("nan"),
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "annotations_json": str(pathlib.Path(args.annotations_json).resolve()),
        "value_target_mode": str(args.value_target_mode),
        "num_phase_names": int(len(per_phase_rows)),
        "num_phase_samples": int(len(rows)),
        "num_failed_phase_samples": int(sum(1 for r in rows if int(r["phase_success"]) == 0)),
        "failed_phase_sample_ratio": float(sum(1 for r in rows if int(r["phase_success"]) == 0) / max(1, len(rows))),
        "global_target_mean": float(np.mean(target_arr)),
        "global_target_std": float(np.std(target_arr)),
        "target_progress_corr": float(np.corrcoef(progress_arr, target_arr)[0, 1]),
        "baselines": {
            "global_mean": {
                "mae": _mae(target_arr, pred_global),
                "rmse": _rmse(target_arr, pred_global),
            },
            "phase_name_mean": {
                "mae": _mae(target_arr, pred_phase),
                "rmse": _rmse(target_arr, pred_phase),
            },
            "phase_name_plus_progress_bin_mean": {
                "mae": _mae(target_arr, pred_phase_bin),
                "rmse": _rmse(target_arr, pred_phase_bin),
            },
        },
    }

    summary_path = out_dir / "summary.json"
    per_phase_path = out_dir / "per_phase_summary.csv"
    sample_path = out_dir / "sample_rows.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with per_phase_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase_name",
                "num_samples",
                "num_spans",
                "failed_spans",
                "failed_span_ratio",
                "phase_len_mean",
                "phase_len_std",
                "phase_len_min",
                "phase_len_max",
                "target_mean",
                "target_std",
                "within_progress_bin_target_std_mean",
                "within_progress_bin_target_std_max",
            ],
        )
        writer.writeheader()
        for row in per_phase_summary:
            writer.writerow(row)

    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_index",
                "phase_index",
                "phase_name",
                "frame_index",
                "phase_start",
                "phase_end",
                "phase_len",
                "is_last_phase",
                "episode_success",
                "phase_success",
                "progress",
                "progress_bin",
                "target",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-phase summary: {per_phase_path}")
    print(f"Saved sample rows: {sample_path}")


if __name__ == "__main__":
    main()
