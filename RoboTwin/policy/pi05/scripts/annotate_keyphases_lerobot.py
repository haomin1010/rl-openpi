#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from openpi_online_ppo.data.local_lerobot_loader import ensure_local_hf_cache


def _to_hwc_uint8(img) -> np.ndarray:
    arr = img
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 image, got shape={arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _load_episode_rows_from_meta(dataset_root: pathlib.Path) -> tuple[dict[int, list[tuple[int, int]]], dict[int, int]]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes metadata: {episodes_path}")

    ep_rows: dict[int, list[tuple[int, int]]] = {}
    lengths: dict[int, int] = {}
    next_ds_idx = 0

    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec["episode_index"])
            length = int(rec["length"])
            lengths[ep] = length
            ep_rows[ep] = [(next_ds_idx + frame_idx, frame_idx) for frame_idx in range(length)]
            next_ds_idx += length

    return ep_rows, lengths


def _export_episode_images(ds, rows: list[tuple[int, int]], *, image_key: str, out_dir: pathlib.Path) -> None:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    for ds_idx, frame_idx in rows:
        item = ds[ds_idx]
        img = _to_hwc_uint8(item[image_key])
        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        if out_path.exists():
            continue
        cv2.imwrite(str(out_path), img[:, :, ::-1])


def _parse_phase_entry(raw: str, *, num_frames: int) -> dict[str, int | str]:
    text = raw.strip()
    if not text:
        raise ValueError("Empty keyphase entry.")
    if ":" not in text:
        raise ValueError(f"Expected `name:start-end`, got: {raw!r}")
    name, frame_range = text.split(":", 1)
    phase_name = name.strip()
    if not phase_name:
        raise ValueError(f"Missing phase name in: {raw!r}")
    if "-" not in frame_range:
        raise ValueError(f"Expected frame range `start-end`, got: {raw!r}")
    start_s, end_s = frame_range.split("-", 1)
    start = max(0, min(num_frames - 1, int(start_s.strip())))
    end = max(0, min(num_frames - 1, int(end_s.strip())))
    if end < start:
        raise ValueError(f"Invalid range `{start}-{end}` in: {raw!r}")
    return {"name": phase_name, "start": start, "end": end}


def _parse_phase_names(raw: str) -> list[str]:
    names = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not names:
        raise ValueError("Phase name list must not be empty.")
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate phase name: {name}")
        seen.add(name)
        deduped.append(name)
    return deduped


def _parse_indexed_phase_entry(
    raw: str,
    *,
    num_frames: int,
    phase_names: list[str],
) -> dict[str, int | str]:
    text = raw.strip()
    if not text:
        raise ValueError("Empty keyphase entry.")
    if ":" not in text:
        raise ValueError(f"Expected `index:start-end`, got: {raw!r}")
    index_s, frame_range = text.split(":", 1)
    phase_idx = int(index_s.strip())
    if phase_idx < 1 or phase_idx > len(phase_names):
        raise ValueError(f"Phase index {phase_idx} out of range 1..{len(phase_names)}")
    if "-" not in frame_range:
        raise ValueError(f"Expected frame range `start-end`, got: {raw!r}")
    start_s, end_s = frame_range.split("-", 1)
    start = max(0, min(num_frames - 1, int(start_s.strip())))
    end = max(0, min(num_frames - 1, int(end_s.strip())))
    if end < start:
        raise ValueError(f"Invalid range `{start}-{end}` in: {raw!r}")
    return {"name": str(phase_names[phase_idx - 1]), "start": start, "end": end}


def _validate_phase_ranges(phases: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    phases = list(phases)
    phases.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["name"])))
    prev_end = -1
    for phase in phases:
        start = int(phase["start"])
        if start <= prev_end:
            raise ValueError("Keyphase ranges must be non-overlapping and ordered.")
        prev_end = int(phase["end"])
    return phases


def _parse_keyphases(raw: str, *, num_frames: int) -> list[dict[str, int | str]]:
    text = raw.strip()
    if not text:
        return []
    phases = [_parse_phase_entry(part, num_frames=num_frames) for part in text.split(";") if part.strip()]
    return _validate_phase_ranges(phases)


def _parse_indexed_keyphases(
    raw: str,
    *,
    num_frames: int,
    phase_names: list[str],
) -> list[dict[str, int | str]]:
    text = raw.strip()
    if not text:
        return []
    phases = [
        _parse_indexed_phase_entry(part, num_frames=num_frames, phase_names=phase_names)
        for part in text.split(",")
        if part.strip()
    ]
    return _validate_phase_ranges(phases)


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive keyphase annotation tool for local LeRobot datasets.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--end_episode", type=int, default=-1)
    p.add_argument("--extract_images", action="store_true")
    p.add_argument("--image_key", type=str, default="observation.images.cam_high")
    p.add_argument("--images_out_dir", type=str, default=None)
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ensure_local_hf_cache()
    dataset_root = pathlib.Path(args.dataset_root).resolve()
    ep_rows, lengths = _load_episode_rows_from_meta(dataset_root)
    ds = LeRobotDataset(repo_id=args.repo_id, root=dataset_root)
    ep_ids = sorted(lengths.keys())
    if args.end_episode >= 0:
        ep_ids = [e for e in ep_ids if args.start_episode <= e <= args.end_episode]
    else:
        ep_ids = [e for e in ep_ids if e >= args.start_episode]

    out_path = pathlib.Path(args.out).resolve()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = {"dataset_root": str(dataset_root), "repo_id": args.repo_id, "episodes": {}}

    phase_names = result.get("phase_names")
    if phase_names is None:
        while True:
            raw_phase_names = input("phase names (comma-separated, e.g. pick, put)> ").strip()
            try:
                phase_names = _parse_phase_names(raw_phase_names)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"Invalid phase names: {exc}")
        result["phase_names"] = phase_names
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        phase_names = [str(x) for x in phase_names]

    phase_legend = ", ".join(f"{i + 1}:{name}" for i, name in enumerate(phase_names))
    print("Phase legend:", phase_legend)
    print("Enter keyphases as `1:40-50,2:70-80`. Empty = no keyphases. 'q' to quit.")
    for ep in ep_ids:
        if str(ep) in result["episodes"]:
            continue
        if args.extract_images:
            base = pathlib.Path(args.images_out_dir) if args.images_out_dir else (out_path.parent / "annotation_images")
            ep_dir = base / f"episode_{ep:06d}"
            _export_episode_images(ds, ep_rows.get(ep, []), image_key=args.image_key, out_dir=ep_dir)
            print(f"[episode {ep}] exported images -> {ep_dir}")
        num_frames = lengths[ep]
        while True:
            raw = input(f"[episode {ep}] num_frames={num_frames} keyphases> ").strip()
            if raw.lower() in {"q", "quit", "exit"}:
                print(f"Saved annotations: {out_path}")
                return
            try:
                if any(ch.isalpha() for ch in raw):
                    phases = _parse_keyphases(raw, num_frames=num_frames)
                else:
                    phases = _parse_indexed_keyphases(raw, num_frames=num_frames, phase_names=phase_names)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"Invalid input: {exc}")
        result["episodes"][str(ep)] = phases
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved annotations: {out_path}")


if __name__ == "__main__":
    main()
