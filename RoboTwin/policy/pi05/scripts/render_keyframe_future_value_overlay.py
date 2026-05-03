#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import pickle
import shutil
import subprocess
import time
from typing import Any

import cv2
import flax.nnx as nnx
import numpy as np

from openpi.training import config as train_config
from openpi_online_ppo.data.local_lerobot_loader import ensure_local_hf_cache
from openpi_online_ppo.models import pi0_aux as _rl_pi0
from openpi_online_ppo.models import pi0_fast_rl as _rl_pi0_fast
from openpi_online_ppo.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy
from openpi_online_ppo.rl.pi0_value_policy import create_pi0_value_policy

KEYFRAME_NUM_BINS = 16


def _resolve_policy_config_name(raw_name: str, policy_path: str) -> str:
    name = str(raw_name).strip()
    if name and name != "auto":
        return name
    policy_path_l = str(policy_path).lower()
    if "ee_delta" in policy_path_l:
        return "pi0_fast_aloha_robotwin_ppo_ee_delta"
    return "pi0_fast_aloha_robotwin_ppo"


def _to_rl_train_config(cfg: train_config.TrainConfig) -> train_config.TrainConfig:
    base = cfg.model
    if isinstance(base, _rl_pi0_fast.Pi0FASTConfig):
        rl_model = _rl_pi0_fast.Pi0FASTConfig(
            dtype=base.dtype,
            paligemma_variant=base.paligemma_variant,
            action_dim=base.action_dim,
            action_horizon=base.action_horizon,
            max_token_len=base.max_token_len,
            fast_model_tokenizer=base.fast_model_tokenizer,
            fast_model_tokenizer_kwargs=base.fast_model_tokenizer_kwargs,
            use_value_head=True,
            use_keyframe_head=True,
            keyframe_num_bins=KEYFRAME_NUM_BINS,
        )
        return dataclasses.replace(cfg, model=rl_model)
    if hasattr(base, "pi05") and hasattr(base, "action_dim") and hasattr(base, "action_horizon"):
        rl_model = _rl_pi0.Pi0AuxConfig(
            dtype=base.dtype,
            paligemma_variant=base.paligemma_variant,
            action_expert_variant=base.action_expert_variant,
            action_dim=base.action_dim,
            action_horizon=base.action_horizon,
            max_token_len=base.max_token_len,
            pi05=base.pi05,
            discrete_state_input=base.discrete_state_input,
            use_value_head=True,
            use_keyframe_head=True,
            keyframe_num_bins=KEYFRAME_NUM_BINS,
        )
        return dataclasses.replace(cfg, model=rl_model)
    raise TypeError(f"Config `{cfg.name}` is not compatible with value/keyframe overlay rendering.")


def _parse_episode_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _to_chw_uint8(x: Any) -> np.ndarray:
    arr = _as_numpy(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 image, got shape={arr.shape}")
    if arr.shape[0] == 3:
        out = arr
    elif arr.shape[-1] == 3:
        out = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Expected image with 3 channels, got shape={arr.shape}")
    if out.dtype != np.uint8:
        if np.issubdtype(out.dtype, np.floating) and float(np.nanmax(out)) <= 1.0:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _chw_to_hwc_uint8(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected CHW image, got shape={arr.shape}")
    return np.transpose(arr, (1, 2, 0)).astype(np.uint8, copy=False)


def _group_episode_indices(ds) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i in range(len(ds)):
        item = ds[i]
        ep = int(_as_numpy(item["episode_index"]).item())
        out.setdefault(ep, []).append(i)
    return out


def _group_selected_episode_indices_from_meta(dataset_root: pathlib.Path, *, episode_ids: list[int]) -> dict[int, list[int]] | None:
    if not episode_ids:
        return {}
    episodes_meta_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_meta_path.exists():
        return None

    lengths_by_episode: dict[int, int] = {}
    ordered_episodes: list[int] = []
    with episodes_meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec["episode_index"])
            length = int(rec["length"])
            if length < 0 or ep in lengths_by_episode:
                return None
            ordered_episodes.append(ep)
            lengths_by_episode[ep] = length

    selected_set = set(int(ep) for ep in episode_ids)
    if not selected_set.issubset(lengths_by_episode):
        return None

    grouped: dict[int, list[int]] = {}
    local_offset = 0
    for ep in ordered_episodes:
        length = lengths_by_episode[ep]
        if ep in selected_set:
            grouped[ep] = list(range(local_offset, local_offset + length))
            local_offset += length
    return grouped


def _load_keyphases(path: str | pathlib.Path) -> dict[int, list[dict[str, Any]]]:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
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
            raise ValueError(f"Episode {ep} keyphases must be a list.")
        spans: list[dict[str, Any]] = []
        prev_end = -1
        for item in spans_raw:
            if not isinstance(item, dict):
                raise ValueError(f"Episode {ep} phase item must be an object, got {type(item).__name__}.")
            name = str(item.get("name", "")).strip()
            start = int(item.get("start"))
            end = int(item.get("end"))
            if not name:
                raise ValueError(f"Episode {ep} phase name must be non-empty.")
            if end < start:
                raise ValueError(f"Episode {ep} phase `{name}` invalid range [{start}, {end}]")
            if start <= prev_end:
                raise ValueError(f"Episode {ep} phase `{name}` overlaps previous range.")
            spans.append({"name": name, "start": start, "end": end})
            prev_end = end
        parsed[ep] = spans
    return parsed


def _load_fps(dataset_root: pathlib.Path, default_fps: int) -> int:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return int(default_fps)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info.get("fps", default_fps))


def _load_value_state(policy, state_file: str) -> None:
    path = pathlib.Path(state_file).resolve()
    with path.open("rb") as f:
        payload = pickle.load(f)
    pure = payload["params_pure"] if isinstance(payload, dict) and "params_pure" in payload else payload
    model = policy.model
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(pure)
    policy.sync_model(nnx.merge(graphdef, state))


def _iter_batches(items: list[int], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _build_obs(item: dict[str, Any], *, default_prompt: str) -> dict[str, Any]:
    prompt = item.get("task")
    if prompt is None:
        prompt = default_prompt
    return {
        "state": _as_numpy(item["observation.state"]).astype(np.float32).reshape(-1),
        "images": {
            "cam_high": _to_chw_uint8(item["observation.images.cam_high"]),
            "cam_left_wrist": _to_chw_uint8(item["observation.images.cam_left_wrist"]),
            "cam_right_wrist": _to_chw_uint8(item["observation.images.cam_right_wrist"]),
        },
        "prompt": str(prompt),
    }


class _FFmpegVideoWriter:
    def __init__(self, proc: subprocess.Popen[bytes], *, out_path: pathlib.Path) -> None:
        self._proc = proc
        self._out_path = out_path
        if proc.stdin is None:
            raise RuntimeError("ffmpeg writer requires stdin pipe.")

    def isOpened(self) -> bool:
        return self._proc.poll() is None and self._proc.stdin is not None and not self._proc.stdin.closed

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            raise RuntimeError(f"ffmpeg writer for {self._out_path} is not open.")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read().decode("utf-8", errors="replace") if self._proc.stderr is not None else ""
        ret = self._proc.wait()
        if self._proc.stderr is not None:
            self._proc.stderr.close()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self._out_path}:\n{stderr.strip()}")


def _open_video_writer(out_path: pathlib.Path, *, fps: int, frame_size: tuple[int, int]):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        width, height = frame_size
        cmd = [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(float(fps)), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.stdin is not None:
            return _FFmpegVideoWriter(proc, out_path=out_path)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")
    return writer


def _compute_axis_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        pad = max(0.5, abs(vmin) * 0.1 + 0.1)
        return vmin - pad, vmax + pad
    pad = 0.08 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _draw_phase_curve_chart(
    values: np.ndarray,
    *,
    phase_start: int,
    phase_end: int,
    current_index: int,
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    x_pad = 40
    y_pad_top = 20
    y_pad_bottom = 28
    plot_w = max(1, width - x_pad - 12)
    plot_h = max(1, height - y_pad_top - y_pad_bottom)
    x0, y0 = x_pad, y_pad_top
    x1, y1 = x0 + plot_w, y0 + plot_h
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 220, 220), 1)

    phase_start = int(max(0, phase_start))
    phase_end = int(min(len(values) - 1, phase_end))
    if phase_end < phase_start:
        return canvas
    segment = np.asarray(values[phase_start : phase_end + 1], dtype=np.float32)
    if segment.size == 0:
        return canvas
    y_min, y_max = _compute_axis_range(segment)

    def _map(i: int, v: float) -> tuple[int, int]:
        xx = int(round(x0 + plot_w * (i / max(1, len(segment) - 1))))
        ratio = float(np.clip((v - y_min) / max(1e-6, y_max - y_min), 0.0, 1.0))
        yy = int(round(y1 - plot_h * ratio))
        return xx, yy

    pts = np.asarray([_map(i, float(v)) for i, v in enumerate(segment)], dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], False, (60, 120, 220), 2, lineType=cv2.LINE_AA)
    local_cur = int(np.clip(current_index - phase_start, 0, len(segment) - 1))
    cv2.circle(canvas, tuple(pts[local_cur, 0]), 4, (30, 60, 220), -1, lineType=cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (10, 16), font, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_max:.3f}", (4, y0 + 4), font, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_min:.3f}", (4, y1), font, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"phase frames {phase_start}-{phase_end} cur={current_index} value={float(values[current_index]):.3f}",
        (10, height - 8),
        font,
        0.45,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _overlay_text(frame: np.ndarray, lines: list[str], *, color: tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    x, y = 14, 24
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line in lines:
        cv2.putText(out, line, (x, y), font, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), font, 0.6, color, 1, cv2.LINE_AA)
        y += 24
    return out


def _render_episode_video(
    *,
    frames_hwc: list[np.ndarray],
    values: np.ndarray,
    phase_spans: list[dict[str, Any]],
    frame_indices: list[int],
    fps: int,
    out_path: pathlib.Path,
    panel_height: int,
) -> list[dict[str, Any]]:
    height, width = frames_hwc[0].shape[:2]
    writer = _open_video_writer(out_path, fps=fps, frame_size=(width, height + panel_height))
    events: list[dict[str, Any]] = []
    try:
        for i, frame in enumerate(frames_hwc):
            frame_idx = int(frame_indices[i])
            active_phase = None
            for span in phase_spans:
                if int(span["start"]) <= frame_idx <= int(span["end"]):
                    active_phase = span
                    break
            is_key = active_phase is not None
            lines = [
                f"frame={i} dataset_frame={frame_idx}",
                f"is_keyphase={int(is_key)}",
                f"value={float(values[i]):.3f}",
            ]
            if active_phase is not None:
                lines.append(f"phase={active_phase['name']}")
            color = (30, 150, 50) if is_key else (80, 80, 80)
            frame_anno = _overlay_text(frame, lines, color=color)
            if is_key:
                chart = _draw_phase_curve_chart(
                    values,
                    phase_start=int(active_phase["start"]),
                    phase_end=int(active_phase["end"]),
                    current_index=int(frame_idx),
                    width=width,
                    height=panel_height,
                    title=f"phase value curve: {active_phase['name']}",
                )
                events.append(
                    {
                        "frame_index": int(i),
                        "dataset_frame_index": frame_idx,
                        "phase_name": str(active_phase["name"]),
                        "phase_start": int(active_phase["start"]),
                        "phase_end": int(active_phase["end"]),
                        "value": float(values[i]),
                    }
                )
            else:
                chart = np.full((panel_height, width, 3), 248, dtype=np.uint8)
                cv2.putText(
                    chart,
                    "not in annotated keyphase",
                    (12, max(24, panel_height // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 100),
                    1,
                    cv2.LINE_AA,
                )
            stacked = np.concatenate([frame_anno, chart], axis=0)
            writer.write(stacked[:, :, ::-1])
    finally:
        writer.release()
    return events


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render video overlay: annotated keyphase gate + phase-local value curve.")
    p.add_argument("--policy.path", dest="policy_path", type=str, default="")
    p.add_argument("--policy.config", dest="policy_config", type=str, default="auto")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--annotations_json", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--episodes", type=str, default="", help="Comma-separated episode indices. Empty means all.")
    p.add_argument("--value_state_file", type=str, default=None, help="Optional latest.pkl from serve_value_mc_ws.")
    p.add_argument("--fps", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--panel_height", type=int, default=140)
    p.add_argument("--default_prompt", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ensure_local_hf_cache()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    policy_config_name = _resolve_policy_config_name(args.policy_config, args.policy_path)
    cfg = _to_rl_train_config(train_config.get_config(policy_config_name))
    if isinstance(cfg.model, _rl_pi0_fast.Pi0FASTConfig):
        if not args.policy_path:
            raise ValueError("Pi0FAST overlay rendering requires --policy.path.")
        policy = create_trained_pi0_fast_rl_policy(cfg, args.policy_path, default_prompt=args.default_prompt)
    elif isinstance(cfg.model, _rl_pi0.Pi0AuxConfig):
        policy = create_pi0_value_policy(cfg, args.policy_path or None, default_prompt=args.default_prompt)
    else:
        raise TypeError(f"Unsupported model type for overlay rendering: {type(cfg.model).__name__}")
    if args.value_state_file:
        _load_value_state(policy, args.value_state_file)

    dataset_root = pathlib.Path(args.dataset_root).resolve()
    selected = _parse_episode_list(args.episodes)
    fps = int(args.fps) if int(args.fps) > 0 else _load_fps(dataset_root, 50)
    keyphases_by_ep = _load_keyphases(args.annotations_json)
    ds = LeRobotDataset(repo_id=args.repo_id, root=dataset_root, episodes=selected)
    grouped = _group_selected_episode_indices_from_meta(dataset_root, episode_ids=selected) if selected is not None else None
    if grouped is None:
        grouped = _group_episode_indices(ds)
    episode_ids = selected if selected is not None else sorted(grouped.keys())

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "repo_id": str(args.repo_id),
        "policy_config": policy_config_name,
        "policy_path": str(args.policy_path) if args.policy_path else None,
        "value_state_file": str(args.value_state_file) if args.value_state_file else None,
        "annotations_json": str(pathlib.Path(args.annotations_json).resolve()),
        "episodes": {},
    }

    for ep in episode_ids:
        ep_indices = grouped[ep]
        frames_hwc: list[np.ndarray] = []
        values: list[float] = []
        keyframe_probs: list[float] = []
        frame_indices: list[int] = []
        start_time = time.time()
        for _, batch_indices in _iter_batches(ep_indices, int(args.batch_size)):
            batch_obs: list[dict[str, Any]] = []
            batch_frames: list[np.ndarray] = []
            batch_fis: list[int] = []
            for ds_idx in batch_indices:
                item = ds[ds_idx]
                obs = _build_obs(item, default_prompt=args.default_prompt)
                batch_obs.append(obs)
                batch_frames.append(_chw_to_hwc_uint8(obs["images"]["cam_high"]))
                batch_fis.append(int(_as_numpy(item["frame_index"]).item()))
            values.extend(policy.predict_value_batch(batch_obs).tolist())
            keyframe_probs.extend(policy.predict_keyframe_prob_batch(batch_obs).tolist())
            frames_hwc.extend(batch_frames)
            frame_indices.extend(batch_fis)

        values_arr = np.asarray(values, dtype=np.float32)
        out_video = out_dir / f"episode_{ep:06d}_future_value_overlay.mp4"
        phase_spans = keyphases_by_ep.get(int(ep), [])
        events = _render_episode_video(
            frames_hwc=frames_hwc,
            values=values_arr,
            phase_spans=phase_spans,
            frame_indices=frame_indices,
            fps=int(fps),
            out_path=out_video,
            panel_height=int(args.panel_height),
        )
        summary["episodes"][str(ep)] = {
            "num_frames": int(len(frames_hwc)),
            "frame_indices": frame_indices,
            "video_path": str(out_video),
            "num_keyphase_frames": int(len(events)),
            "keyphase_events": events,
            "elapsed_sec": round(float(time.time() - start_time), 2),
        }
        print(json.dumps({"episode": int(ep), "video_path": str(out_video), "num_keyphase_frames": len(events)}, ensure_ascii=False), flush=True)

    summary_path = out_dir / "future_value_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "summary_path": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
