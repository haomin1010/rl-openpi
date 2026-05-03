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
from openpi_online_ppo.rl.pi0_value_policy import create_pi0_value_policy
from openpi_online_ppo.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy

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


def _group_selected_episode_indices_from_meta(
    dataset_root: pathlib.Path,
    *,
    episode_ids: list[int],
) -> dict[int, list[int]] | None:
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
    offset = 0
    for ep in ordered_episodes:
        if ep not in selected_set:
            continue
        length = lengths_by_episode[ep]
        grouped[ep] = list(range(offset, offset + length))
        offset += length
    return grouped


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
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
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


def _draw_series_chart(
    values: np.ndarray,
    *,
    width: int,
    height: int,
    current_index: int,
    y_min: float,
    y_max: float,
    title: str,
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
    value_label: str,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    margin_left = 56
    margin_right = 20
    margin_top = 18
    margin_bottom = 34
    plot_w = max(1, width - margin_left - margin_right)
    plot_h = max(1, height - margin_top - margin_bottom)
    x0 = margin_left
    y0 = margin_top
    x1 = x0 + plot_w
    y1 = y0 + plot_h

    cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 220, 220), 1)
    for frac in (0.25, 0.5, 0.75):
        yy = int(round(y0 + plot_h * frac))
        cv2.line(canvas, (x0, yy), (x1, yy), (230, 230, 230), 1)

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -1.0, 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def _map_point(i: int, v: float) -> tuple[int, int]:
        if len(values) <= 1:
            xx = x0
        else:
            xx = int(round(x0 + plot_w * (i / (len(values) - 1))))
        vv = float(np.clip((v - y_min) / (y_max - y_min), 0.0, 1.0))
        yy = int(round(y1 - plot_h * vv))
        return xx, yy

    pts = np.asarray([_map_point(i, float(v)) for i, v in enumerate(values)], dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], False, line_color, 2, lineType=cv2.LINE_AA)
    elif len(pts) == 1:
        cv2.circle(canvas, tuple(pts[0, 0]), 2, line_color, -1, lineType=cv2.LINE_AA)

    current_index = int(np.clip(current_index, 0, len(values) - 1))
    cur_pt = _map_point(current_index, float(values[current_index]))
    cv2.line(canvas, (cur_pt[0], y0), (cur_pt[0], y1), (200, 200, 200), 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, cur_pt, 4, point_color, -1, lineType=cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (12, 18), font, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_max:.3f}", (6, y0 + 5), font, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_min:.3f}", (6, y1), font, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"frame={current_index} {value_label}={float(values[current_index]):.4f}",
        (12, height - 10),
        font,
        0.5,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _compute_axis_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        pad = max(0.5, abs(vmin) * 0.1 + 0.1)
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.08
    return vmin - pad, vmax + pad


def _open_video_writer(out_path: pathlib.Path, *, fps: int, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        width, height = frame_size
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if proc.stdin is not None:
            return _FFmpegVideoWriter(proc, out_path=out_path)

    codec_candidates = ("avc1", "H264", "mp4v")
    last_error: str | None = None
    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()
        last_error = codec
    raise RuntimeError(
        f"Failed to open video writer for {out_path} with codecs={codec_candidates}. "
        f"Last attempted codec={last_error}."
    )


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


def _render_episode_video(
    *,
    episode_index: int,
    frames_hwc: list[np.ndarray],
    values: np.ndarray | None,
    keyframe_probs: np.ndarray | None,
    fps: int,
    out_path: pathlib.Path,
    chart_height: int,
    show_value: bool,
    show_keyframe: bool,
) -> None:
    if not frames_hwc:
        raise ValueError(f"Episode {episode_index} has no frames.")
    if not show_value and not show_keyframe:
        raise ValueError("At least one curve must be enabled.")
    height, width = frames_hwc[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_video_writer(
        out_path,
        fps=fps,
        frame_size=(width, height + chart_height),
    )
    chart_panels = int(show_value) + int(show_keyframe)
    panel_heights = [chart_height // chart_panels] * chart_panels
    panel_heights[-1] += chart_height - sum(panel_heights)
    try:
        for i, frame in enumerate(frames_hwc):
            charts: list[np.ndarray] = []
            panel_idx = 0
            if show_value:
                assert values is not None
                y_min, y_max = _compute_axis_range(values)
                charts.append(
                    _draw_series_chart(
                        values,
                        width=width,
                        height=max(80, panel_heights[panel_idx]),
                        current_index=i,
                        y_min=y_min,
                        y_max=y_max,
                        title="value",
                        line_color=(70, 120, 220),
                        point_color=(30, 60, 220),
                        value_label="value",
                    )
                )
                panel_idx += 1
            if show_keyframe:
                assert keyframe_probs is not None
                prob_min, prob_max = _compute_axis_range(keyframe_probs)
                charts.append(
                    _draw_series_chart(
                        keyframe_probs,
                        width=width,
                        height=max(80, panel_heights[panel_idx]),
                        current_index=i,
                        y_min=prob_min,
                        y_max=prob_max,
                        title="keyframe_prob",
                        line_color=(70, 170, 90),
                        point_color=(20, 110, 40),
                        value_label="p",
                    )
                )
            chart = charts[0] if len(charts) == 1 else np.concatenate(charts, axis=0)
            stacked = np.concatenate([frame, chart], axis=0)
            writer.write(stacked[:, :, ::-1])
    finally:
        writer.release()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render LeRobot episode videos with value-head overlay curves.")
    p.add_argument("--policy.path", dest="policy_path", type=str, default="")
    p.add_argument("--policy.config", dest="policy_config", type=str, default="auto")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--episodes", type=str, default="", help="Comma-separated episode indices. Empty means all.")
    p.add_argument("--value_state_file", type=str, default=None, help="Optional latest.pkl from serve_value_mc_ws.")
    p.add_argument("--fps", type=int, default=0, help="Override fps. Default reads dataset meta/info.json.")
    p.add_argument("--chart_height", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=8, help="Frames per inference batch.")
    p.add_argument("--curves", type=str, default="both", choices=("both", "value", "keyframe"))
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
    ds = LeRobotDataset(repo_id=args.repo_id, root=dataset_root, episodes=selected)
    grouped = (
        _group_selected_episode_indices_from_meta(dataset_root, episode_ids=selected)
        if selected is not None
        else None
    )
    if grouped is None:
        grouped = _group_episode_indices(ds)
    episode_ids = selected if selected is not None else sorted(grouped.keys())
    show_value = args.curves in ("both", "value")
    show_keyframe = args.curves in ("both", "keyframe")

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "repo_id": str(args.repo_id),
        "policy_path": str(args.policy_path) if args.policy_path else None,
        "policy_config": policy_config_name,
        "value_state_file": str(args.value_state_file) if args.value_state_file else None,
        "batch_size": int(args.batch_size),
        "curves": str(args.curves),
        "episodes": {},
    }

    for ep in episode_ids:
        if ep not in grouped:
            raise KeyError(f"Episode {ep} not found in dataset.")
        frames_hwc: list[np.ndarray] = []
        values: list[float] = []
        keyframe_probs: list[float] = []
        frame_indices: list[int] = []
        ep_indices = grouped[ep]
        ep_start_time = time.time()
        print(
            json.dumps(
                {
                    "episode": int(ep),
                    "stage": "infer_start",
                    "num_frames": int(len(ep_indices)),
                    "batch_size": int(args.batch_size),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for batch_start, batch_indices in _iter_batches(ep_indices, int(args.batch_size)):
            batch_obs: list[dict[str, Any]] = []
            batch_frame_indices: list[int] = []
            batch_frames_hwc: list[np.ndarray] = []
            for ds_idx in batch_indices:
                item = ds[ds_idx]
                obs = _build_obs(item, default_prompt=args.default_prompt)
                batch_obs.append(obs)
                batch_frame_indices.append(int(_as_numpy(item["frame_index"]).item()))
                batch_frames_hwc.append(_chw_to_hwc_uint8(obs["images"]["cam_high"]))
            if show_value:
                values.extend(policy.predict_value_batch(batch_obs).tolist())
            if show_keyframe:
                keyframe_probs.extend(policy.predict_keyframe_prob_batch(batch_obs).tolist())
            frame_indices.extend(batch_frame_indices)
            frames_hwc.extend(batch_frames_hwc)
            done = batch_start + len(batch_indices)
            elapsed = time.time() - ep_start_time
            print(
                json.dumps(
                    {
                        "episode": int(ep),
                        "stage": "infer_progress",
                        "done_frames": int(done),
                        "num_frames": int(len(ep_indices)),
                        "elapsed_sec": round(float(elapsed), 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        values_arr = np.asarray(values, dtype=np.float32) if show_value else None
        keyframe_probs_arr = np.asarray(keyframe_probs, dtype=np.float32) if show_keyframe else None
        out_video = out_dir / f"episode_{ep:06d}_value_overlay.mp4"
        print(
            json.dumps(
                {
                    "episode": int(ep),
                    "stage": "render_start",
                    "video_path": str(out_video),
                    "num_frames": int(len(frames_hwc)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        _render_episode_video(
            episode_index=ep,
            frames_hwc=frames_hwc,
            values=values_arr,
            keyframe_probs=keyframe_probs_arr,
            fps=fps,
            out_path=out_video,
            chart_height=int(args.chart_height),
            show_value=show_value,
            show_keyframe=show_keyframe,
        )
        summary["episodes"][str(ep)] = {
            "num_frames": int(len(frames_hwc)),
            "frame_indices": frame_indices,
            "values": [float(v) for v in values_arr.tolist()] if values_arr is not None else None,
            "keyframe_probs": [float(v) for v in keyframe_probs_arr.tolist()] if keyframe_probs_arr is not None else None,
            "video_path": str(out_video),
            "elapsed_sec": round(float(time.time() - ep_start_time), 2),
        }
        print(
            json.dumps(
                {
                    "episode": int(ep),
                    "stage": "done",
                    "video_path": str(out_video),
                    "num_frames": int(len(frames_hwc)),
                    "elapsed_sec": round(float(time.time() - ep_start_time), 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary_path = out_dir / "value_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "summary_path": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
