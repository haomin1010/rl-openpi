#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames
import yaml

from openpi_client import msgpack_numpy


def _resolve_repo_root(repo_root: str | None) -> pathlib.Path:
    if repo_root:
        return pathlib.Path(repo_root).resolve()
    # scripts/ -> pi05/ -> policy/ -> RoboTwin/
    return pathlib.Path(__file__).resolve().parents[3]


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data)}")
    return data


def _get_embodiment_config(robot_file: str) -> dict[str, Any]:
    cfg_path = pathlib.Path(robot_file) / "config.yml"
    return _load_yaml(cfg_path)


def _build_task_args(repo_root: pathlib.Path, task_name: str, task_config: str) -> dict[str, Any]:
    config_root = repo_root / "task_config"
    task_cfg = _load_yaml(config_root / f"{task_config}.yml")
    task_cfg["task_name"] = task_name
    task_cfg["task_config"] = task_config

    embodiment_cfg = _load_yaml(config_root / "_embodiment_config.yml")
    camera_cfg = _load_yaml(config_root / "_camera_config.yml")

    embodiment_type = task_cfg.get("embodiment")
    if not isinstance(embodiment_type, list) or len(embodiment_type) not in (1, 3):
        raise ValueError(f"Unexpected embodiment format: {embodiment_type}")

    def _embodiment_file(name: str) -> str:
        file_path = embodiment_cfg[name]["file_path"]
        if file_path is None:
            raise ValueError(f"No embodiment file path configured for: {name}")
        return file_path

    if len(embodiment_type) == 1:
        task_cfg["left_robot_file"] = _embodiment_file(embodiment_type[0])
        task_cfg["right_robot_file"] = _embodiment_file(embodiment_type[0])
        task_cfg["dual_arm_embodied"] = True
    else:
        task_cfg["left_robot_file"] = _embodiment_file(embodiment_type[0])
        task_cfg["right_robot_file"] = _embodiment_file(embodiment_type[1])
        task_cfg["embodiment_dis"] = embodiment_type[2]
        task_cfg["dual_arm_embodied"] = False

    task_cfg["left_embodiment_config"] = _get_embodiment_config(task_cfg["left_robot_file"])
    task_cfg["right_embodiment_config"] = _get_embodiment_config(task_cfg["right_robot_file"])

    head_camera_type = task_cfg["camera"]["head_camera_type"]
    task_cfg["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    task_cfg["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    # Keep headless and lightweight by default for online server use.
    task_cfg["eval_mode"] = True
    task_cfg["render_freq"] = 0
    task_cfg["eval_video_log"] = False
    task_cfg["save_data"] = False

    return task_cfg


def _instantiate_task(repo_root: pathlib.Path, task_name: str):
    import sys

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    policy_root = str(repo_root / "policy")
    if policy_root not in sys.path:
        sys.path.insert(0, policy_root)
    desc_utils_root = str(repo_root / "description" / "utils")
    if desc_utils_root not in sys.path:
        sys.path.insert(0, desc_utils_root)

    module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(module, task_name)
    return env_class()


def _maybe_import_instruction_generator():
    from generate_episode_instructions import generate_episode_descriptions

    return generate_episode_descriptions


def _to_chw_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected image rank=3, got shape={arr.shape}")
    if arr.shape[-1] == 3:
        return np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 3:
        return arr
    raise ValueError(f"Unsupported image shape for CHW conversion: {arr.shape}")


@dataclass
class EpisodeState:
    seed: int
    task_name: str
    prompt: str


class LeRobotEnvDatasetWriter:
    """Write per-step env transitions to a LeRobot dataset compatible with Aloha format."""

    def __init__(
        self,
        *,
        root: pathlib.Path,
        repo_id: str,
        fps: int = 50,
        robot_type: str = "aloha",
        overwrite: bool = False,
        resize_to_640x480: bool = True,
    ) -> None:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self._root = root.resolve()
        self._repo_id = str(repo_id)
        self._fps = int(fps)
        self._resize_to_640x480 = bool(resize_to_640x480)
        self._episode_open = False
        self._episode_has_frames = False
        self._episode_frame_idx = 0
        self._saved_episode_count = 0
        if overwrite and self._root.exists():
            shutil.rmtree(self._root)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": [[
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                ]],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": [[
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper",
                ]],
            },
            "observation.images.cam_high": {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "image",
                "shape": (3, 480, 640),
                "names": ["channels", "height", "width"],
            },
        }
        self._dataset = LeRobotDataset.create(
            repo_id=self._repo_id,
            root=self._root,
            fps=self._fps,
            robot_type=robot_type,
            features=features,
            use_videos=False,
        )
        self._outcome_log_path = self._root / "meta" / "online_episode_outcomes.jsonl"

    @property
    def root(self) -> pathlib.Path:
        return self._root

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def saved_episode_count(self) -> int:
        return self._saved_episode_count

    @property
    def has_open_frames(self) -> bool:
        return bool(self._episode_open and self._episode_has_frames)

    def start_episode(self) -> None:
        self._episode_open = True
        self._episode_has_frames = False
        self._episode_frame_idx = 0

    def add_step(self, *, raw_obs: dict[str, Any], action: np.ndarray, task: str) -> None:
        if not self._episode_open:
            self.start_episode()
        cam_obs = raw_obs["observation"]
        head_rgb = np.asarray(cam_obs["head_camera"]["rgb"])
        left_rgb = np.asarray(cam_obs["left_camera"]["rgb"])
        right_rgb = np.asarray(cam_obs["right_camera"]["rgb"])
        if self._resize_to_640x480:
            head_rgb = cv2.resize(head_rgb, (640, 480))
            left_rgb = cv2.resize(left_rgb, (640, 480))
            right_rgb = cv2.resize(right_rgb, (640, 480))
        frame = {
            "observation.state": np.asarray(raw_obs["joint_action"]["vector"], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32).reshape(-1),
            "observation.images.cam_high": _to_chw_uint8(head_rgb),
            "observation.images.cam_left_wrist": _to_chw_uint8(left_rgb),
            "observation.images.cam_right_wrist": _to_chw_uint8(right_rgb),
            "task": str(task),
        }
        self._dataset.add_frame(frame)
        self._episode_has_frames = True
        self._episode_frame_idx += 1

    def end_episode(
        self,
        *,
        commit: bool = True,
        success: bool | None = None,
        step_lim: int | None = None,
        seed: int | None = None,
        prompt: str | None = None,
    ) -> None:
        if not self._episode_open:
            return
        if commit and self._episode_has_frames:
            self._dataset.save_episode()
            if success is not None:
                self._outcome_log_path.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "episode_index": int(self._saved_episode_count),
                    "success": bool(success),
                    "num_frames": int(self._episode_frame_idx),
                    "step_lim": int(step_lim) if step_lim is not None else None,
                    "seed": int(seed) if seed is not None else None,
                    "prompt": str(prompt) if prompt is not None else "",
                }
                with self._outcome_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._saved_episode_count += 1
        self._episode_open = False
        self._episode_has_frames = False
        self._episode_frame_idx = 0


class LeRobotRoundDatasetManager:
    """Manage per-round LeRobot datasets under a shared base directory."""

    def __init__(
        self,
        *,
        base_root: pathlib.Path,
        base_repo_id: str,
        fps: int = 50,
        overwrite: bool = False,
        resize_to_640x480: bool = True,
    ) -> None:
        self._base_root = base_root.resolve()
        self._base_repo_id = str(base_repo_id)
        self._fps = int(fps)
        self._resize_to_640x480 = bool(resize_to_640x480)
        if overwrite and self._base_root.exists():
            shutil.rmtree(self._base_root)
        self._base_root.mkdir(parents=True, exist_ok=True)
        self._round_manifest = self._base_root / "rounds.jsonl"
        self._current_writer: LeRobotEnvDatasetWriter | None = None
        self._next_round_index = self._discover_next_round_index()

    @property
    def base_root(self) -> pathlib.Path:
        return self._base_root

    def _discover_next_round_index(self) -> int:
        max_idx = -1
        for child in self._base_root.glob("round_*"):
            if not child.is_dir():
                continue
            try:
                max_idx = max(max_idx, int(child.name.split("_")[-1]))
            except Exception:
                continue
        return max_idx + 1

    def _round_root(self, round_index: int) -> pathlib.Path:
        return self._base_root / f"round_{int(round_index):06d}"

    def _round_repo_id(self, round_index: int) -> str:
        return f"{self._base_repo_id}_round_{int(round_index):06d}"

    def _ensure_writer(self) -> LeRobotEnvDatasetWriter:
        if self._current_writer is None:
            round_index = int(self._next_round_index)
            self._current_writer = LeRobotEnvDatasetWriter(
                root=self._round_root(round_index),
                repo_id=self._round_repo_id(round_index),
                fps=self._fps,
                overwrite=True,
                resize_to_640x480=self._resize_to_640x480,
            )
        return self._current_writer

    def start_episode(self) -> None:
        self._ensure_writer().start_episode()

    def add_step(self, *, raw_obs: dict[str, Any], action: np.ndarray, task: str) -> None:
        self._ensure_writer().add_step(raw_obs=raw_obs, action=action, task=task)

    def end_episode(
        self,
        *,
        commit: bool = True,
        success: bool | None = None,
        step_lim: int | None = None,
        seed: int | None = None,
        prompt: str | None = None,
    ) -> None:
        if self._current_writer is None:
            return
        self._current_writer.end_episode(
            commit=commit,
            success=success,
            step_lim=step_lim,
            seed=seed,
            prompt=prompt,
        )

    def finalize_round(self) -> dict[str, Any]:
        if self._current_writer is None:
            return {
                "ok": True,
                "round_index": None,
                "dataset_root": None,
                "repo_id": None,
                "num_episodes": 0,
            }

        writer = self._current_writer
        if writer.has_open_frames:
            writer.end_episode(commit=True, success=None, step_lim=None, seed=None, prompt=None)
        else:
            writer.end_episode(commit=False)
        round_index = int(self._next_round_index)
        num_episodes = int(writer.saved_episode_count)
        dataset_root = writer.root
        repo_id = writer.repo_id

        if num_episodes <= 0:
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            result = {
                "ok": True,
                "round_index": round_index,
                "dataset_root": None,
                "repo_id": repo_id,
                "num_episodes": 0,
            }
        else:
            rec = {
                "round_index": round_index,
                "dataset_root": str(dataset_root),
                "repo_id": str(repo_id),
                "num_episodes": num_episodes,
            }
            self._round_manifest.parent.mkdir(parents=True, exist_ok=True)
            with self._round_manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            result = {"ok": True, **rec}

        self._current_writer = None
        self._next_round_index += 1
        return result

    def close(self) -> None:
        if self._current_writer is not None:
            self._current_writer.end_episode(commit=False)


class RoboTwinEnvSession:
    def __init__(
        self,
        *,
        repo_root: pathlib.Path,
        task_name: str,
        task_config: str,
        default_prompt: str | None,
        seed_start: int,
        instruction_type: str,
        prompt_max_descriptions: int,
        prompt_seed_max_tries: int,
        save_video: bool,
        video_save_dir: str | None,
        save_lerobot: bool,
        lerobot_root: str | None,
        lerobot_repo_id: str,
        lerobot_fps: int,
        lerobot_overwrite: bool,
        lerobot_resize_to_640x480: bool,
    ) -> None:
        self._repo_root = repo_root
        self._task_name = task_name
        self._task_config = task_config
        self._default_prompt = default_prompt
        self._next_seed = int(seed_start)
        self._episode_id = 0
        self._instruction_type = instruction_type
        self._prompt_max_descriptions = int(prompt_max_descriptions)
        self._prompt_seed_max_tries = int(prompt_seed_max_tries)
        self._save_video = bool(save_video)
        self._video_save_dir = str(video_save_dir) if video_save_dir else None
        self._ffmpeg_started = False
        self._lerobot_manager: LeRobotRoundDatasetManager | None = None
        self._latest_raw_obs: dict[str, Any] | None = None

        self._task_env = _instantiate_task(repo_root, task_name)
        self._task_args = _build_task_args(repo_root, task_name, task_config)
        if self._save_video:
            base_dir = pathlib.Path(self._video_save_dir) if self._video_save_dir else (repo_root / "eval_result" / "online_ws_video")
            base_dir.mkdir(parents=True, exist_ok=True)
            self._task_args["eval_video_save_dir"] = str(base_dir)
            self._task_args["eval_video_log"] = True
        if save_lerobot:
            lerobot_base = pathlib.Path(lerobot_root) if lerobot_root else (repo_root / "eval_result" / "online_ws_lerobot")
            self._lerobot_manager = LeRobotRoundDatasetManager(
                base_root=lerobot_base,
                base_repo_id=str(lerobot_repo_id),
                fps=int(lerobot_fps),
                overwrite=bool(lerobot_overwrite),
                resize_to_640x480=bool(lerobot_resize_to_640x480),
            )
        self._state: EpisodeState | None = None

    @property
    def task_name(self) -> str:
        return self._task_name

    def close(self) -> None:
        self._stop_episode_recording()
        if self._lerobot_manager is not None:
            self._lerobot_manager.close()
        try:
            self._task_env.close_env(clear_cache=False)
        except Exception:
            pass

    def _convert_observation(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        cam_obs = raw_obs["observation"]
        images = {
            "cam_high": _to_chw_uint8(cam_obs["head_camera"]["rgb"]),
            "cam_left_wrist": _to_chw_uint8(cam_obs["left_camera"]["rgb"]),
            "cam_right_wrist": _to_chw_uint8(cam_obs["right_camera"]["rgb"]),
        }
        state = np.asarray(raw_obs["joint_action"]["vector"], dtype=np.float32)
        prompt = self._state.prompt if self._state is not None else (self._default_prompt or self._task_name)
        return {
            "images": images,
            "state": state,
            "prompt": prompt,
        }

    def _safe_close_env(self) -> None:
        self._stop_episode_recording()
        if self._lerobot_manager is not None:
            self._lerobot_manager.close()
        try:
            self._task_env.close_env(clear_cache=False)
        except Exception:
            pass

    def _start_episode_recording(self) -> None:
        if not self._save_video:
            return
        if getattr(self._task_env, "eval_video_path", None) is None:
            return
        if self._ffmpeg_started:
            return

        width = int(self._task_args.get("head_camera_w", 640))
        height = int(self._task_args.get("head_camera_h", 480))
        episode_idx = int(getattr(self._task_env, "ep_num", 0))
        out_dir = pathlib.Path(self._task_env.eval_video_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode{episode_idx}.mp4"
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                "20",
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
        )
        self._task_env._set_eval_video_ffmpeg(ffmpeg)
        self._ffmpeg_started = True

    def _stop_episode_recording(self) -> None:
        if not self._ffmpeg_started:
            return
        try:
            self._task_env._del_eval_video_ffmpeg()
        except Exception:
            pass
        self._ffmpeg_started = False

    def _write_video_frame_from_obs(self, raw_obs: dict[str, Any]) -> None:
        if not self._ffmpeg_started:
            return
        try:
            frame = np.asarray(raw_obs["observation"]["head_camera"]["rgb"], dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                return
            self._task_env.eval_video_ffmpeg.stdin.write(frame.tobytes())
        except Exception:
            # Video logging should never break env serving.
            pass

    def _generate_prompt_for_seed(self, seed: int) -> str:
        from envs.utils.create_actor import UnStableError

        generate_episode_descriptions = _maybe_import_instruction_generator()

        try:
            self._task_env.setup_demo(
                now_ep_num=self._episode_id,
                seed=seed,
                is_test=True,
                **self._task_args,
            )
            episode_info = self._task_env.play_once()
            expert_ok = bool(self._task_env.plan_success and self._task_env.check_success())
            if not expert_ok:
                raise RuntimeError("Expert rollout failed for prompt generation.")
            info_dict = dict(episode_info.get("info", {}))
            generated = generate_episode_descriptions(
                self._task_name,
                [info_dict],
                max_descriptions=self._prompt_max_descriptions,
            )
            if not generated:
                raise RuntimeError("No prompt candidates generated from episode info.")
            candidates = list(generated[0].get(self._instruction_type, []))
            if not candidates:
                fallback_type = "seen" if self._instruction_type != "seen" else "unseen"
                candidates = list(generated[0].get(fallback_type, []))
            if not candidates:
                raise RuntimeError("Prompt candidate list is empty.")
            return str(np.random.choice(candidates))
        except UnStableError:
            raise
        finally:
            self._safe_close_env()

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        start_seed = int(payload.get("seed", self._next_seed))
        user_prompt = payload.get("prompt")

        if user_prompt is not None:
            seed = start_seed
            prompt = str(user_prompt)
        else:
            seed = start_seed
            prompt = None
            last_error: Exception | None = None
            for i in range(self._prompt_seed_max_tries):
                cand_seed = start_seed + i
                try:
                    prompt = self._generate_prompt_for_seed(cand_seed)
                    seed = cand_seed
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
            if prompt is None:
                if self._default_prompt is not None:
                    prompt = str(self._default_prompt)
                else:
                    raise RuntimeError(
                        f"Failed to generate prompt within {self._prompt_seed_max_tries} seed attempts "
                        f"starting at seed={start_seed}. Last error: {last_error}"
                    )

        self._next_seed = seed + 1
        self._safe_close_env()

        self._task_env.setup_demo(
            now_ep_num=self._episode_id,
            seed=seed,
            is_test=True,
            **self._task_args,
        )
        self._start_episode_recording()
        self._task_env.set_instruction(instruction=prompt)

        self._state = EpisodeState(seed=seed, task_name=self._task_name, prompt=prompt)
        self._episode_id += 1
        if self._lerobot_manager is not None:
            self._lerobot_manager.start_episode()

        raw_obs = self._task_env.get_obs()
        self._latest_raw_obs = raw_obs
        self._write_video_frame_from_obs(raw_obs)
        obs = self._convert_observation(raw_obs)
        return {
            "observation": obs,
            "reward": 0.0,
            "done": False,
            "info": {
                "task_name": self._task_name,
                "seed": seed,
                "prompt": prompt,
                "take_action_cnt": int(getattr(self._task_env, "take_action_cnt", 0)),
                "step_lim": int(getattr(self._task_env, "step_lim", 0)),
            },
        }

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._state is None:
            return self.reset(payload={})

        if "action_chunk" not in payload:
            raise KeyError("`step` payload must contain `action_chunk`.")

        action_chunk = np.asarray(payload["action_chunk"], dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.ndim != 2:
            raise ValueError(f"`action_chunk` must be rank-2, got shape={action_chunk.shape}")

        consumed = 0
        current_raw_obs = self._latest_raw_obs if self._latest_raw_obs is not None else self._task_env.get_obs()
        for action in action_chunk:
            if self._task_env.eval_success or self._task_env.take_action_cnt >= self._task_env.step_lim:
                break
            pre_obs = current_raw_obs
            self._task_env.take_action(action)
            current_raw_obs = self._task_env.get_obs()
            if self._lerobot_manager is not None and self._state is not None:
                self._lerobot_manager.add_step(
                    raw_obs=pre_obs,
                    action=np.asarray(action, dtype=np.float32),
                    task=self._state.prompt,
                )
            self._write_video_frame_from_obs(current_raw_obs)
            consumed += 1

        done = bool(self._task_env.eval_success or self._task_env.take_action_cnt >= self._task_env.step_lim)
        if done:
            # User-specified reward shaping:
            # - success terminal: 0
            # - failure terminal: -100
            reward = 0.0 if self._task_env.eval_success else -100.0
            outcome = "SUCCESS" if self._task_env.eval_success else "FAIL"
            print(
                f"[env_ws_episode_done] outcome={outcome} "
                f"episode={int(getattr(self._task_env, 'ep_num', -1))} "
                f"seed={int(self._state.seed)} "
                f"steps={int(self._task_env.take_action_cnt)}/{int(self._task_env.step_lim)}",
                flush=True,
            )
            self._stop_episode_recording()
            if self._lerobot_manager is not None:
                self._lerobot_manager.end_episode(
                    commit=True,
                    success=bool(self._task_env.eval_success),
                    step_lim=int(self._task_env.step_lim),
                    seed=int(self._state.seed) if self._state is not None else None,
                    prompt=str(self._state.prompt) if self._state is not None else None,
                )
        else:
            # non-terminal step penalty
            reward = -1.0
        self._latest_raw_obs = current_raw_obs
        obs = self._convert_observation(current_raw_obs)

        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": {
                "task_name": self._task_name,
                "seed": self._state.seed,
                "prompt": self._state.prompt,
                "consumed_actions": consumed,
                "take_action_cnt": int(self._task_env.take_action_cnt),
                "step_lim": int(self._task_env.step_lim),
                "eval_success": bool(self._task_env.eval_success),
                "plan_success": bool(getattr(self._task_env, "plan_success", True)),
            },
        }

    def finalize_collection_round(self) -> dict[str, Any]:
        self._stop_episode_recording()
        if self._lerobot_manager is None:
            return {"ok": True, "dataset_root": None, "repo_id": None, "num_episodes": 0, "round_index": None}
        result = self._lerobot_manager.finalize_round()
        print(f"[env_ws_round_finalized] {json.dumps(result, ensure_ascii=False)}", flush=True)
        return result


class RoboTwinWebsocketEnvServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        session: RoboTwinEnvSession,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._session = session

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        log = logging.getLogger("robotwin_env_ws")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"server": "robotwin_env_ws", "task_name": self._session.task_name}))
        log.info("client connected: %s", websocket.remote_address)

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                if not isinstance(msg, dict):
                    raise TypeError(f"Expected dict message, got {type(msg)}")

                cmd = msg.get("cmd")
                if cmd == "reset":
                    resp = self._session.reset(msg)
                elif cmd == "step":
                    resp = self._session.step(msg)
                elif cmd == "finalize_collection_round":
                    resp = self._session.finalize_collection_round()
                else:
                    raise ValueError(f"Unknown cmd: {cmd}")

                await websocket.send(packer.pack(resp))
            except websockets.ConnectionClosed:
                log.info("client disconnected: %s", websocket.remote_address)
                break
            except Exception as exc:
                await websocket.send(str(exc))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Error string included in previous frame.",
                )
                raise

    async def run(self) -> None:
        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
        ) as server:
            await server.serve_forever()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoboTwin websocket env server for online PPO client.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--default_prompt", type=str, default=None)
    parser.add_argument("--instruction_type", type=str, default="unseen")
    parser.add_argument("--prompt_max_descriptions", type=int, default=100)
    parser.add_argument("--prompt_seed_max_tries", type=int, default=64)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_save_dir", type=str, default=None)
    parser.add_argument("--save_lerobot", action="store_true")
    parser.add_argument("--lerobot_root", type=str, default=None)
    parser.add_argument("--lerobot_repo_id", type=str, default="lerobot-hammer-online")
    parser.add_argument("--lerobot_fps", type=int, default=50)
    parser.add_argument("--lerobot_overwrite", action="store_true")
    parser.add_argument(
        "--lerobot_resize_to_640x480",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize exported LeRobot images to 640x480 before writing (default: enabled).",
    )
    parser.add_argument("--seed_start", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _resolve_repo_root(args.repo_root)
    # Some RoboTwin env utilities use relative paths like "./assets/...".
    # Force working directory to RoboTwin root for path stability.
    os.chdir(repo_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    session = RoboTwinEnvSession(
        repo_root=repo_root,
        task_name=args.task_name,
        task_config=args.task_config,
        default_prompt=args.default_prompt,
        instruction_type=args.instruction_type,
        prompt_max_descriptions=args.prompt_max_descriptions,
        prompt_seed_max_tries=args.prompt_seed_max_tries,
        save_video=args.save_video,
        video_save_dir=args.video_save_dir,
        save_lerobot=args.save_lerobot,
        lerobot_root=args.lerobot_root,
        lerobot_repo_id=args.lerobot_repo_id,
        lerobot_fps=args.lerobot_fps,
        lerobot_overwrite=args.lerobot_overwrite,
        lerobot_resize_to_640x480=args.lerobot_resize_to_640x480,
        seed_start=args.seed_start,
    )

    server = RoboTwinWebsocketEnvServer(
        host=args.host,
        port=args.port,
        session=session,
    )

    try:
        asyncio.run(server.run())
    finally:
        session.close()


if __name__ == "__main__":
    main()
