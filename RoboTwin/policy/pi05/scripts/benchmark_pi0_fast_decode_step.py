#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0_fast as _base_pi0_fast
from openpi.training import config as train_config
from openpi_online_ppo.data.local_lerobot_loader import ensure_local_hf_cache
from openpi_online_ppo.models import pi0_fast_rl as _rl_pi0_fast
from openpi_online_ppo.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy


KEYFRAME_NUM_BINS = 16


def _to_rollout_only_config(cfg: train_config.TrainConfig) -> train_config.TrainConfig:
    if not isinstance(cfg.model, _base_pi0_fast.Pi0FASTConfig):
        raise TypeError(f"Config `{cfg.name}` is not a Pi0FAST config.")

    base = cfg.model
    rollout_model = _rl_pi0_fast.Pi0FASTConfig(
        dtype=base.dtype,
        paligemma_variant=base.paligemma_variant,
        action_dim=base.action_dim,
        action_horizon=base.action_horizon,
        max_token_len=base.max_token_len,
        fast_model_tokenizer=base.fast_model_tokenizer,
        fast_model_tokenizer_kwargs=base.fast_model_tokenizer_kwargs,
        use_value_head=False,
        use_keyframe_head=False,
        keyframe_num_bins=KEYFRAME_NUM_BINS,
    )
    return dataclasses.replace(cfg, model=rollout_model)


def _resolve_policy_config_name(raw_name: str, policy_path: str) -> str:
    name = str(raw_name).strip()
    if name and name != "auto":
        return name
    policy_path_l = str(policy_path).lower()
    if "ee_delta" in policy_path_l:
        return "pi0_fast_aloha_robotwin_ppo_ee_delta"
    return "pi0_fast_aloha_robotwin_ppo"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Pi0FAST _decode_step and row-decode latency.")
    p.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    p.add_argument("--policy.config", dest="policy_config", type=str, default="auto")
    p.add_argument("--dataset_root", type=str, default="/mnt/data/lhm/vla-rl/RoboTwin/eval_result/lerobot_ppo_corpus")
    p.add_argument("--repo_id", type=str, default="lerobot_ppo_corpus")
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--max_decoding_steps", type=int, default=256)
    p.add_argument("--decode_repeats", type=int, default=32)
    p.add_argument("--warmup_repeats", type=int, default=2)
    p.add_argument("--token", type=int, default=48, help="Token id used for fixed-token _decode_step benchmark.")
    p.add_argument("--benchmark_row_decode", action="store_true")
    p.add_argument("--row_limit", type=int, default=2)
    p.add_argument("--benchmark_row_replay", action="store_true")
    return p.parse_args()


def _summarize(name: str, xs: list[float]) -> str:
    if not xs:
        return f"{name}: n=0"
    return (
        f"{name}: n={len(xs)} "
        f"mean={statistics.mean(xs):.4f}s "
        f"min={min(xs):.4f}s "
        f"max={max(xs):.4f}s"
    )


def _block_tree(tree: Any) -> Any:
    return jax.tree.map(jax.block_until_ready, tree)


def _load_obs(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ensure_local_hf_cache()
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    item = ds[int(args.sample_index)]
    task_val = item["task"] if "task" in item else ""
    prompt = task_val if isinstance(task_val, str) else str(np.asarray(task_val).item())
    return {
        "images": {
            "cam_high": np.asarray(item["observation.images.cam_high"], dtype=np.float32),
            "cam_left_wrist": np.asarray(item["observation.images.cam_left_wrist"], dtype=np.float32),
            "cam_right_wrist": np.asarray(item["observation.images.cam_right_wrist"], dtype=np.float32),
        },
        "state": np.asarray(item["observation.state"], dtype=np.float32),
        "prompt": prompt,
    }


def main() -> None:
    args = _parse_args()
    cfg = train_config.get_config(_resolve_policy_config_name(args.policy_config, args.policy_path))
    cfg = _to_rollout_only_config(cfg)
    policy = create_trained_pi0_fast_rl_policy(
        cfg,
        args.policy_path,
        sample_kwargs={"temperature": 0.0, "max_decoding_steps": int(args.max_decoding_steps)},
    )

    obs = _load_obs(args)
    observation, _ = policy._prepare_observation(obs)  # noqa: SLF001
    print(f"loaded_obs sample_index={int(args.sample_index)} prompt={obs['prompt']!r}")

    prepare_times: list[float] = []
    decode_times: list[float] = []

    total_rounds = max(1, int(args.warmup_repeats) + int(args.decode_repeats))
    for round_idx in range(total_rounds):
        t0 = time.perf_counter()
        decode_state = policy.model._prepare_decode_prefix(observation, int(args.max_decoding_steps))
        _block_tree(decode_state)
        prepare_elapsed = time.perf_counter() - t0
        if round_idx >= int(args.warmup_repeats):
            prepare_times.append(prepare_elapsed)

        last_logit = decode_state["last_logit"]
        kv_cache = decode_state["kv_cache"]
        prefill_size = int(decode_state["prefill_size"])
        prefill_len = decode_state["prefill_len"]
        prefix_start = decode_state["prefix_start"]

        t0 = time.perf_counter()
        next_logit, next_cache = policy._decode_step(  # noqa: SLF001
            last_logit=last_logit,
            cache=kv_cache,
            token=int(args.token),
            step_idx=0,
            prefill_len=prefill_len,
            prefill_size=prefill_size,
            prefix_start=prefix_start,
            max_decoding_steps=int(args.max_decoding_steps),
        )
        _block_tree((next_logit, next_cache))
        decode_elapsed = time.perf_counter() - t0
        if round_idx >= int(args.warmup_repeats):
            decode_times.append(decode_elapsed)

    print(_summarize("prepare_decode_prefix", prepare_times))
    print(_summarize("single_decode_step", decode_times))

    if args.benchmark_row_decode:
        decode_state = policy.model._prepare_decode_prefix(observation, int(args.max_decoding_steps))
        _block_tree(decode_state)
        last_logit = decode_state["last_logit"]
        kv_cache = decode_state["kv_cache"]
        prefill_size = int(decode_state["prefill_size"])
        prefill_len = decode_state["prefill_len"]
        prefix_start = decode_state["prefix_start"]

        step_idx = 0
        for token in policy._fast_tokenizer.action_prefix_tokens().tolist():  # noqa: SLF001
            last_logit, kv_cache = policy._decode_step(  # noqa: SLF001
                last_logit=last_logit,
                cache=kv_cache,
                token=int(token),
                step_idx=step_idx,
                prefill_len=prefill_len,
                prefill_size=prefill_size,
                prefix_start=prefix_start,
                max_decoding_steps=int(args.max_decoding_steps),
            )
            _block_tree((last_logit, kv_cache))
            step_idx += 1

        row_count = policy._fast_tokenizer.num_rows(  # noqa: SLF001
            action_horizon=policy._action_horizon,  # noqa: SLF001
            action_dim=policy._action_dim,  # noqa: SLF001
        )
        row_char_width = policy._fast_tokenizer.row_char_width(  # noqa: SLF001
            action_horizon=policy._action_horizon,  # noqa: SLF001
            action_dim=policy._action_dim,  # noqa: SLF001
        )
        row_limit = max(1, min(int(args.row_limit), int(row_count)))
        print(f"benchmark_row_decode rows={row_limit} row_char_width={row_char_width}")

        row_times: list[float] = []
        row_pick_times: list[float] = []
        row_step_times: list[float] = []
        row_text_times: list[float] = []
        row_replay_times: list[float] = []
        for row_idx in range(row_limit):
            row_start_logit = last_logit
            row_start_cache = kv_cache
            row_start_step_idx = step_idx
            current_row_pg_tokens: list[int] = []
            row_decode_len = 0
            max_row_tokens = max(8, row_char_width * 4)
            row_t0 = time.perf_counter()
            while True:
                pick_t0 = time.perf_counter()
                token = policy._pick_valid_row_token(  # noqa: SLF001
                    logit=np.asarray(last_logit[0, 0, :], dtype=np.float32),
                    current_row_pg_tokens=current_row_pg_tokens,
                    row_char_width=row_char_width,
                    temperature=0.0,
                )
                row_pick_times.append(time.perf_counter() - pick_t0)
                current_row_pg_tokens.append(int(token))
                step_idx += 1
                step_t0 = time.perf_counter()
                last_logit, kv_cache = policy._decode_step(  # noqa: SLF001
                    last_logit=last_logit,
                    cache=kv_cache,
                    token=int(token),
                    step_idx=step_idx - 1,
                    prefill_len=prefill_len,
                    prefill_size=prefill_size,
                    prefix_start=prefix_start,
                    max_decoding_steps=int(args.max_decoding_steps),
                )
                _block_tree((last_logit, kv_cache))
                row_step_times.append(time.perf_counter() - step_t0)
                text_t0 = time.perf_counter()
                row_decode_len = len(
                    policy._fast_tokenizer.decode_pg_tokens_to_text(np.asarray(current_row_pg_tokens, dtype=np.int32))  # noqa: SLF001
                )
                row_text_times.append(time.perf_counter() - text_t0)
                if row_decode_len >= row_char_width or len(current_row_pg_tokens) >= max_row_tokens:
                    break
            row_elapsed = time.perf_counter() - row_t0
            row_times.append(row_elapsed)
            print(
                f"row[{row_idx}] elapsed={row_elapsed:.4f}s tokens={len(current_row_pg_tokens)} "
                f"text_len={row_decode_len}"
            )
            if args.benchmark_row_replay:
                replay_t0 = time.perf_counter()
                replay_logit = row_start_logit
                replay_cache = row_start_cache
                replay_step_idx = row_start_step_idx
                for token in current_row_pg_tokens:
                    replay_logit, replay_cache = policy._decode_step(  # noqa: SLF001
                        last_logit=replay_logit,
                        cache=replay_cache,
                        token=int(token),
                        step_idx=replay_step_idx,
                        prefill_len=prefill_len,
                        prefill_size=prefill_size,
                        prefix_start=prefix_start,
                        max_decoding_steps=int(args.max_decoding_steps),
                    )
                    _block_tree((replay_logit, replay_cache))
                    replay_step_idx += 1
                replay_elapsed = time.perf_counter() - replay_t0
                row_replay_times.append(replay_elapsed)
                print(f"row[{row_idx}] replay_elapsed={replay_elapsed:.4f}s replay_tokens={len(current_row_pg_tokens)}")
        print(_summarize("row_decode_loop", row_times))
        print(_summarize("row_pick_valid_row_token", row_pick_times))
        print(_summarize("row_decode_step_only", row_step_times))
        print(_summarize("row_decode_text_only", row_text_times))
        if row_replay_times:
            print(_summarize("row_replay_loop", row_replay_times))


if __name__ == "__main__":
    main()
