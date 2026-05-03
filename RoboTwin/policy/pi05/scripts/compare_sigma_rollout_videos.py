#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import numpy as np

from openpi.models import pi0_fast as _base_pi0_fast
from openpi.training import config as train_config
from openpi_online_ppo.env.single_env_ws import SingleEnvWebsocketEnv
from openpi_online_ppo.env.single_env_ws import SingleEnvWsConfig
from openpi_online_ppo.models import pi0_fast_rl as _rl_pi0_fast
from openpi_online_ppo.rl.exploration import KeyframeExplorationConfig
from openpi_online_ppo.rl.exploration import LinearKeyframeNet
from openpi_online_ppo.rl.exploration import load_action_perturb_net
from openpi_online_ppo.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy
from openpi_online_ppo.rl.value_ws_client import ValueWebsocketClient

KEYFRAME_NUM_BINS = 16


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _parse_dim_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"all", "none"}:
        return None
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return tuple(out) if out else None


def _resolve_policy_config_name(raw_name: str, policy_path: str) -> str:
    name = str(raw_name).strip()
    if name and name != "auto":
        return name
    policy_path_l = str(policy_path).lower()
    if "ee_delta" in policy_path_l:
        return "pi0_fast_aloha_robotwin_ppo_ee_delta"
    return "pi0_fast_aloha_robotwin_ppo"


def _to_rollout_only_config(
    cfg: train_config.TrainConfig,
    *,
    enable_model_head_keyframe: bool,
) -> train_config.TrainConfig:
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
        use_keyframe_head=bool(enable_model_head_keyframe),
        keyframe_num_bins=KEYFRAME_NUM_BINS,
    )
    return dataclasses.replace(cfg, model=rollout_model)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one no-explore and one sigma-explore episode for side-by-side video comparison.")
    p.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    p.add_argument("--policy.config", dest="policy_config", type=str, default="auto")
    p.add_argument("--env.ws_url", dest="env_ws_url", type=str, default="ws://127.0.0.1:8765")
    p.add_argument("--value.ws_url", dest="value_ws_url", type=str, default=None)
    p.add_argument("--env.task", dest="env_task", type=str, default=None)
    p.add_argument("--env.seed", dest="env_seed", type=int, default=100000)
    p.add_argument("--env.reset_cmd", dest="env_reset_cmd", type=str, default="reset")
    p.add_argument("--env.step_cmd", dest="env_step_cmd", type=str, default="step")
    p.add_argument("--env.obs_key", dest="env_obs_key", type=str, default="observation")
    p.add_argument("--env.reward_key", dest="env_reward_key", type=str, default="reward")
    p.add_argument("--env.done_key", dest="env_done_key", type=str, default="done")
    p.add_argument("--env.info_key", dest="env_info_key", type=str, default="info")
    p.add_argument("--env.action_key", dest="env_action_key", type=str, default="action_chunk")
    p.add_argument("--max_chunks", type=int, default=64)
    p.add_argument(
        "--num_sigma_rollouts",
        type=int,
        default=1,
        help="How many sigma-explore rollouts to run for the same env seed/setting.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma_ckpt", type=str, required=True)
    p.add_argument("--explore_mode", type=str, default="conditional_keyframe", choices=("always", "conditional_keyframe"))
    p.add_argument("--explore_keyframe_threshold", type=float, default=0.5)
    p.add_argument("--explore_keyframe_gate", type=str, default="model_head", choices=("model_head", "small_net", "none"))
    p.add_argument("--explore_keyframe_net_ckpt", type=str, default=None)
    p.add_argument("--explore_action_noise_dims", type=str, default="")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument(
        "--log_file",
        type=str,
        default="",
        help="Optional log file path for rollout timing logs.",
    )
    return p.parse_args()


def _make_env(args: argparse.Namespace) -> SingleEnvWebsocketEnv:
    return SingleEnvWebsocketEnv(
        SingleEnvWsConfig(
            ws_url=str(args.env_ws_url),
            task_name=args.env_task,
            seed=int(args.env_seed),
            reset_cmd=str(args.env_reset_cmd),
            step_cmd=str(args.env_step_cmd),
            obs_key=str(args.env_obs_key),
            reward_key=str(args.env_reward_key),
            done_key=str(args.env_done_key),
            info_key=str(args.env_info_key),
            action_key=str(args.env_action_key),
        )
    )


def _make_policy(
    *,
    args: argparse.Namespace,
    explore_mode: str,
):
    build_t0 = time.perf_counter()
    gate = str(args.explore_keyframe_gate) if str(explore_mode) != "never" else "none"
    _progress(
        "loading_policy "
        f"explore_mode={explore_mode} "
        f"keyframe_gate={gate} "
        f"sigma_ckpt={args.sigma_ckpt if str(explore_mode) != 'never' else None}"
    )
    policy_config_name = _resolve_policy_config_name(args.policy_config, args.policy_path)
    cfg = train_config.get_config(policy_config_name)
    cfg = _to_rollout_only_config(cfg, enable_model_head_keyframe=(gate == "model_head"))

    perturb_net = None
    if str(explore_mode) != "never":
        perturb_net = load_action_perturb_net(args.sigma_ckpt)

    exploration_cfg = KeyframeExplorationConfig(
        mode=str(explore_mode),
        always_on=(str(explore_mode) != "never"),
        relaxed_fast_decoding=True,
        keyframe_prob_threshold=float(args.explore_keyframe_threshold),
        perturb_backend=("dct_sigma_network" if str(explore_mode) != "never" else "dct_gaussian"),
        action_noise_indices=_parse_dim_list(args.explore_action_noise_dims),
        network_checkpoint=(str(args.sigma_ckpt) if str(explore_mode) != "never" else None),
    )

    value_client = ValueWebsocketClient(args.value_ws_url) if gate == "small_net" and args.value_ws_url else None
    keyframe_net = LinearKeyframeNet.load(args.explore_keyframe_net_ckpt) if gate == "small_net" and args.explore_keyframe_net_ckpt else None
    if gate == "small_net" and keyframe_net is None and value_client is None:
        raise ValueError(
            "small_net gate requires either --explore_keyframe_net_ckpt "
            "or --value.ws_url (for remote keyframe inference)."
        )

    keyframe_prob_fn = None
    policy_holder: dict[str, Any] = {}
    if gate == "small_net":
        if keyframe_net is not None:
            keyframe_prob_fn = lambda obs: float(keyframe_net.predict_prob(obs))  # noqa: E731
        else:
            def _keyframe_prob_fn_remote(obs: dict[str, Any]) -> float:
                t0 = time.perf_counter()
                policy_obj = policy_holder.get("policy")
                if policy_obj is None:
                    raise RuntimeError("policy is not initialized for remote keyframe inference")
                transformed = policy_obj.transform_observation(obs)
                out = float(value_client.predict_keyframe(transformed))
                logging.info(
                    "remote_keyframe_timing ws_url=%s elapsed=%.3fs prob=%.4f",
                    str(args.value_ws_url),
                    time.perf_counter() - t0,
                    out,
                )
                return out

            keyframe_prob_fn = _keyframe_prob_fn_remote
    elif gate == "none":
        keyframe_prob_fn = lambda obs: 0.0  # noqa: E731

    policy = create_trained_pi0_fast_rl_policy(
        cfg,
        args.policy_path,
        sample_kwargs={"temperature": 0.0},
        exploration_config=exploration_cfg,
        keyframe_prob_fn=keyframe_prob_fn,
        perturb_net=perturb_net,
    )
    policy_holder["policy"] = policy
    logging.info(
        "make_policy_timing explore_mode=%s keyframe_gate=%s elapsed=%.3fs",
        str(explore_mode),
        gate,
        time.perf_counter() - build_t0,
    )
    return policy, value_client


def _run_episode(
    *,
    env: SingleEnvWebsocketEnv,
    policy: Any,
    label: str,
    max_chunks: int,
) -> dict[str, Any]:
    _progress(f"episode_start label={label} seed={int(env._cfg.seed)}")  # noqa: SLF001
    obs = env.reset()
    prompt = str(obs.get("prompt", ""))
    _progress(f"episode_reset_done label={label} prompt={prompt}")

    rewards: list[float] = []
    keyframe_probs: list[float] = []
    action_delta_l2: list[float] = []
    action_delta_abs_max: list[float] = []
    dct_delta_l2: list[float] = []
    explored_flags: list[bool] = []
    done = False
    done_reason: str | None = None
    info_last: dict[str, Any] = {}
    chunks = 0

    while not done and chunks < int(max_chunks):
        _progress(f"before_sample_chunk label={label} chunk={chunks + 1}")
        sample_t0 = time.perf_counter()
        trace = policy.sample_chunk(obs, include_logprobs=False)
        sample_elapsed = time.perf_counter() - sample_t0
        _progress(
            "after_sample_chunk "
            f"label={label} "
            f"chunk={chunks + 1} "
            f"explored={bool(trace.get('exploration_applied', False))} "
            f"keyframe_prob={float(trace.get('keyframe_prob', 0.0)):.4f}"
        )
        logging.info(
            "rollout_timing label=%s chunk=%d phase=sample_chunk elapsed=%.3fs explored=%s keyframe_prob=%.4f",
            label,
            chunks + 1,
            sample_elapsed,
            bool(trace.get("exploration_applied", False)),
            float(trace.get("keyframe_prob", 0.0)),
        )
        _progress(f"before_env_step label={label} chunk={chunks + 1}")
        step_t0 = time.perf_counter()
        next_obs, env_reward, done, info = env.step(trace["action_chunk"])
        step_elapsed = time.perf_counter() - step_t0
        _progress(f"after_env_step label={label} chunk={chunks + 1} done={done} reward={float(env_reward):.4f}")
        logging.info(
            "rollout_timing label=%s chunk=%d phase=env_step elapsed=%.3fs done=%s reward=%.4f",
            label,
            chunks + 1,
            step_elapsed,
            bool(done),
            float(env_reward),
        )
        rewards.append(float(env_reward))
        keyframe_probs.append(float(trace.get("keyframe_prob", 0.0)))
        explored = bool(trace.get("exploration_applied", False))
        explored_flags.append(explored)

        sampled_dct = np.asarray(trace["sampled_dct_coeffs"], dtype=np.float32)
        executed_dct = np.asarray(trace["dct_coeffs"], dtype=np.float32)
        dct_delta = executed_dct - sampled_dct
        dct_delta_l2.append(float(np.linalg.norm(dct_delta.reshape(-1), ord=2)))

        sampled_action = np.asarray(policy._fast_tokenizer.decode_action_dct_coeffs(sampled_dct), dtype=np.float32)  # noqa: SLF001
        executed_action = np.asarray(trace["decoded_actions"], dtype=np.float32)
        action_delta = executed_action - sampled_action
        action_delta_l2.append(float(np.linalg.norm(action_delta.reshape(-1), ord=2)))
        action_delta_abs_max.append(float(np.max(np.abs(action_delta))) if action_delta.size > 0 else 0.0)

        obs = next_obs
        info_last = dict(info)
        done_reason = str(info.get("done_reason")) if isinstance(info, dict) and info.get("done_reason") is not None else done_reason
        chunks += 1
        _progress(
            "episode_chunk "
            f"label={label} "
            f"chunk={chunks} "
            f"reward={float(env_reward):.4f} "
            f"explored={explored} "
            f"keyframe_prob={float(trace.get('keyframe_prob', 0.0)):.4f} "
            f"action_delta_abs_max={(action_delta_abs_max[-1] if action_delta_abs_max else 0.0):.6f} "
            f"done={done}"
        )

    result = {
        "label": label,
        "seed": int(info_last.get("seed", 0)),
        "prompt": prompt,
        "chunks": int(chunks),
        "done": bool(done),
        "done_reason": done_reason,
        "env_reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        "env_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "keyframe_prob_mean": float(np.mean(keyframe_probs)) if keyframe_probs else 0.0,
        "keyframe_prob_max": float(np.max(keyframe_probs)) if keyframe_probs else 0.0,
        "exploration_applied_chunks": int(np.sum(np.asarray(explored_flags, dtype=np.int32))),
        "exploration_applied_rate": float(np.mean(np.asarray(explored_flags, dtype=np.float32))) if explored_flags else 0.0,
        "action_delta_l2_mean": float(np.mean(action_delta_l2)) if action_delta_l2 else 0.0,
        "action_delta_l2_max": float(np.max(action_delta_l2)) if action_delta_l2 else 0.0,
        "action_delta_abs_max_mean": float(np.mean(action_delta_abs_max)) if action_delta_abs_max else 0.0,
        "action_delta_abs_max_max": float(np.max(action_delta_abs_max)) if action_delta_abs_max else 0.0,
        "dct_delta_l2_mean": float(np.mean(dct_delta_l2)) if dct_delta_l2 else 0.0,
        "dct_delta_l2_max": float(np.max(dct_delta_l2)) if dct_delta_l2 else 0.0,
        "last_info": info_last,
    }
    _progress(f"episode_done label={label} summary={json.dumps(result, ensure_ascii=True)}")
    return result


def _aggregate_episode_metrics(runs: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    if not runs:
        return {"label": label, "num_runs": 0}

    scalar_keys = [
        "chunks",
        "env_reward_sum",
        "env_reward_mean",
        "keyframe_prob_mean",
        "keyframe_prob_max",
        "exploration_applied_chunks",
        "exploration_applied_rate",
        "action_delta_l2_mean",
        "action_delta_l2_max",
        "action_delta_abs_max_mean",
        "action_delta_abs_max_max",
        "dct_delta_l2_mean",
        "dct_delta_l2_max",
    ]
    summary: dict[str, Any] = {
        "label": label,
        "num_runs": int(len(runs)),
        "done_count": int(sum(1 for run in runs if bool(run.get("done", False)))),
        "done_reasons": [run.get("done_reason") for run in runs],
        "seeds": [int(run.get("seed", 0)) for run in runs],
        "prompts": [str(run.get("prompt", "")) for run in runs],
    }
    for key in scalar_keys:
        vals = [float(run.get(key, 0.0)) for run in runs]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_min"] = float(np.min(vals))
        summary[f"{key}_max"] = float(np.max(vals))
    return summary


def main() -> None:
    args = _parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        log_path = pathlib.Path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=handlers, force=True)
    np.random.seed(int(args.seed))
    if int(args.num_sigma_rollouts) <= 0:
        raise ValueError("--num_sigma_rollouts must be positive.")

    sigma_runs: list[dict[str, Any]] = []
    for rollout_idx in range(int(args.num_sigma_rollouts)):
        sigma_env = _make_env(args)
        sigma_policy, sigma_value_client = _make_policy(
            args=args,
            explore_mode=str(args.explore_mode),
        )
        sigma = _run_episode(
            env=sigma_env,
            policy=sigma_policy,
            label=f"sigma_explore_{rollout_idx:03d}",
            max_chunks=int(args.max_chunks),
        )
        sigma_runs.append(sigma)
        sigma_env.close()
        if sigma_value_client is not None:
            sigma_value_client.close()

    baseline_env = _make_env(args)
    baseline_policy, baseline_value_client = _make_policy(
        args=args,
        explore_mode="never",
    )
    baseline = _run_episode(
        env=baseline_env,
        policy=baseline_policy,
        label="no_explore",
        max_chunks=int(args.max_chunks),
    )
    baseline_env.close()
    if baseline_value_client is not None:
        baseline_value_client.close()

    sigma_summary = _aggregate_episode_metrics(sigma_runs, label="sigma_explore_summary")

    summary = {
        "policy_path": str(pathlib.Path(args.policy_path).resolve()),
        "sigma_ckpt": str(pathlib.Path(args.sigma_ckpt).resolve()),
        "env_ws_url": str(args.env_ws_url),
        "env_seed": int(args.env_seed),
        "max_chunks": int(args.max_chunks),
        "num_sigma_rollouts": int(args.num_sigma_rollouts),
        "baseline": baseline,
        "sigma_explore": sigma_runs[0],
        "sigma_explore_runs": sigma_runs,
        "sigma_explore_summary": sigma_summary,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    if args.out_json:
        out = pathlib.Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved rollout comparison to {out}")


if __name__ == "__main__":
    main()
