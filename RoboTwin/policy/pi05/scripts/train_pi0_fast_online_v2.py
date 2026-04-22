#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Any

import numpy as np

from openpi.models import pi0_fast as _base_pi0_fast
from openpi.training import config as train_config
from openpi_online_ppo.env.single_env_ws import SingleEnvWebsocketEnv
from openpi_online_ppo.env.single_env_ws import SingleEnvWsConfig
from openpi_online_ppo.models import pi0_fast_rl as _rl_pi0_fast
from openpi_online_ppo.rl.exploration import CallableDCTNoiseFn
from openpi_online_ppo.rl.exploration import load_action_perturb_net
from openpi_online_ppo.rl.exploration import KeyframeExplorationConfig
from openpi_online_ppo.rl.exploration import LinearKeyframeNet
from openpi_online_ppo.rl.exploration import default_dct_noise
from openpi_online_ppo.rl.pi0_fast_online_trainer import Pi0FastOnlineRLConfig
from openpi_online_ppo.rl.pi0_fast_online_trainer import Pi0FastOnlineTrainer
from openpi_online_ppo.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy
from openpi_online_ppo.rl.pi0_fast_rollout import Pi0FastChunkCollector
from openpi_online_ppo.rl.reward_value import EnvChunkRewardProvider
from openpi_online_ppo.rl.reward_value import FixedRewardProvider
from openpi_online_ppo.rl.value_ws_client import ValueWebsocketClient


def _parse_dim_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() in {"all", "none"}:
        return None
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    return tuple(out) if out else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    p.add_argument("--policy.config", dest="policy_config", type=str, default="pi0_fast_aloha_robotwin_ppo")

    p.add_argument("--env.ws_url", dest="env_ws_url", type=str, default="ws://127.0.0.1:8765")
    p.add_argument("--env.task", dest="env_task", type=str, default=None)
    p.add_argument("--env.seed", dest="env_seed", type=int, default=0)
    p.add_argument("--env.reset_cmd", dest="env_reset_cmd", type=str, default="reset")
    p.add_argument("--env.step_cmd", dest="env_step_cmd", type=str, default="step")
    p.add_argument(
        "--env.finalize_collection_on_exit",
        dest="env_finalize_collection_on_exit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--env.obs_key", dest="env_obs_key", type=str, default="observation")
    p.add_argument("--env.reward_key", dest="env_reward_key", type=str, default="reward")
    p.add_argument("--env.done_key", dest="env_done_key", type=str, default="done")
    p.add_argument("--env.info_key", dest="env_info_key", type=str, default="info")
    p.add_argument("--env.action_key", dest="env_action_key", type=str, default="action_chunk")

    # Kept for compatibility/future use.
    p.add_argument("--repo_root", type=str, default=None)

    p.add_argument("--rollout_batch_size", type=int, default=64)
    p.add_argument("--mini_batch_size", type=int, default=32)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--value_epochs", type=int, default=1)
    p.add_argument("--buffer_capacity", type=int, default=1024)
    p.add_argument("--max_policy_lag", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--total_updates", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    # User decision: fixed reward is 0.
    p.add_argument("--fixed_reward", type=float, default=0.0)
    p.add_argument("--reward_mode", type=str, default="env_chunk", choices=("env_chunk", "fixed"))

    # User decision: keyframe exploration always on.
    p.add_argument(
        "--explore_mode",
        type=str,
        default="always",
        choices=("always", "conditional_keyframe", "never"),
    )
    p.add_argument("--explore_keyframe_threshold", type=float, default=0.5)
    p.add_argument("--explore_keyframe_gate", type=str, default="model_head", choices=("model_head", "small_net", "none"))
    p.add_argument("--explore_keyframe_net_ckpt", type=str, default=None)
    p.add_argument("--explore_dct_dims", type=int, default=4)
    p.add_argument("--explore_noise_std", type=float, default=0.01)
    p.add_argument(
        "--explore_perturb_backend",
        type=str,
        default="dct_gaussian",
        choices=("dct_gaussian", "action_formula", "action_network", "action_hybrid"),
    )
    p.add_argument("--explore_action_radius_min", type=float, default=0.0)
    p.add_argument("--explore_action_radius_max", type=float, default=0.15)
    p.add_argument("--explore_action_abs_clip", type=float, default=1.0)
    p.add_argument(
        "--explore_action_noise_dims",
        type=str,
        default="",
        help="Comma-separated action dim indices to perturb. Empty/all means all dims.",
    )
    p.add_argument("--explore_network_mix_alpha", type=float, default=0.5)
    p.add_argument("--explore_network_ckpt", type=str, default=None)
    p.add_argument("--explore_network_obs_dim", type=int, default=32)
    p.add_argument("--explore_network_latent_dim", type=int, default=16)
    p.add_argument("--explored_chunk_weight", type=float, default=1.0)
    p.add_argument("--non_explored_chunk_weight", type=float, default=1.0)
    p.add_argument("--value.ws_url", dest="value_ws_url", type=str, default=None)
    return p.parse_args()


def _to_rl_train_config(cfg: train_config.TrainConfig) -> train_config.TrainConfig:
    if not isinstance(cfg.model, _base_pi0_fast.Pi0FASTConfig):
        raise TypeError(f"Config `{cfg.name}` is not a Pi0FAST config.")

    base = cfg.model
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
        keyframe_num_bins=max(2, int(base.action_horizon) // 2),
    )
    return dataclasses.replace(cfg, model=rl_model)


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
        keyframe_num_bins=max(2, int(base.action_horizon) // 2),
    )
    return dataclasses.replace(cfg, model=rollout_model)


def _run_collection_only(
    *,
    collector: Pi0FastChunkCollector,
    total_updates: int,
    rollout_batch_size: int,
) -> None:
    collector.reset()
    global_chunks = 0
    for update_idx in range(max(1, int(total_updates))):
        rollout_samples = []
        while len(rollout_samples) < int(rollout_batch_size):
            rollout_samples.extend(collector.collect_chunk_batch(policy_version=0))
        global_chunks += len(rollout_samples)
        rollout_reward = float(np.mean([sample.reward for sample in rollout_samples])) if rollout_samples else 0.0
        logging.info(
            "collection_only_metrics=%s",
            {
                "rollout_reward": rollout_reward,
                "global_chunks": float(global_chunks),
                "global_updates": float(update_idx + 1),
                "rollout_batch_size": float(len(rollout_samples)),
            },
        )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    collect_only = int(args.ppo_epochs) <= 0 and int(args.value_epochs) <= 0
    cfg = train_config.get_config(args.policy_config)
    if collect_only:
        cfg = _to_rollout_only_config(
            cfg,
            enable_model_head_keyframe=(
                str(args.explore_mode) != "never" and str(args.explore_keyframe_gate) == "model_head"
            ),
        )
    else:
        cfg = _to_rl_train_config(cfg)

    exploration_cfg = KeyframeExplorationConfig(
        mode=str(args.explore_mode),
        always_on=(str(args.explore_mode) != "never"),
        explore_dct_dims=max(0, int(args.explore_dct_dims)),
        relaxed_fast_decoding=True,
        noise_std=float(args.explore_noise_std),
        keyframe_prob_threshold=float(args.explore_keyframe_threshold),
        perturb_backend=str(args.explore_perturb_backend),
        action_radius_min=float(args.explore_action_radius_min),
        action_radius_max=float(args.explore_action_radius_max),
        action_abs_clip=float(args.explore_action_abs_clip),
        action_noise_indices=_parse_dim_list(args.explore_action_noise_dims),
        network_mix_alpha=float(args.explore_network_mix_alpha),
        network_checkpoint=(str(args.explore_network_ckpt) if args.explore_network_ckpt else None),
        network_obs_dim=int(args.explore_network_obs_dim),
        network_latent_dim=int(args.explore_network_latent_dim),
    )
    perturb_net = None
    if exploration_cfg.network_checkpoint:
        perturb_net = load_action_perturb_net(exploration_cfg.network_checkpoint)

    value_client = None if collect_only else (ValueWebsocketClient(args.value_ws_url) if args.value_ws_url else None)
    keyframe_net = None
    keyframe_mode = "none"
    if str(args.explore_keyframe_gate) == "small_net":
        if args.explore_keyframe_net_ckpt:
            keyframe_net = LinearKeyframeNet.load(args.explore_keyframe_net_ckpt)
            keyframe_mode = "local_small_net"
        elif value_client is not None:
            keyframe_mode = "remote_value_ws"
        else:
            raise ValueError(
                "small_net gate requires either --explore_keyframe_net_ckpt "
                "or --value.ws_url (for remote keyframe inference)."
            )
    def _noise_impl(**kwargs):
        md = dict(kwargs.get("metadata") or {})
        if perturb_net is not None:
            md["perturb_net"] = perturb_net
        kwargs["metadata"] = md
        return default_dct_noise(
            **kwargs,
            noise_std=exploration_cfg.noise_std,
            cfg=exploration_cfg,
        )

    noise_fn = CallableDCTNoiseFn(_noise_impl)

    keyframe_prob_fn = None
    policy_holder: dict[str, Any] = {}
    if str(args.explore_keyframe_gate) == "small_net":
        if keyframe_mode == "local_small_net":
            keyframe_prob_fn = lambda obs: float(keyframe_net.predict_prob(obs))  # noqa: E731
        elif keyframe_mode == "remote_value_ws":
            def _keyframe_prob_fn_remote(obs):
                policy_obj = policy_holder.get("policy")
                if policy_obj is None:
                    raise RuntimeError("policy is not initialized for remote keyframe inference")
                transformed = policy_obj.transform_observation(obs)
                return float(value_client.predict_keyframe(transformed))

            keyframe_prob_fn = _keyframe_prob_fn_remote
    elif str(args.explore_keyframe_gate) == "none":
        keyframe_prob_fn = lambda obs: 0.0  # noqa: E731

    policy = create_trained_pi0_fast_rl_policy(
        cfg,
        args.policy_path,
        exploration_config=exploration_cfg,
        dct_noise_fn=noise_fn,
        keyframe_prob_fn=keyframe_prob_fn,
    )
    policy_holder["policy"] = policy

    env = SingleEnvWebsocketEnv(
        SingleEnvWsConfig(
            ws_url=args.env_ws_url,
            task_name=args.env_task,
            repo_root=args.repo_root or None,
            seed=args.env_seed,
            reset_cmd=args.env_reset_cmd,
            step_cmd=args.env_step_cmd,
            obs_key=args.env_obs_key,
            reward_key=args.env_reward_key,
            done_key=args.env_done_key,
            info_key=args.env_info_key,
            action_key=args.env_action_key,
        )
    )
    if args.reward_mode == "fixed":
        provider = FixedRewardProvider(reward=float(args.fixed_reward))
    else:
        provider = EnvChunkRewardProvider()
    collector = Pi0FastChunkCollector(
        env=env,
        policy=policy,
        provider=provider,
        value_predictor=value_client,
        compute_values=not collect_only,
        include_logprobs=not collect_only,
    )
    completed = False
    try:
        if collect_only:
            logging.info(
                "collection_only_mode enabled: skipping trainer/value updates and rollout logprob recomputation"
            )
            _run_collection_only(
                collector=collector,
                total_updates=args.total_updates,
                rollout_batch_size=args.rollout_batch_size,
            )
        else:
            rl_cfg = Pi0FastOnlineRLConfig(
                rollout_batch_size=args.rollout_batch_size,
                mini_batch_size=args.mini_batch_size,
                ppo_epochs=args.ppo_epochs,
                value_epochs=args.value_epochs,
                gamma=args.gamma,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                buffer_capacity=args.buffer_capacity,
                max_policy_lag=args.max_policy_lag,
                total_updates=args.total_updates,
                seed=args.seed,
                explored_chunk_weight=args.explored_chunk_weight,
                non_explored_chunk_weight=args.non_explored_chunk_weight,
            )

            trainer = Pi0FastOnlineTrainer(
                cfg=rl_cfg,
                policy=policy,
                collector=collector,
            )
            trainer.train()
        completed = True
    finally:
        if completed and bool(args.env_finalize_collection_on_exit):
            try:
                finalize_resp = env.finalize_collection_round()
                logging.info("env_finalize_collection_round=%s", finalize_resp)
            except Exception:
                logging.exception("Failed to finalize collection round on env websocket server.")
        env.close()
        if value_client is not None:
            value_client.close()


if __name__ == "__main__":
    main()
