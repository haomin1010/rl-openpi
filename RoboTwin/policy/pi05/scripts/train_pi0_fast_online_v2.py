#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import logging

from openpi.models import pi0_fast as _base_pi0_fast
from openpi.training import config as train_config
from openpi_online_ppo.env.single_env_ws import SingleEnvWebsocketEnv
from openpi_online_ppo.env.single_env_ws import SingleEnvWsConfig
from openpi_online_ppo.models import pi0_fast_rl as _rl_pi0_fast
from openpi_online_ppo.rl.exploration import CallableDCTNoiseFn
from openpi_online_ppo.rl.exploration import KeyframeExplorationConfig
from openpi_online_ppo.rl.exploration import default_dct_noise
from openpi_online_ppo.rl.pi0_fast_online_trainer import Pi0FastOnlineRLConfig
from openpi_online_ppo.rl.pi0_fast_online_trainer import Pi0FastOnlineTrainer
from openpi_online_ppo.rl.pi0_fast_policy import create_trained_pi0_fast_rl_policy
from openpi_online_ppo.rl.pi0_fast_rollout import Pi0FastChunkCollector
from openpi_online_ppo.rl.reward_value import FixedRewardProvider


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy.path", dest="policy_path", type=str, required=True)
    p.add_argument("--policy.config", dest="policy_config", type=str, default="pi0_fast_aloha_robotwin_ppo")

    p.add_argument("--env.ws_url", dest="env_ws_url", type=str, default="ws://127.0.0.1:8765")
    p.add_argument("--env.task", dest="env_task", type=str, default=None)
    p.add_argument("--env.seed", dest="env_seed", type=int, default=0)
    p.add_argument("--env.reset_cmd", dest="env_reset_cmd", type=str, default="reset")
    p.add_argument("--env.step_cmd", dest="env_step_cmd", type=str, default="step")
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

    # User decision: keyframe exploration always on.
    p.add_argument("--explore_dct_dims", type=int, default=4)
    p.add_argument("--explore_noise_std", type=float, default=0.01)
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
    )
    return dataclasses.replace(cfg, model=rl_model)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cfg = train_config.get_config(args.policy_config)
    cfg = _to_rl_train_config(cfg)

    exploration_cfg = KeyframeExplorationConfig(
        always_on=True,
        explore_dct_dims=max(0, int(args.explore_dct_dims)),
        relaxed_fast_decoding=True,
        noise_std=float(args.explore_noise_std),
    )
    noise_fn = CallableDCTNoiseFn(
        lambda **kwargs: default_dct_noise(
            **kwargs,
            noise_std=exploration_cfg.noise_std,
        )
    )

    policy = create_trained_pi0_fast_rl_policy(
        cfg,
        args.policy_path,
        exploration_config=exploration_cfg,
        dct_noise_fn=noise_fn,
    )

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
    provider = FixedRewardProvider(reward=float(args.fixed_reward))
    collector = Pi0FastChunkCollector(env=env, policy=policy, provider=provider)

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
    )

    trainer = Pi0FastOnlineTrainer(cfg=rl_cfg, policy=policy, collector=collector)
    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
