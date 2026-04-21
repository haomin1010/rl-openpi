from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

import numpy as np

from openpi_online_ppo.env.single_env_ws import SingleEnvWebsocketEnv
from openpi_online_ppo.rl.chunk_types import ChunkSample
from openpi_online_ppo.rl.pi0_fast_policy import Pi0FastRLPolicy
from openpi_online_ppo.rl.reward_value import ChunkRewardValueProvider


@dataclass
class _EnvState:
    obs: dict[str, Any]
    transformed_obs: dict[str, Any]
    task: str | None
    value: float
    step_id: int = 0


class ValuePredictor(Protocol):
    def predict(self, transformed_obs: dict[str, Any]) -> float: ...


class Pi0FastChunkCollector:
    """Collect chunk-level online RL samples from a single websocket env."""

    def __init__(
        self,
        *,
        env: SingleEnvWebsocketEnv,
        policy: Pi0FastRLPolicy,
        provider: ChunkRewardValueProvider,
        value_predictor: ValuePredictor | None = None,
    ) -> None:
        self._env = env
        self._policy = policy
        self._provider = provider
        self._value_predictor = value_predictor
        self._state: _EnvState | None = None

    def _predict_value(self, obs: dict[str, Any], transformed_obs: dict[str, Any] | None = None) -> float:
        if self._value_predictor is None:
            return self._policy.predict_value(obs)
        transformed_obs = transformed_obs if transformed_obs is not None else self._policy.transform_observation(obs)
        return float(self._value_predictor.predict(transformed_obs))

    def reset(self) -> None:
        obs = self._env.reset()
        task = obs.get("prompt")
        if task is None:
            task = ""
            obs["prompt"] = task
        transformed = self._policy.transform_observation(obs)
        value = self._predict_value(obs, transformed)
        self._state = _EnvState(obs=obs, transformed_obs=transformed, task=str(task), value=value, step_id=0)

    def collect_chunk_batch(self, *, policy_version: int) -> list[ChunkSample]:
        if self._state is None:
            raise RuntimeError("Collector must be reset before collecting chunks.")

        state = self._state
        trace = self._policy.sample_chunk(state.obs)
        action_chunk = trace["action_chunk"]
        next_obs, env_reward, done, info = self._env.step(action_chunk)
        task = next_obs.get("prompt", state.task or "")
        next_obs["prompt"] = task

        provider_result = self._provider.evaluate_chunk(
            obs_t=state.obs,
            action_chunk=action_chunk,
            obs_t_plus_1=next_obs,
            task=str(task),
            metadata={"env_info": info, "env_reward": env_reward},
        )
        reward = float(provider_result["reward"])
        next_transformed_obs = self._policy.transform_observation(next_obs)
        next_value = self._predict_value(next_obs, next_transformed_obs)

        sample = ChunkSample(
            obs_t=state.obs,
            obs_t_plus_1=next_obs,
            model_inputs_t=trace["transformed_obs"],
            action_chunk=np.asarray(action_chunk, dtype=np.float32),
            action_tokens=np.asarray(trace["action_tokens"], dtype=np.int32),
            action_token_mask=np.asarray(trace["action_token_mask"], dtype=bool),
            old_token_logprobs=np.asarray(trace["old_token_logprobs"], dtype=np.float32),
            reward=reward,
            value=float(state.value),
            next_value=next_value,
            sampled_action_tokens=np.asarray(trace["sampled_action_tokens"], dtype=np.int32),
            sampled_action_token_mask=np.asarray(trace["sampled_action_token_mask"], dtype=bool),
            sampled_dct_coeffs=np.asarray(trace["sampled_dct_coeffs"], dtype=np.float32),
            executed_dct_coeffs=np.asarray(trace["dct_coeffs"], dtype=np.float32),
            keyframe_prob=float(trace["keyframe_prob"]),
            exploration_applied=bool(trace["exploration_applied"]),
            bootstrap_mask=0.0 if done else 1.0,
            task=str(task),
            done=bool(done),
            done_reason=str(info.get("done_reason")) if isinstance(info, dict) else None,
            policy_version=policy_version,
            step_id=state.step_id,
        )

        if done:
            obs = self._env.reset()
            task2 = str(obs.get("prompt", ""))
            obs["prompt"] = task2
            transformed = self._policy.transform_observation(obs)
            self._state = _EnvState(
                obs=obs,
                transformed_obs=transformed,
                task=task2,
                value=self._predict_value(obs, transformed),
                step_id=0,
            )
        else:
            self._state = _EnvState(
                obs=next_obs,
                transformed_obs=next_transformed_obs,
                task=str(task),
                value=next_value,
                step_id=state.step_id + 1,
            )

        return [sample]
