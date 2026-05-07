from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any
from typing import Protocol

import numpy as np

from openpi_online_ppo.env.single_env_ws import SingleEnvWebsocketEnv
from openpi_online_ppo.rl.chunk_types import ChunkSample
from openpi_online_ppo.rl.pi0_fast_policy import Pi0FastRLPolicy
from openpi_online_ppo.rl.reward_value import ChunkRewardValueProvider

timing_logger = logging.getLogger("openpi_online_ppo.timing")


@dataclass
class _EnvState:
    obs: dict[str, Any]
    transformed_obs: dict[str, Any]
    task: str | None
    value: float
    phase_class: int | None = None
    subtask_class: int | None = None
    step_id: int = 0


class ValuePredictor(Protocol):
    def predict(self, transformed_obs: dict[str, Any]) -> float: ...
    def predict_phase_value(self, obs: dict[str, Any], phase_class: int) -> float: ...
    def predict_phase_value_batch(self, obs_batch: list[dict[str, Any]], phase_class: int) -> list[float]: ...
    def predict_subtask_value(self, obs: dict[str, Any], subtask_class: int) -> float: ...
    def predict_subtask_value_batch(self, obs_batch: list[dict[str, Any]], subtask_class: int) -> list[float]: ...
    def predict_keyframe_class(self, transformed_obs: dict[str, Any]) -> int: ...
    def predict_subtask_class(self, transformed_obs: dict[str, Any]) -> int: ...


class Pi0FastChunkCollector:
    """Collect chunk-level online RL samples from a single websocket env."""

    def __init__(
        self,
        *,
        env: SingleEnvWebsocketEnv,
        policy: Pi0FastRLPolicy,
        provider: ChunkRewardValueProvider,
        value_predictor: ValuePredictor | None = None,
        compute_values: bool = True,
        include_logprobs: bool = True,
        explore_phase_ids: tuple[int, ...] | None = None,
    ) -> None:
        self._env = env
        self._policy = policy
        self._provider = provider
        self._value_predictor = value_predictor
        self._compute_values = bool(compute_values)
        self._include_logprobs = bool(include_logprobs)
        self._explore_phase_ids = (
            tuple(sorted({int(x) for x in explore_phase_ids}))
            if explore_phase_ids is not None
            else None
        )
        self._state: _EnvState | None = None

    def _predict_value(self, obs: dict[str, Any], transformed_obs: dict[str, Any] | None = None) -> float:
        return self._predict_phase_value(obs, phase_class=None, transformed_obs=transformed_obs)

    def _predict_phase_value(
        self,
        obs: dict[str, Any],
        *,
        phase_class: int | None,
        transformed_obs: dict[str, Any] | None = None,
    ) -> float:
        if not self._compute_values:
            return 0.0
        if self._value_predictor is None:
            return self._policy.predict_value(obs)
        if phase_class is not None:
            return float(self._value_predictor.predict_subtask_value(obs, int(phase_class)))
        transformed_obs = transformed_obs if transformed_obs is not None else self._policy.transform_observation(obs)
        return float(self._value_predictor.predict(transformed_obs))

    def _predict_phase_class(self, obs: dict[str, Any], transformed_obs: dict[str, Any] | None = None) -> int:
        if not self._compute_values and self._value_predictor is None:
            return 0
        if self._value_predictor is None:
            return int(self._policy.predict_keyframe_class(obs))
        transformed_obs = transformed_obs if transformed_obs is not None else self._policy.transform_observation(obs)
        return int(self._value_predictor.predict_keyframe_class(transformed_obs))

    def _predict_subtask_class(self, obs: dict[str, Any], transformed_obs: dict[str, Any] | None = None) -> int:
        if not self._compute_values or self._value_predictor is None:
            return 0
        transformed_obs = transformed_obs if transformed_obs is not None else self._policy.transform_observation(obs)
        return int(self._value_predictor.predict_subtask_class(transformed_obs))

    def _chunk_window_mean_values(
        self,
        prefix_obs: list[dict[str, Any]],
        suffix_obs: list[dict[str, Any]],
        *,
        phase_class: int | None,
    ) -> tuple[float, float]:
        prefix = list(prefix_obs[:3])
        suffix = list(suffix_obs[-3:])
        merged = prefix + suffix
        if not merged:
            return 0.0, 0.0
        if self._value_predictor is not None and phase_class is not None:
            values = self._value_predictor.predict_subtask_value_batch(merged, int(phase_class))
        else:
            values = [self._predict_phase_value(obs, phase_class=phase_class) for obs in merged]
        values_np = np.asarray(values, dtype=np.float32)
        prefix_n = len(prefix)
        prefix_mean = float(np.mean(values_np[:prefix_n])) if prefix_n > 0 else 0.0
        suffix_mean = float(np.mean(values_np[prefix_n:])) if len(suffix) > 0 else 0.0
        return prefix_mean, suffix_mean

    def reset(self) -> None:
        obs = self._env.reset()
        task = obs.get("prompt")
        if task is None:
            task = ""
            obs["prompt"] = task
        transformed = self._policy.transform_observation(obs)
        phase_class = self._predict_phase_class(obs, transformed)
        subtask_class = self._predict_subtask_class(obs, transformed)
        self._state = _EnvState(
            obs=obs,
            transformed_obs=transformed,
            task=str(task),
            value=float("nan"),
            phase_class=phase_class,
            subtask_class=subtask_class,
            step_id=0,
        )

    def collect_chunk_batch(self) -> list[ChunkSample]:
        if self._state is None:
            raise RuntimeError("Collector must be reset before collecting chunks.")

        total_t0 = time.perf_counter()
        state = self._state
        phase_class = None if state.phase_class is None else int(state.phase_class)
        subtask_class = None if state.subtask_class is None else int(state.subtask_class)
        if self._explore_phase_ids is None:
            use_value = phase_class is not None and phase_class > 0
        else:
            use_value = phase_class is not None and phase_class in self._explore_phase_ids
        sample_t0 = time.perf_counter()
        trace = self._policy.sample_chunk_with_options(
            state.obs,
            include_logprobs=self._include_logprobs,
            enable_exploration=bool(use_value),
            keyframe_prob_override=1.0 if use_value else 0.0,
            use_logprob_lookup=bool(use_value),
        )
        sample_s = time.perf_counter() - sample_t0
        action_chunk = trace["action_chunk"]
        env_t0 = time.perf_counter()
        next_obs, env_reward, done, info = self._env.step(action_chunk)
        env_step_s = time.perf_counter() - env_t0
        task = next_obs.get("prompt", state.task or "")
        next_obs["prompt"] = task

        reward_t0 = time.perf_counter()
        provider_result = self._provider.evaluate_chunk(
            obs_t=state.obs,
            action_chunk=action_chunk,
            obs_t_plus_1=next_obs,
            task=str(task),
            metadata={"env_info": info, "env_reward": env_reward},
        )
        reward = float(provider_result["reward"])
        reward_s = time.perf_counter() - reward_t0
        value_t0 = time.perf_counter()
        chunk_prefix_observations = list(info.get("chunk_prefix_observations", [])) if isinstance(info, dict) else []
        chunk_suffix_observations = list(info.get("chunk_suffix_observations", [])) if isinstance(info, dict) else []
        next_transformed_obs = self._policy.transform_observation(next_obs)
        next_phase_class = self._predict_phase_class(next_obs, next_transformed_obs)
        next_subtask_class = self._predict_subtask_class(next_obs, next_transformed_obs)
        if use_value:
            value, next_value = self._chunk_window_mean_values(
                chunk_prefix_observations,
                chunk_suffix_observations,
                phase_class=state.subtask_class,
            )
        else:
            value, next_value = float("nan"), float("nan")
        value_s = time.perf_counter() - value_t0

        sample = ChunkSample(
            obs_t=state.obs,
            obs_t_plus_1=next_obs,
            model_inputs_t=trace["transformed_obs"],
            action_chunk=np.asarray(action_chunk, dtype=np.float32),
            action_tokens=np.asarray(trace["action_tokens"], dtype=np.int32),
            action_token_mask=np.asarray(trace["action_token_mask"], dtype=bool),
            old_token_logprobs=np.asarray(trace["old_token_logprobs"], dtype=np.float32),
            reward=reward,
            value=value,
            next_value=next_value,
            uses_value=bool(use_value),
            chunk_prefix_observations=chunk_prefix_observations,
            chunk_suffix_observations=chunk_suffix_observations,
            sampled_action_tokens=np.asarray(trace["sampled_action_tokens"], dtype=np.int32),
            sampled_action_token_mask=np.asarray(trace["sampled_action_token_mask"], dtype=bool),
            sampled_dct_coeffs=np.asarray(trace["sampled_dct_coeffs"], dtype=np.float32),
            executed_dct_coeffs=np.asarray(trace["dct_coeffs"], dtype=np.float32),
            keyframe_prob=float(trace["keyframe_prob"]),
            phase_class=state.phase_class,
            subtask_class=state.subtask_class,
            exploration_applied=bool(trace["exploration_applied"]),
            bootstrap_mask=0.0 if done else 1.0,
            task=str(task),
            done=bool(done),
            done_reason=str(info.get("done_reason")) if isinstance(info, dict) else None,
            step_id=state.step_id,
        )

        if done:
            obs = self._env.reset()
            task2 = str(obs.get("prompt", ""))
            obs["prompt"] = task2
            transformed = self._policy.transform_observation(obs)
            phase_class = self._predict_phase_class(obs, transformed)
            subtask_class = self._predict_subtask_class(obs, transformed)
            self._state = _EnvState(
                obs=obs,
                transformed_obs=transformed,
                task=task2,
                value=float("nan"),
                phase_class=phase_class,
                subtask_class=subtask_class,
                step_id=0,
            )
        else:
            self._state = _EnvState(
                obs=next_obs,
                transformed_obs=next_transformed_obs,
                task=str(task),
                value=float("nan"),
                phase_class=next_phase_class,
                subtask_class=next_subtask_class,
                step_id=state.step_id + 1,
            )

        if not timing_logger.disabled:
            timing_logger.info(
                json.dumps(
                    {
                        "event": "collector_collect_chunk_batch",
                        "total_s": time.perf_counter() - total_t0,
                        "sample_chunk_s": sample_s,
                        "env_step_s": env_step_s,
                        "reward_provider_s": reward_s,
                        "value_phase_s": value_s,
                        "uses_value": bool(use_value),
                        "phase_class": None if state.phase_class is None else int(state.phase_class),
                        "consumed_actions": int(info.get("consumed_actions", 0)) if isinstance(info, dict) else 0,
                        "done": bool(done),
                        "prefix_frames": int(len(chunk_prefix_observations)),
                        "suffix_frames": int(len(chunk_suffix_observations)),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )

        return [sample]
