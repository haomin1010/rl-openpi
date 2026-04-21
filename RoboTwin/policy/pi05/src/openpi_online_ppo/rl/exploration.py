from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import pathlib
import pickle
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class KeyframeExplorationConfig:
    # Scheduling mode:
    # - "always": always explore when backend is enabled
    # - "conditional_keyframe": explore only when keyframe prob is high enough
    # - "never": never explore
    mode: str = "always"
    # Kept for backward compatibility with legacy callers.
    always_on: bool = True
    # Legacy flag kept for compatibility; backend may ignore this when operating in action space.
    explore_dct_dims: int = 0
    relaxed_fast_decoding: bool = True
    noise_std: float = 0.01
    keyframe_prob_threshold: float = 0.5

    # Perturbation backend:
    # - "dct_gaussian": legacy dct-space gaussian
    # - "action_formula": isotropic action-space perturbation with bounded radius
    # - "action_network": linear network predicts direction/scale in action space
    # - "action_hybrid": formula + network mixture
    perturb_backend: str = "dct_gaussian"

    # Action-space magnitude constraints (for action_* backends).
    action_radius_min: float = 0.0
    action_radius_max: float = 0.15
    action_abs_clip: float = 1.0
    # Optional whitelist of action dimensions that are allowed to be perturbed.
    # None means all action dimensions.
    action_noise_indices: tuple[int, ...] | None = None

    # Hybrid/network controls.
    network_mix_alpha: float = 0.5
    network_checkpoint: str | None = None
    network_obs_dim: int = 32
    network_latent_dim: int = 16


class DCTNoiseFn(Protocol):
    def __call__(
        self,
        *,
        dct_coeffs: np.ndarray,
        num_noisy_dims: int,
        obs: dict[str, Any],
        keyframe_prob: float,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Return perturbed DCT coefficients."""


class CallableDCTNoiseFn:
    def __init__(self, fn: Callable[..., np.ndarray]):
        self._fn = fn

    def __call__(
        self,
        *,
        dct_coeffs: np.ndarray,
        num_noisy_dims: int,
        obs: dict[str, Any],
        keyframe_prob: float,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self._fn(
                dct_coeffs=np.asarray(dct_coeffs),
                num_noisy_dims=int(num_noisy_dims),
                obs=obs,
                keyframe_prob=float(keyframe_prob),
                metadata=metadata,
            ),
            dtype=np.float32,
        )


def _normalize_l2(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < eps:
        return np.zeros_like(vec)
    return vec / n


def _uniform_direction(shape: tuple[int, ...]) -> np.ndarray:
    v = np.random.normal(0.0, 1.0, size=shape).astype(np.float32)
    return _normalize_l2(v)


def _extract_obs_state(obs: dict[str, Any], target_dim: int) -> np.ndarray:
    state = np.asarray(obs.get("state", []), dtype=np.float32).reshape(-1)
    if state.size >= target_dim:
        return state[:target_dim]
    out = np.zeros((target_dim,), dtype=np.float32)
    if state.size > 0:
        out[: state.size] = state
    return out


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _build_ortho_dct_matrix(n: int) -> np.ndarray:
    # DCT-II orthonormal matrix.
    if n <= 0:
        raise ValueError("n must be positive")
    k = np.arange(n, dtype=np.float32)[:, None]
    t = np.arange(n, dtype=np.float32)[None, :]
    mat = np.sqrt(2.0 / n) * np.cos(np.pi * (t + 0.5) * k / n)
    mat[0, :] = np.sqrt(1.0 / n)
    return mat.astype(np.float32)


class DirectionRadiusActionPerturbNet:
    """MLP perturbation net that predicts direction + radius in DCT first-k subspace.

    It does not output sigma; stochasticity comes from latent z in the input.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        out_dim: int,
        w: list[np.ndarray],
        b: list[np.ndarray],
        x_mean: np.ndarray | None,
        x_std: np.ndarray | None,
        chunk_size: int,
        action_dim: int,
        dct_k: int,
        radius_min: float,
        radius_max: float,
        latent_dim: int,
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_depth = int(hidden_depth)
        self.out_dim = int(out_dim)
        self.w = [np.asarray(x, dtype=np.float32) for x in w]
        self.b = [np.asarray(x, dtype=np.float32) for x in b]
        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1) if x_mean is not None else None
        self.x_std = np.asarray(x_std, dtype=np.float32).reshape(1, -1) if x_std is not None else None
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.dct_k = int(dct_k)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.latent_dim = int(latent_dim)
        self._dct_mat = _build_ortho_dct_matrix(self.chunk_size)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "DirectionRadiusActionPerturbNet":
        ckpt_path = pathlib.Path(path)
        with ckpt_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid perturb net checkpoint format at `{path}`.")
        model_type = str(payload.get("model_type", ""))
        if model_type != "direction_radius_mlp_jax_v1":
            raise ValueError(f"Unsupported perturb net checkpoint `{path}` model_type={model_type}.")
        meta = dict(payload.get("meta", {}))
        layers = payload.get("layers")
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError(f"Invalid layers in perturb net checkpoint `{path}`.")
        w = [np.asarray(layer["w"], dtype=np.float32) for layer in layers]
        b = [np.asarray(layer["b"], dtype=np.float32) for layer in layers]
        return cls(
            in_dim=int(meta["in_dim"]),
            hidden_dim=int(meta["hidden_dim"]),
            hidden_depth=int(meta["hidden_depth"]),
            out_dim=int(meta["out_dim"]),
            w=w,
            b=b,
            x_mean=np.asarray(payload.get("x_mean"), dtype=np.float32) if payload.get("x_mean") is not None else None,
            x_std=np.asarray(payload.get("x_std"), dtype=np.float32) if payload.get("x_std") is not None else None,
            chunk_size=int(meta["chunk_size"]),
            action_dim=int(meta["action_dim"]),
            dct_k=int(meta["dct_k"]),
            radius_min=float(meta["radius_min"]),
            radius_max=float(meta["radius_max"]),
            latent_dim=int(meta["latent_dim"]),
        )

    def _mlp(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32).reshape(-1)
        if h.shape[0] != self.in_dim:
            raise ValueError(f"network input dim mismatch: got {h.shape[0]}, expect {self.in_dim}")
        if self.x_mean is not None and self.x_std is not None:
            h = ((h.reshape(1, -1) - self.x_mean) / (self.x_std + 1e-6)).reshape(-1)
        for i in range(len(self.w)):
            h = h @ self.w[i] + self.b[i]
            if i < len(self.w) - 1:
                h = _gelu(h)
        return h.astype(np.float32)

    def _action_to_dct(self, actions: np.ndarray) -> np.ndarray:
        # actions: [T, A], dct: [T, A]
        return self._dct_mat @ np.asarray(actions, dtype=np.float32)

    def _dct_to_action(self, dct_coeffs: np.ndarray) -> np.ndarray:
        return self._dct_mat.T @ np.asarray(dct_coeffs, dtype=np.float32)

    def sample_delta(self, *, actions: np.ndarray, keyframe_prob: float, obs: dict[str, Any]) -> np.ndarray:
        del obs
        act = np.asarray(actions, dtype=np.float32)
        if act.ndim != 2:
            raise ValueError(f"Expected actions [T,A], got {act.shape}")
        t, a = act.shape
        if t != self.chunk_size or a != self.action_dim:
            raise ValueError(
                f"Action shape mismatch for perturb net. Got {(t, a)}, expect {(self.chunk_size, self.action_dim)}"
            )
        dct = self._action_to_dct(act)
        dct_firstk = dct[: self.dct_k, :].reshape(-1)
        z = np.random.normal(0.0, 1.0, size=(self.latent_dim,)).astype(np.float32)
        x = np.concatenate([dct_firstk, np.asarray([float(keyframe_prob)], dtype=np.float32), z], axis=0)
        out = self._mlp(x)
        dir_raw = out[: self.out_dim]
        radius_logit = float(out[self.out_dim])
        direction = _normalize_l2(dir_raw)
        base_radius = self.radius_min + (self.radius_max - self.radius_min) * (1.0 / (1.0 + np.exp(-radius_logit)))
        # Keep compatibility with keyframe-gated exploration strength.
        radius = float(base_radius * np.clip(keyframe_prob, 0.0, 1.0))
        delta_flat = (direction * radius).astype(np.float32)
        delta_dct = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        delta_dct[: self.dct_k, :] = delta_flat.reshape(self.dct_k, self.action_dim)
        delta_act = self._dct_to_action(delta_dct)
        return np.asarray(delta_act, dtype=np.float32)


def load_action_perturb_net(path: str | pathlib.Path) -> DirectionRadiusActionPerturbNet:
    return DirectionRadiusActionPerturbNet.load(path)


def _extract_obs_by_dotted_key(obs: dict[str, Any], dotted_key: str) -> np.ndarray:
    node: Any = obs
    for p in dotted_key.split("."):
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return np.zeros((1,), dtype=np.float32)
    arr = np.asarray(node, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return arr


class LinearKeyframeNet:
    """Small linear binary classifier for keyframe gating."""

    def __init__(
        self,
        *,
        in_dim: int,
        w: np.ndarray,
        b: np.ndarray,
        obs_key: str = "state",
        x_mean: np.ndarray | None = None,
        x_std: np.ndarray | None = None,
    ):
        self.in_dim = int(in_dim)
        self.w = np.asarray(w, dtype=np.float32).reshape(self.in_dim)
        self.b = np.asarray(b, dtype=np.float32).reshape(1)
        self.obs_key = str(obs_key)
        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1) if x_mean is not None else None
        self.x_std = np.asarray(x_std, dtype=np.float32).reshape(1, -1) if x_std is not None else None

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "LinearKeyframeNet":
        d = np.load(path, allow_pickle=False)
        return cls(
            in_dim=int(np.asarray(d["in_dim"]).reshape(-1)[0]),
            w=d["w"],
            b=d["b"],
            obs_key=str(d["obs_key"]) if "obs_key" in d else "state",
            x_mean=d["x_mean"] if "x_mean" in d else None,
            x_std=d["x_std"] if "x_std" in d else None,
        )

    def _project_obs(self, obs: dict[str, Any]) -> np.ndarray:
        raw = _extract_obs_by_dotted_key(obs, self.obs_key)
        x = np.zeros((self.in_dim,), dtype=np.float32)
        x[: min(self.in_dim, raw.shape[0])] = raw[: min(self.in_dim, raw.shape[0])]
        if self.x_mean is not None and self.x_std is not None:
            x = ((x.reshape(1, -1) - self.x_mean) / (self.x_std + 1e-6)).reshape(-1)
        return x

    def predict_logit(self, obs: dict[str, Any]) -> float:
        x = self._project_obs(obs)
        return float(x @ self.w + self.b[0])

    def predict_prob(self, obs: dict[str, Any]) -> float:
        logit = self.predict_logit(obs)
        return float(1.0 / (1.0 + np.exp(-logit)))


def _action_formula_delta(
    actions: np.ndarray,
    *,
    keyframe_prob: float,
    cfg: KeyframeExplorationConfig,
) -> np.ndarray:
    flat = actions.reshape(-1)
    direction = _uniform_direction(flat.shape)
    kp = float(np.clip(keyframe_prob, 0.0, 1.0))
    radius = float(cfg.action_radius_min + (cfg.action_radius_max - cfg.action_radius_min) * kp)
    return (direction * radius).reshape(actions.shape).astype(np.float32)


def _apply_action_dim_mask(delta: np.ndarray, cfg: KeyframeExplorationConfig) -> np.ndarray:
    allowed = getattr(cfg, "action_noise_indices", None)
    if allowed is None:
        return np.asarray(delta, dtype=np.float32)
    out = np.asarray(delta, dtype=np.float32).copy()
    if out.ndim < 2:
        return out
    action_dim = out.shape[-1]
    mask = np.zeros((action_dim,), dtype=np.float32)
    for idx in allowed:
        i = int(idx)
        if 0 <= i < action_dim:
            mask[i] = 1.0
    out = out * mask[None, ...]
    return out


def _action_network_delta(
    actions: np.ndarray,
    *,
    obs: dict[str, Any],
    keyframe_prob: float,
    cfg: KeyframeExplorationConfig,
    net: DirectionRadiusActionPerturbNet,
) -> np.ndarray:
    del cfg
    return net.sample_delta(actions=np.asarray(actions, dtype=np.float32), keyframe_prob=keyframe_prob, obs=obs)


def default_dct_noise(
    *,
    dct_coeffs: np.ndarray,
    num_noisy_dims: int,
    obs: dict[str, Any],
    keyframe_prob: float,
    metadata: dict[str, Any] | None = None,
    noise_std: float = 0.01,
    cfg: KeyframeExplorationConfig | None = None,
) -> np.ndarray:
    """Unified perturbation entrypoint.

    For action-space backends this expects metadata to provide:
    - `decode_dct_to_actions`: Callable[[np.ndarray], np.ndarray]
    - `encode_actions_to_dct`: Callable[[np.ndarray], np.ndarray]
    - optional `perturb_net`: LinearActionPerturbNet
    """

    cfg = cfg or KeyframeExplorationConfig()
    coeffs = np.asarray(dct_coeffs, dtype=np.float32).copy()
    backend = str(getattr(cfg, "perturb_backend", "dct_gaussian")).lower()

    if backend == "dct_gaussian":
        if num_noisy_dims <= 0:
            return coeffs
        dims = min(int(num_noisy_dims), coeffs.shape[-1])
        coeffs[..., :dims] += np.random.normal(0.0, noise_std, size=coeffs[..., :dims].shape).astype(np.float32)
        return coeffs

    metadata = metadata or {}
    decode_fn = metadata.get("decode_dct_to_actions")
    encode_fn = metadata.get("encode_actions_to_dct")
    if decode_fn is None or encode_fn is None:
        # Fallback to legacy dct gaussian if action-space mapping is unavailable.
        if num_noisy_dims <= 0:
            return coeffs
        dims = min(int(num_noisy_dims), coeffs.shape[-1])
        coeffs[..., :dims] += np.random.normal(0.0, noise_std, size=coeffs[..., :dims].shape).astype(np.float32)
        return coeffs

    actions = np.asarray(decode_fn(coeffs), dtype=np.float32)
    formula_delta = _apply_action_dim_mask(_action_formula_delta(actions, keyframe_prob=keyframe_prob, cfg=cfg), cfg)
    net: DirectionRadiusActionPerturbNet | None = metadata.get("perturb_net")

    if backend == "action_formula":
        delta = formula_delta
    elif backend == "action_network":
        if net is None:
            delta = formula_delta
        else:
            delta = _action_network_delta(actions, obs=obs, keyframe_prob=keyframe_prob, cfg=cfg, net=net)
    elif backend == "action_hybrid":
        if net is None:
            delta = formula_delta
        else:
            net_delta = _action_network_delta(actions, obs=obs, keyframe_prob=keyframe_prob, cfg=cfg, net=net)
            alpha = float(np.clip(cfg.network_mix_alpha, 0.0, 1.0))
            delta = (1.0 - alpha) * formula_delta + alpha * net_delta
    else:
        delta = formula_delta

    delta = _apply_action_dim_mask(delta, cfg)
    pert_actions = np.asarray(actions + delta, dtype=np.float32)
    if cfg.action_abs_clip > 0:
        pert_actions = np.clip(pert_actions, -float(cfg.action_abs_clip), float(cfg.action_abs_clip)).astype(np.float32)
    return np.asarray(encode_fn(pert_actions), dtype=np.float32)


def maybe_apply_dct_exploration(
    *,
    dct_coeffs: np.ndarray,
    cfg: KeyframeExplorationConfig,
    keyframe_prob: float,
    noise_fn: DCTNoiseFn | None,
    obs: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool]:
    coeffs = np.asarray(dct_coeffs, dtype=np.float32)
    if noise_fn is None:
        return coeffs, False
    mode = str(getattr(cfg, "mode", "always")).lower()
    if mode == "never":
        return coeffs, False
    if mode == "conditional_keyframe" and float(keyframe_prob) < float(cfg.keyframe_prob_threshold):
        return coeffs, False
    if mode not in {"always", "conditional_keyframe", "never", "legacy"}:
        if not bool(getattr(cfg, "always_on", True)):
            return coeffs, False
    updated = noise_fn(
        dct_coeffs=coeffs,
        num_noisy_dims=min(int(cfg.explore_dct_dims), coeffs.shape[-1]),
        obs=obs,
        keyframe_prob=float(keyframe_prob),
        metadata=metadata,
    )
    return np.asarray(updated, dtype=np.float32), True
