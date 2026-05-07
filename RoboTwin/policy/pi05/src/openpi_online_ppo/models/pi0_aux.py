import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0AuxConfig(pi0_config.Pi0Config):
    use_value_head: bool = False
    value_num_bins: int = 201
    value_bin_min: float = -1.0
    value_bin_max: float = 0.0
    value_hidden_dim: int | None = None
    use_keyframe_head: bool = False
    keyframe_hidden_dim: int | None = None
    keyframe_num_bins: int | None = None
    use_subtask_head: bool = False
    subtask_hidden_dim: int | None = None
    subtask_num_bins: int | None = None

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Aux":
        return Pi0Aux(self, rngs=nnx.Rngs(rng))


class Pi0Aux(_pi0.Pi0):
    def __init__(self, config: Pi0AuxConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.config = config
        paligemma_config = self.PaliGemma.llm.module.configs[0]
        self.aux_cls_embed = nnx.Linear(2, paligemma_config.width, use_bias=False, rngs=rngs)
        self.value_head = None
        if config.use_value_head:
            hidden_dim = int(config.value_hidden_dim or paligemma_config.width)
            self.value_head = nnx.Dict(
                proj_in=nnx.Linear(paligemma_config.width, hidden_dim, rngs=rngs),
                proj_out=nnx.Linear(hidden_dim, config.value_num_bins, rngs=rngs),
            )
        self.keyframe_head = None
        if config.use_keyframe_head:
            hidden_dim = int(config.keyframe_hidden_dim or paligemma_config.width)
            keyframe_num_bins = int(config.keyframe_num_bins or max(2, config.action_horizon // 2))
            self.keyframe_head = nnx.Dict(
                proj_in=nnx.Linear(paligemma_config.width, hidden_dim, rngs=rngs),
                proj_out=nnx.Linear(hidden_dim, keyframe_num_bins, rngs=rngs),
            )
        self.subtask_head = None
        if config.use_subtask_head:
            hidden_dim = int(config.subtask_hidden_dim or paligemma_config.width)
            subtask_num_bins = int(config.subtask_num_bins or max(2, config.action_horizon // 2))
            self.subtask_head = nnx.Dict(
                proj_in=nnx.Linear(paligemma_config.width, hidden_dim, rngs=rngs),
                proj_out=nnx.Linear(hidden_dim, subtask_num_bins, rngs=rngs),
            )

    def _extract_aux_features(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> tuple[at.Float[at.Array, "b emb"], at.Float[at.Array, "b emb"]]:
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, input_mask, ar_mask = self.embed_prefix(observation)
        bsize = prefix_tokens.shape[0]
        aux_onehot = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=prefix_tokens.dtype)
        aux_emb = self.aux_cls_embed(aux_onehot)
        aux_emb = jnp.broadcast_to(aux_emb[None, :, :], (bsize, aux_emb.shape[0], aux_emb.shape[1]))
        aux_mask = jnp.ones((bsize, 2), dtype=input_mask.dtype)
        aux_ar = jnp.zeros((2,), dtype=ar_mask.dtype)
        prefix_tokens = jnp.concatenate([prefix_tokens, aux_emb], axis=1)
        input_mask = jnp.concatenate([input_mask, aux_mask], axis=1)
        ar_mask = jnp.concatenate([ar_mask, aux_ar], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        value_cls = prefix_out[:, -2, :]
        keyframe_cls = prefix_out[:, -1, :]
        if stop_gradient:
            value_cls = jax.lax.stop_gradient(value_cls)
            keyframe_cls = jax.lax.stop_gradient(keyframe_cls)
        return value_cls, keyframe_cls

    def predict_value_logits(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> at.Float[at.Array, "b bins"]:
        if self.value_head is None:
            raise ValueError("Value head is not enabled for this Pi0Aux model.")
        value_features, _ = self._extract_aux_features(observation, stop_gradient=stop_gradient)
        hidden = self.value_head.proj_in(value_features)
        hidden = jax.nn.gelu(hidden)
        return self.value_head.proj_out(hidden)

    def predict_value(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> at.Float[at.Array, "b"]:
        logits = self.predict_value_logits(observation, stop_gradient=stop_gradient)
        centers = self.value_bin_centers()
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * centers[None, :], axis=-1)

    def predict_keyframe_logits(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> at.Float[at.Array, "b bins"]:
        if self.keyframe_head is None:
            raise ValueError("Keyframe head is not enabled for this Pi0Aux model.")
        _, keyframe_features = self._extract_aux_features(observation, stop_gradient=stop_gradient)
        hidden = self.keyframe_head.proj_in(keyframe_features)
        hidden = jax.nn.gelu(hidden)
        return self.keyframe_head.proj_out(hidden)

    def predict_keyframe_prob(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> at.Float[at.Array, "b"]:
        logits = self.predict_keyframe_logits(observation, stop_gradient=stop_gradient)
        probs = jax.nn.softmax(logits, axis=-1)
        if logits.shape[-1] == 2:
            return probs[:, 1]
        return jnp.sum(probs[:, 1:], axis=-1)

    def predict_keyframe_class(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> at.Int[at.Array, "b"]:
        logits = self.predict_keyframe_logits(observation, stop_gradient=stop_gradient)
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    def predict_subtask_logits(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> at.Float[at.Array, "b bins"]:
        if self.subtask_head is None:
            raise ValueError("Subtask head is not enabled for this Pi0Aux model.")
        _, subtask_features = self._extract_aux_features(observation, stop_gradient=stop_gradient)
        hidden = self.subtask_head.proj_in(subtask_features)
        hidden = jax.nn.gelu(hidden)
        return self.subtask_head.proj_out(hidden)

    def predict_subtask_prob(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> at.Float[at.Array, "b"]:
        logits = self.predict_subtask_logits(observation, stop_gradient=stop_gradient)
        probs = jax.nn.softmax(logits, axis=-1)
        if logits.shape[-1] == 2:
            return probs[:, 1]
        return jnp.sum(probs[:, 1:], axis=-1)

    def predict_subtask_class(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> at.Int[at.Array, "b"]:
        logits = self.predict_subtask_logits(observation, stop_gradient=stop_gradient)
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    def value_bin_centers(self) -> at.Float[at.Array, "bins"]:
        config = self.config
        assert isinstance(config, Pi0AuxConfig)
        return jnp.linspace(config.value_bin_min, config.value_bin_max, config.value_num_bins, dtype=jnp.float32)

    def project_values_to_bins(
        self,
        values: at.Float[at.Array, "b"],
    ) -> at.Float[at.Array, "b bins"]:
        bin_centers = self.value_bin_centers()
        values = jnp.clip(values, bin_centers[0], bin_centers[-1])
        step = bin_centers[1] - bin_centers[0]
        scaled = (values - bin_centers[0]) / step
        low = jnp.floor(scaled).astype(jnp.int32)
        high = jnp.clip(low + 1, 0, bin_centers.shape[0] - 1)
        frac = scaled - low.astype(scaled.dtype)
        low_w = 1.0 - frac
        high_w = frac
        target = jnp.zeros((values.shape[0], bin_centers.shape[0]), dtype=jnp.float32)
        target = target.at[jnp.arange(values.shape[0]), low].add(low_w.astype(jnp.float32))
        target = target.at[jnp.arange(values.shape[0]), high].add(high_w.astype(jnp.float32))
        return target
