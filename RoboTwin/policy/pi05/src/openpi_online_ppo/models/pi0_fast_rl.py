import dataclasses
import logging
from typing import Any
import time

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250

    # Tokenizer for the fast model.
    fast_model_tokenizer: Any | None = None
    # Keyword arguments for the fast model tokenizer.
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None

    # Distributional value head config.
    use_value_head: bool = False
    value_num_bins: int = 201
    value_bin_min: float = -1.0
    value_bin_max: float = 0.0
    value_hidden_dim: int | None = None
    use_keyframe_head: bool = False
    keyframe_hidden_dim: int | None = None
    # If None, defaults to action_horizon // 2 bins.
    keyframe_num_bins: int | None = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FAST":
        return Pi0FAST(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class Pi0FAST(_model.BaseModel):
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # Learnable CLS token generator for value / keyframe:
        # pass fixed one-hot [2] through a trainable linear layer.
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

    @at.typecheck
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # embed images
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * input_mask[-1])

        # add tokenized inputs
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        # return embeddings, input mask, and ar mask
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        # Compute inputs: one big forward pass of prefix + suffix at once
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )

        # Each input predicts *next* token, so we don't input the last token.
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Only decode logits for the target tokens to save memory
        # (decoding matmul is large because it is a seq_len x vocab_size dense layer).
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss on token targets
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        return self._decode_tokens(
            rng,
            observation,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
        )["tokens"]

    def _prefix_forward(
        self,
        observation: _model.Observation,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"], at.Bool[at.Array, "b s s"]]:
        observation = _model.preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
        )
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        return input_token_embeddings, input_mask, ar_mask, attn_mask

    def _extract_aux_features(
        self,
        observation: _model.Observation,
        *,
        stop_gradient: bool = False,
    ) -> tuple[at.Float[at.Array, "b emb"], at.Float[at.Array, "b emb"]]:
        input_token_embeddings, input_mask, ar_mask, _ = self._prefix_forward(observation)
        bsize = input_token_embeddings.shape[0]
        aux_onehot = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=input_token_embeddings.dtype)
        aux_emb = self.aux_cls_embed(aux_onehot)  # [2, emb]
        aux_emb = jnp.broadcast_to(aux_emb[None, :, :], (bsize, aux_emb.shape[0], aux_emb.shape[1]))
        aux_mask = jnp.ones((bsize, 2), dtype=input_mask.dtype)
        aux_ar = jnp.zeros((bsize, 2), dtype=ar_mask.dtype)
        input_token_embeddings = jnp.concatenate([input_token_embeddings, aux_emb], axis=1)
        input_mask = jnp.concatenate([input_mask, aux_mask], axis=1)
        ar_mask = jnp.concatenate([ar_mask, aux_ar], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings,
            mask=attn_mask,
            return_prelogits=True,
        )
        value_cls = pre_logits[:, -2, :]
        keyframe_cls = pre_logits[:, -1, :]
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
            raise ValueError("Value head is not enabled for this Pi0FAST model.")
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
            raise ValueError("Keyframe head is not enabled for this Pi0FAST model.")
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

    def value_bin_centers(self) -> at.Float[at.Array, "bins"]:
        config = self.config
        assert isinstance(config, Pi0FASTConfig)
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
        high_weight = jnp.clip(scaled - low.astype(jnp.float32), 0.0, 1.0)
        low_weight = 1.0 - high_weight
        target = jnp.zeros((values.shape[0], bin_centers.shape[0]), dtype=jnp.float32)
        target = target.at[jnp.arange(values.shape[0]), low].add(low_weight)
        target = target.at[jnp.arange(values.shape[0]), high].add(high_weight)
        return target

    def sample_actions_with_trace(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
        selected_vocab_indices: at.Int[at.Array, "k"] | None = None,
    ) -> dict[str, at.Array]:
        """Sample autoregressive output tokens and expose per-token decoding trace.

        This is an additive RL-oriented API and does not affect the default policy
        / eval / SFT paths, which still call `sample_actions()`.
        """
        return self._decode_tokens(
            rng,
            observation,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
            selected_vocab_indices=selected_vocab_indices,
        )

    def recompute_action_logprobs(
        self,
        observation: _model.Observation,
        action_tokens: at.Int[at.Array, "b l"],
        *,
        action_token_mask: at.Bool[at.Array, "b l"] | None = None,
    ) -> dict[str, at.Array]:
        """Teacher-force a generated suffix and recompute token log-probs.

        The `action_tokens` are the autoregressively generated suffix tokens from
        `sample_actions_with_trace()`, i.e. the same tokens the model would return
        from `sample_actions()`. This method is intended for PPO replay and is not
        used by the existing eval/SFT codepaths.
        """
        observation = _model.preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
        )

        if action_token_mask is None:
            action_token_mask = jnp.ones_like(action_tokens, dtype=jnp.bool_)

        decode_state = self._prepare_decode_prefix(observation, action_tokens.shape[1])
        last_logit = decode_state["last_logit"]
        kv_cache = decode_state["kv_cache"]
        prefill_len = decode_state["prefill_len"]
        prefill_size = decode_state["prefill_size"]
        prefix_start = decode_state["prefix_start"]

        action_tokens = jnp.asarray(action_tokens, dtype=jnp.int32)
        action_token_mask = jnp.asarray(action_token_mask, dtype=jnp.bool_)

        def step(carry, token_and_mask):
            current_logit, cache, step_idx = carry
            token, token_mask = token_and_mask

            logp_all = jax.nn.log_softmax(current_logit[:, 0, :], axis=-1)
            chosen_logp = jnp.take_along_axis(logp_all, token[:, None], axis=-1)[:, 0]
            entropy = -jnp.sum(jnp.exp(logp_all) * logp_all, axis=-1)

            chosen_logp = jnp.where(token_mask, chosen_logp, 0.0)
            entropy = jnp.where(token_mask, entropy, 0.0)

            token_for_decode = jnp.where(token_mask, token, PALIGEMMA_EOS_TOKEN)
            token_embedding = self.PaliGemma.llm(token_for_decode[:, None], embed_only=True)
            positions = prefill_len[:, None] + step_idx + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + action_tokens.shape[1])[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + action_tokens.shape[1])[None, None, :]
                < jnp.broadcast_to(prefill_size + step_idx + 1, (prefix_start.shape[0], 1, 1)),
            )
            next_logit, next_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding,
                mask=mask,
                positions=positions,
                decode=True,
                kv_cache=cache,
            )
            return (next_logit, next_cache, step_idx + 1), {
                "token_logprobs": chosen_logp,
                "token_entropy": entropy,
            }

        _, outputs = jax.lax.scan(
            step,
            (last_logit, kv_cache, 0),
            (jnp.swapaxes(action_tokens, 0, 1), jnp.swapaxes(action_token_mask, 0, 1)),
        )

        return {
            "token_logprobs": jnp.swapaxes(outputs["token_logprobs"], 0, 1),
            "token_entropy": jnp.swapaxes(outputs["token_entropy"], 0, 1),
            "action_token_mask": action_token_mask,
        }

    def _prepare_decode_prefix(
        self,
        observation: _model.Observation,
        max_decoding_steps: int | at.Int[at.Array, ""],
    ) -> dict[str, at.Array]:
        # TODO: this is a hack to get the image keys.
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images.keys()))

        # embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )
        return {
            "last_logit": prefix_logits[:, -1:],
            "kv_cache": kv_cache,
            "prefill_size": prefill_size,
            "prefill_len": prefill_len,
            "prefix_start": prefix_start,
        }

    def _decode_tokens(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
        selected_vocab_indices: at.Int[at.Array, "k"] | None = None,
    ) -> dict[str, at.Array]:
        t0 = time.perf_counter()
        decode_state = self._prepare_decode_prefix(observation, max_decoding_steps)
        prepare_decode_prefix_s = time.perf_counter() - t0
        last_logit = decode_state["last_logit"]
        kv_cache = decode_state["kv_cache"]
        prefill_size = decode_state["prefill_size"]
        prefill_len = decode_state["prefill_len"]
        prefix_start = decode_state["prefix_start"]

        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)
        output_token_logprobs = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=last_logit.dtype)
        output_token_mask = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.bool_)
        if selected_vocab_indices is not None:
            selected_vocab_indices = jnp.asarray(selected_vocab_indices, dtype=jnp.int32).reshape(-1)
            output_selected_logprobs = jnp.zeros(
                (last_logit.shape[0], max_decoding_steps, selected_vocab_indices.shape[0]),
                dtype=last_logit.dtype,
            )
        else:
            output_selected_logprobs = jnp.zeros((last_logit.shape[0], 0, 0), dtype=last_logit.dtype)

        def step(carry):
            (
                rng,
                last_logit,
                output_tokens,
                output_token_logprobs,
                output_token_mask,
                output_selected_logprobs,
                cache,
                finished,
                step,
            ) = carry

            rng, rng_step = jax.random.split(rng)
            logp_all = jax.nn.log_softmax(last_logit[:, 0, :], axis=-1)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit[:, 0, :] / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit[:, 0, :], axis=-1),
                operand=None,
            )
            token = jnp.where(finished, PALIGEMMA_EOS_TOKEN, token)
            valid_token = ~finished
            chosen_logp = jnp.take_along_axis(logp_all, token[:, None], axis=-1)[:, 0]

            output_tokens = put_along_last_axis(
                output_tokens,
                jnp.broadcast_to(step, (token.shape[0], 1)),
                token[:, None],
            )
            output_token_logprobs = put_along_last_axis(
                output_token_logprobs,
                jnp.broadcast_to(step, (token.shape[0], 1)),
                jnp.where(valid_token, chosen_logp, 0.0)[:, None],
            )
            output_token_mask = put_along_last_axis(
                output_token_mask,
                jnp.broadcast_to(step, (token.shape[0], 1)),
                valid_token[:, None],
            )
            if selected_vocab_indices is not None:
                selected_logp = jnp.take(logp_all, selected_vocab_indices, axis=-1)
                output_selected_logprobs = output_selected_logprobs.at[:, step, :].set(selected_logp)

            finished = jnp.logical_or(finished, token == PALIGEMMA_EOS_TOKEN)
            token_embedding = self.PaliGemma.llm(token[:, None], embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return (
                rng,
                last_logit,
                output_tokens,
                output_token_logprobs,
                output_token_mask,
                output_selected_logprobs,
                kv_cache,
                finished,
                step + 1,
            )

        def cond(carry):
            _, _, _, _, _, _, _, finished, step = carry
            return jnp.logical_and(jnp.logical_not(jnp.all(finished)), step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        loop_t0 = time.perf_counter()
        (
            _,
            _,
            output_tokens,
            output_token_logprobs,
            output_token_mask,
            output_selected_logprobs,
            _,
            _,
            _,
        ) = jax.lax.while_loop(
            cond,
            step,
            (
                rng,
                last_logit,
                output_tokens,
                output_token_logprobs,
                output_token_mask,
                output_selected_logprobs,
                kv_cache,
                jnp.zeros((last_logit.shape[0],), dtype=jnp.bool_),
                0,
            ),
        )
        decode_loop_s = time.perf_counter() - loop_t0
        out = {
            "tokens": output_tokens,
            "token_logprobs": output_token_logprobs,
            "token_mask": output_token_mask,
            "timings": {
                "prepare_decode_prefix_s": prepare_decode_prefix_s,
                "decode_loop_s": decode_loop_s,
            },
        }
        if selected_vocab_indices is not None:
            out["selected_token_logprobs"] = output_selected_logprobs
        return out
