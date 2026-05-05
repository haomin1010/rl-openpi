import logging
import os
from typing import Literal

import jax
import numpy as np
import orbax.checkpoint as ocp
from scipy.fft import dct
from scipy.fft import idct
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = "physical-intelligence/fast",
        transpose_dct_before_bpe: bool = False,
        rowwise_bpe: bool = False,
        rowwise_layout: Literal["action_dim_major", "time_major"] = "action_dim_major",
    ):
        self._max_len = max_len
        self._transpose_dct_before_bpe = transpose_dct_before_bpe
        self._rowwise_bpe = bool(rowwise_bpe)
        self._rowwise_layout = str(rowwise_layout)
        if self._rowwise_layout not in {"action_dim_major", "time_major"}:
            raise ValueError(f"Unsupported rowwise_layout: {rowwise_layout}")

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        self._action_prefix_tokens = np.asarray(self._paligemma_tokenizer.encode("Action: "), dtype=np.int32)
        bar_tokens = self._paligemma_tokenizer.encode("|", add_eos=False)
        if len(bar_tokens) == 0:
            raise ValueError("Failed to tokenize '|' separator for FAST action parsing.")
        self._action_sep_token = int(bar_tokens[0])

    def _infer_fast_vocab_size(self) -> int:
        bpe = self._fast_tokenizer.bpe_tokenizer
        for attr in ("get_vocab_size", "vocab_size"):
            candidate = getattr(bpe, attr, None)
            if callable(candidate):
                try:
                    return int(candidate())
                except TypeError:
                    pass
            elif candidate is not None:
                return int(candidate)
        get_vocab = getattr(bpe, "get_vocab", None)
        if callable(get_vocab):
            vocab = get_vocab()
            if isinstance(vocab, dict):
                return int(len(vocab))
        raise AttributeError("Unable to infer FAST BPE vocabulary size from tokenizer.")

    @property
    def rowwise_bpe(self) -> bool:
        return bool(self._rowwise_bpe)

    @property
    def rowwise_layout(self) -> str:
        return str(self._rowwise_layout)

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            action_tokens_in_pg = self._encode_actions_to_pg_tokens(actions)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        dct_coeffs = self.extract_action_dct_coeffs(
            tokens,
            action_horizon=action_horizon,
            action_dim=action_dim,
            relaxed_decoding=True,
        )
        return self.decode_action_dct_coeffs(dct_coeffs)

    def extract_action_dct_coeffs(
        self,
        tokens: np.ndarray,
        *,
        action_horizon: int,
        action_dim: int,
        relaxed_decoding: bool = True,
    ) -> np.ndarray:
        action_tokens_in_pg = self.extract_action_tokens(tokens)
        return self.decode_action_tokens_to_dct_coeffs(
            action_tokens_in_pg,
            action_horizon=action_horizon,
            action_dim=action_dim,
            relaxed_decoding=relaxed_decoding,
        )

    def extract_action_tokens(self, tokens: np.ndarray) -> np.ndarray:
        toks = np.asarray(tokens, dtype=np.int32).reshape(-1)
        n = toks.shape[0]
        prefix = self._action_prefix_tokens
        m = prefix.shape[0]
        if n < m:
            return np.asarray([], dtype=np.int32)

        start = -1
        for i in range(0, n - m + 1):
            if np.array_equal(toks[i : i + m], prefix):
                start = i + m
                break
        if start < 0:
            return np.asarray([], dtype=np.int32)

        end = n
        for j in range(start, n):
            if int(toks[j]) == self._action_sep_token:
                end = j
                break
        if end <= start:
            return np.asarray([], dtype=np.int32)
        return toks[start:end].astype(np.int32, copy=False)

    def decode_action_tokens_to_dct_coeffs(
        self,
        action_tokens_in_pg_vocab: np.ndarray,
        *,
        action_horizon: int,
        action_dim: int,
        relaxed_decoding: bool = True,
    ) -> np.ndarray:
        action_tokens_pg = np.asarray(action_tokens_in_pg_vocab, dtype=np.int32).reshape(-1)
        if action_tokens_pg.size == 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        fast_tokens = self._pg_tokens_to_fast_tokens(action_tokens_pg)
        try:
            decoded_bpe = self._fast_tokenizer.bpe_tokenizer.decode(fast_tokens.tolist())
            decoded_dct_coeff = np.array(list(map(ord, decoded_bpe)), dtype=np.float32) + float(self._fast_tokenizer.min_token)
        except Exception as exc:
            logging.warning(
                "FAST BPE decode failed: %s | fast_tokens_len=%d fast_preview=%s",
                exc,
                int(fast_tokens.shape[0]),
                fast_tokens[: min(32, fast_tokens.shape[0])].tolist(),
            )
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        expected_seq_len = action_horizon * action_dim
        diff = expected_seq_len - int(decoded_dct_coeff.shape[0])
        if relaxed_decoding:
            if diff < 0:
                decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]
            elif diff > 0:
                decoded_dct_coeff = np.pad(decoded_dct_coeff, (0, diff), mode="constant", constant_values=0)
        elif diff != 0:
            raise ValueError(
                f"Decoded DCT coefficients have length {decoded_dct_coeff.shape[0]}, expected {expected_seq_len}."
            )

        decoded_dct_coeff = self._deserialize_dct_coeffs(
            decoded_dct_coeff,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        if decoded_dct_coeff.shape != (action_horizon, action_dim):
            raise ValueError(
                f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({action_horizon}, {action_dim})."
        )
        return decoded_dct_coeff.astype(np.float32, copy=False)

    def action_prefix_tokens(self) -> np.ndarray:
        return np.asarray(self._action_prefix_tokens, dtype=np.int32).copy()

    def action_suffix_tokens(self, *, add_eos: bool = True) -> np.ndarray:
        return np.asarray(self._paligemma_tokenizer.encode("|", add_eos=add_eos), dtype=np.int32)

    def fast_vocab_size(self) -> int:
        return int(self._infer_fast_vocab_size())

    def action_pg_token_ids(self) -> np.ndarray:
        return self._act_tokens_to_paligemma_tokens(np.arange(self.fast_vocab_size(), dtype=np.int32))

    def pg_tokens_to_fast_tokens(self, pg_tokens: np.ndarray | list[int]) -> np.ndarray:
        return np.asarray(self._pg_tokens_to_fast_tokens(pg_tokens), dtype=np.int32)

    def decode_action_dct_coeffs(self, dct_coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(dct_coeffs, dtype=np.float32)
        if coeffs.ndim != 2:
            raise ValueError(f"Expected DCT coefficients with shape [T, D], got {coeffs.shape}.")
        return idct(coeffs / self._fast_tokenizer.scale, axis=0, norm="ortho").astype(np.float32)

    def _encode_actions_to_pg_tokens(self, actions: np.ndarray) -> np.ndarray:
        act = np.asarray(actions, dtype=np.float32)
        if act.ndim != 2:
            raise ValueError(f"Expected actions with shape [T, D], got {act.shape}.")
        if not self._rowwise_bpe:
            action_tokens = self._fast_tokenizer(act[None])[0]
            return self._act_tokens_to_paligemma_tokens(action_tokens)
        dct_coeffs = dct(act, axis=0, norm="ortho").astype(np.float32) * float(self._fast_tokenizer.scale)
        return self.encode_action_dct_coeffs(dct_coeffs)

    def encode_action_dct_coeffs(self, dct_coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(dct_coeffs, dtype=np.float32)
        if coeffs.ndim != 2:
            raise ValueError(f"Expected DCT coefficients with shape [T, D], got {coeffs.shape}.")
        quantized = np.rint(coeffs).astype(np.int32)
        if self._rowwise_bpe:
            encoded_rows: list[np.ndarray] = []
            for row in self._serialize_dct_coeff_rows(quantized):
                encoded_rows.append(self._encode_bpe_chars(row))
            encoded_arr = (
                np.concatenate(encoded_rows, axis=0).astype(np.int32, copy=False)
                if encoded_rows
                else np.asarray([], dtype=np.int32)
            )
        else:
            encoded_arr = self._encode_bpe_chars(self._serialize_dct_coeffs(quantized))
        return self._act_tokens_to_paligemma_tokens(encoded_arr)

    def decode_action_tokens_to_actions(
        self,
        action_tokens_in_pg_vocab: np.ndarray,
        *,
        action_horizon: int,
        action_dim: int,
        relaxed_decoding: bool = True,
    ) -> np.ndarray:
        dct_coeffs = self.decode_action_tokens_to_dct_coeffs(
            action_tokens_in_pg_vocab,
            action_horizon=action_horizon,
            action_dim=action_dim,
            relaxed_decoding=relaxed_decoding,
        )
        return self.decode_action_dct_coeffs(dct_coeffs)

    def format_action_tokens_as_output(self, action_tokens_in_pg_vocab: np.ndarray, *, add_eos: bool = True) -> np.ndarray:
        """Wrap action tokens into the canonical FAST output format:
        `Action: <tokens> |` (optionally with EOS after `|`).
        """
        action_tokens = np.asarray(action_tokens_in_pg_vocab, dtype=np.int32).reshape(-1)
        prefix = np.asarray(self._paligemma_tokenizer.encode("Action: "), dtype=np.int32)
        suffix = np.asarray(self._paligemma_tokenizer.encode("|", add_eos=add_eos), dtype=np.int32)
        return np.concatenate([prefix, action_tokens, suffix], axis=0).astype(np.int32, copy=False)

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

    def _pg_tokens_to_fast_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

    def _serialize_dct_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs)
        if coeffs.ndim != 2:
            raise ValueError(f"Expected DCT coefficients with shape [T, D], got {coeffs.shape}.")
        ordered = coeffs.T if self._transpose_dct_before_bpe else coeffs
        return ordered.reshape(-1)

    def _serialize_dct_coeff_rows(self, coeffs: np.ndarray) -> list[np.ndarray]:
        coeffs = np.asarray(coeffs)
        if coeffs.ndim != 2:
            raise ValueError(f"Expected DCT coefficients with shape [T, D], got {coeffs.shape}.")
        if self._rowwise_layout == "action_dim_major":
            ordered = coeffs.T
        else:
            ordered = coeffs
        return [np.asarray(row).reshape(-1) for row in ordered]

    def row_structure(self, *, action_horizon: int, action_dim: int) -> tuple[int, int]:
        if self._rowwise_layout == "action_dim_major":
            return int(action_dim), int(action_horizon)
        return int(action_horizon), int(action_dim)

    def row_char_width(self, *, action_horizon: int, action_dim: int) -> int:
        _, width = self.row_structure(action_horizon=action_horizon, action_dim=action_dim)
        return int(width)

    def num_rows(self, *, action_horizon: int, action_dim: int) -> int:
        rows, _ = self.row_structure(action_horizon=action_horizon, action_dim=action_dim)
        return int(rows)

    def decode_fast_tokens_to_text(self, fast_tokens: np.ndarray | list[int]) -> str:
        toks = np.asarray(fast_tokens, dtype=np.int32).reshape(-1)
        if toks.size == 0:
            return ""
        return str(self._fast_tokenizer.bpe_tokenizer.decode(toks.tolist()))

    def decode_pg_tokens_to_text(self, pg_tokens: np.ndarray | list[int]) -> str:
        pg = np.asarray(pg_tokens, dtype=np.int32).reshape(-1)
        if pg.size == 0:
            return ""
        return self.decode_fast_tokens_to_text(self._pg_tokens_to_fast_tokens(pg))

    def encode_row_chars_to_pg_tokens(self, row_coeffs: np.ndarray) -> np.ndarray:
        row = np.asarray(row_coeffs, dtype=np.int32).reshape(-1)
        return self._act_tokens_to_paligemma_tokens(self._encode_bpe_chars(row))

    def decode_row_pg_tokens_to_coeffs(self, pg_tokens: np.ndarray, *, row_width: int) -> np.ndarray:
        text = self.decode_pg_tokens_to_text(pg_tokens)
        coeffs = np.asarray([ord(ch) + int(self._fast_tokenizer.min_token) for ch in text], dtype=np.float32)
        if coeffs.shape[0] < int(row_width):
            coeffs = np.pad(coeffs, (0, int(row_width) - coeffs.shape[0]), mode="constant", constant_values=0)
        elif coeffs.shape[0] > int(row_width):
            coeffs = coeffs[: int(row_width)]
        return coeffs.astype(np.float32, copy=False)

    def _encode_bpe_chars(self, quantized: np.ndarray) -> np.ndarray:
        shifted = np.asarray(quantized, dtype=np.int32).reshape(-1) - int(self._fast_tokenizer.min_token)
        shifted = np.clip(shifted, 0, 0x10FFFF)
        bpe_text = "".join(chr(int(v)) for v in shifted.tolist())
        try:
            encoded = self._fast_tokenizer.bpe_tokenizer.encode(bpe_text)
        except Exception as exc:
            raise ValueError(f"Failed to encode DCT coefficients with FAST tokenizer: {exc}") from exc
        return np.asarray(encoded if isinstance(encoded, list) else list(encoded), dtype=np.int32)

    def _deserialize_dct_coeffs(
        self,
        flat_coeffs: np.ndarray,
        *,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        flat_coeffs = np.asarray(flat_coeffs)
        if self._transpose_dct_before_bpe:
            return flat_coeffs.reshape(action_dim, action_horizon).T
        return flat_coeffs.reshape(action_horizon, action_dim)


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
