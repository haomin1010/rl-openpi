import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer


class _FakeBPETokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return [ord(ch) for ch in text]

    def decode(self, tokens):
        return "".join(chr(int(tok)) for tok in tokens)


class _FakeFASTProcessor:
    def __init__(self):
        self.bpe_tokenizer = _FakeBPETokenizer()
        self.min_token = 0
        self.scale = 1.0

    def __call__(self, actions):
        _, horizon, action_dim = actions.shape
        return [np.arange(horizon * action_dim, dtype=np.int32)]


@pytest.fixture
def fast_tokenizer_stub(monkeypatch):
    monkeypatch.setattr(_tokenizer.AutoProcessor, "from_pretrained", lambda *args, **kwargs: _FakeFASTProcessor())


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer(fast_tokenizer_stub):
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_fast_tokenizer_dct_serialization_transpose_roundtrip(fast_tokenizer_stub):
    tokenizer = _tokenizer.FASTTokenizer(max_len=256, transpose_dct_before_bpe=True)
    coeffs = np.arange(12, dtype=np.float32).reshape(3, 4)

    flat = tokenizer._serialize_dct_coeffs(coeffs)
    restored = tokenizer._deserialize_dct_coeffs(flat, action_horizon=3, action_dim=4)

    np.testing.assert_array_equal(flat, coeffs.T.reshape(-1))
    np.testing.assert_array_equal(restored, coeffs)


def test_fast_tokenizer_rowwise_bpe_encodes_each_action_dim_row_independently(fast_tokenizer_stub):
    tokenizer = _tokenizer.FASTTokenizer(
        max_len=256,
        rowwise_bpe=True,
        rowwise_layout="action_dim_major",
    )
    coeffs = np.arange(12, dtype=np.float32).reshape(3, 4)

    tokenizer.encode_action_dct_coeffs(coeffs)

    calls = tokenizer._fast_tokenizer.bpe_tokenizer.calls
    assert len(calls) == 4
    assert [len(call) for call in calls] == [3, 3, 3, 3]


def test_fast_tokenizer_rowwise_bpe_tokenize_uses_rowwise_action_encoding(fast_tokenizer_stub):
    prompt = "Hello"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 4).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(
        max_len=256,
        rowwise_bpe=True,
        rowwise_layout="action_dim_major",
    )

    tokenizer.tokenize(prompt, state, action)

    calls = tokenizer._fast_tokenizer.bpe_tokenizer.calls
    assert len(calls) == action.shape[1]
