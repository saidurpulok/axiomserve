"""Tiny randomly initialized Qwen2: real attention/caches, no network or pretrained quality."""

import time

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from axiomserve.runtime.cache import KVState
from axiomserve.runtime.model_runner import ModelRunner, StepResult


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [2 + ord(character) % 29 for character in text] or [2]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        return "".join(chr(97 + token % 26) for token in ids if token > 1)


@pytest.fixture
def tiny_runner() -> ModelRunner:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = Qwen2ForCausalLM(config)
    model.config._attn_implementation = "sdpa"
    return ModelRunner(model, TinyTokenizer(), torch.device("cpu"))


class ControlledRunner:
    """A deterministic timing double, used only for lifecycle/scheduler tests."""

    eos_ids: set[int] = set()
    context_limit = 256
    kv_bytes_per_token = 8

    def __init__(self, delay: float = 0.003) -> None:
        self.delay = delay
        self.calls: list[tuple[str, int]] = []

    def encode(self, prompt: str) -> list[int]:
        return [2] * len(prompt)

    def text(self, ids: list[int]) -> str:
        return "x" * len(ids)

    def _result(self, lengths: list[int], phase: str) -> StepResult:
        start = time.monotonic()
        time.sleep(self.delay)
        self.calls.append((phase, len(lengths)))
        logits = torch.zeros(len(lengths), 4)
        logits[:, 2] = 1
        states = [
            KVState(((torch.zeros(1, 1, length, 1), torch.zeros(1, 1, length, 1)),))
            for length in lengths
        ]
        return StepResult(logits, states, time.monotonic() - start, 0)

    def prefill(self, sequences: list[list[int]]) -> StepResult:
        return self._result(list(map(len, sequences)), "prefill")

    def decode(self, tokens: list[int], states: list[KVState]) -> StepResult:
        return self._result([state.length + 1 for state in states], "decode")

    def memory(self) -> dict[str, int]:
        return {"allocated": 0, "reserved": 0, "peak_allocated": 0}
