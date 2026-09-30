"""Manual batched Qwen2 forward calls; no generation API in the serving path."""

import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from axiomserve.config import RuntimeConfig
from axiomserve.runtime.batching import decode_inputs, padded_inputs
from axiomserve.runtime.cache import KVState, pack, unpack


@dataclass
class StepResult:
    logits: torch.Tensor  # CPU float32 [batch, vocab]; copying synchronizes CUDA.
    caches: list[KVState]
    seconds: float
    packing_seconds: float


class Runner(Protocol):
    eos_ids: set[int]
    context_limit: int
    kv_bytes_per_token: int

    def encode(self, prompt: str) -> list[int]: ...
    def text(self, ids: list[int]) -> str: ...
    def prefill(self, sequences: list[list[int]]) -> StepResult: ...
    def decode(self, tokens: list[int], states: list[KVState]) -> StepResult: ...
    def memory(self) -> dict[str, int]: ...


class ModelRunner:
    """Full-attention Qwen2 only; fail early for unvalidated architectures."""

    def __init__(self, model: Any, tokenizer: Any, device: torch.device) -> None:
        config = model.config
        self._validate_config(config)
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.context_limit = int(config.max_position_embeddings)
        eos = getattr(model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer requires a pad or EOS token")
        head_dim = config.hidden_size // config.num_attention_heads
        element_size = next(model.parameters()).element_size()
        self.kv_bytes_per_token = (
            2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * element_size
        )

    @staticmethod
    def _validate_config(config: Any) -> None:
        if (
            config.model_type != "qwen2"
            or getattr(config, "use_sliding_window", False)
            or any(kind != "full_attention" for kind in getattr(config, "layer_types", []))
        ):
            raise ValueError("v0.1 supports full-attention Qwen2 models only")

    @classmethod
    def load(cls, config: RuntimeConfig) -> "ModelRunner":
        # CPU logits filtering is small; avoid a large host thread pool per token.
        torch.set_num_threads(config.cpu_threads)
        device_name = config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        dtype = (
            (torch.float16 if device_name == "cuda" else torch.float32)
            if config.dtype == "auto"
            else getattr(torch, config.dtype)
        )
        model_config = AutoConfig.from_pretrained(
            config.model,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        cls._validate_config(model_config)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            revision=config.revision,
            config=model_config,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        return cls(model, tokenizer, torch.device(device_name))

    def encode(self, prompt: str) -> list[int]:
        ids: list[int] = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids:
            raise ValueError("prompt must tokenize to at least one token")
        return ids

    def text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def prefill(self, sequences: list[list[int]]) -> StepResult:
        self._sync()
        start = time.monotonic()
        ids, mask, positions = padded_inputs(sequences, self.pad_id, self.device)
        self._sync()
        packed_at = time.monotonic()
        outputs = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = outputs.logits[:, -1].float().cpu()
        split_at = time.monotonic()
        caches = unpack(outputs.past_key_values, list(map(len, sequences)))
        self._sync()
        end = time.monotonic()
        return StepResult(logits, caches, end - start, packed_at - start + end - split_at)

    @torch.inference_mode()
    def decode(self, tokens: list[int], states: list[KVState]) -> StepResult:
        self._sync()
        start = time.monotonic()
        lengths = [state.length for state in states]
        cache = pack(states)
        ids, mask, positions = decode_inputs(tokens, lengths, self.device)
        self._sync()
        packed_at = time.monotonic()
        outputs = self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = outputs.logits[:, -1].float().cpu()
        split_at = time.monotonic()
        caches = unpack(outputs.past_key_values, [length + 1 for length in lengths])
        self._sync()
        end = time.monotonic()
        return StepResult(logits, caches, end - start, packed_at - start + end - split_at)

    def memory(self) -> dict[str, int]:
        if self.device.type != "cuda":
            return {"allocated": 0, "reserved": 0, "peak_allocated": 0}
        return {
            "allocated": torch.cuda.memory_allocated(self.device),
            "reserved": torch.cuda.memory_reserved(self.device),
            "peak_allocated": torch.cuda.max_memory_allocated(self.device),
        }
