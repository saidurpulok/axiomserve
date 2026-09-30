"""Seeded workloads with planned arrival times and target token distributions."""

import random
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from axiomserve.config import RuntimeConfig
from axiomserve.runtime.model_runner import Runner


def _default_modes() -> list[Literal["sequential", "static", "continuous"]]:
    return ["sequential", "static", "continuous"]


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    modes: list[Literal["sequential", "static", "continuous"]] = Field(
        default_factory=_default_modes
    )
    concurrencies: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    requests: int = Field(default=32, ge=1, le=10000)
    arrival: Literal["constant", "burst"] = "burst"
    rate: float = Field(default=10, gt=0)
    burst_size: int = Field(default=8, ge=1)
    burst_interval: float = Field(default=0.1, ge=0)
    prompt_tokens: tuple[int, int] = (16, 64)
    output_tokens: tuple[int, int] = (16, 64)
    seed: int = Field(default=42, ge=0)
    warmup_requests: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_workload(self) -> "BenchConfig":
        for bounds in [self.prompt_tokens, self.output_tokens]:
            if bounds[0] < 1 or bounds[1] < bounds[0]:
                raise ValueError("token bounds must be positive and ordered")
        if not self.concurrencies or any(c < 1 for c in self.concurrencies) or not self.modes:
            raise ValueError("modes and positive concurrencies must be nonempty")
        if self.output_tokens[1] > self.runtime.max_output:
            raise ValueError("output distribution exceeds runtime output limit")
        return self


@dataclass(frozen=True)
class WorkItem:
    prompt: str
    output_tokens: int
    target_prompt_tokens: int
    arrival_offset: float
    seed: int


def make_workload(config: BenchConfig, runner: Runner) -> list[WorkItem]:
    rng = random.Random(config.seed)
    items = []
    for index in range(config.requests):
        target = rng.randint(*config.prompt_tokens)
        # Decode a token-limited repeated corpus; retokenized lengths are measured,
        # rather than claiming arbitrary tokenizer round trips preserve exact lengths.
        corpus = "Explain how computers store information. " * (target + 1)
        ids = runner.encode(corpus)[:target]
        prompt = runner.text(ids)
        offset = (
            index / config.rate
            if config.arrival == "constant"
            else (index // config.burst_size) * config.burst_interval
        )
        items.append(
            WorkItem(
                prompt, rng.randint(*config.output_tokens), target, offset, config.seed + index
            )
        )
    return items
