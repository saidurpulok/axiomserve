"""Explicit request lifecycle and bounded token delivery."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

import torch

from axiomserve.core.sampling import SamplingParams
from axiomserve.runtime.cache import KVState


class Status(StrEnum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL = {Status.FINISHED, Status.CANCELLED, Status.FAILED}
TRANSITIONS = {
    Status.WAITING: {Status.PREFILLING, Status.CANCELLED, Status.FAILED},
    Status.PREFILLING: {Status.RUNNING, Status.CANCELLED, Status.FAILED},
    Status.RUNNING: TERMINAL,
}


@dataclass(frozen=True)
class TokenEvent:
    token_id: int | None = None
    text: str = ""
    finish_reason: str | None = None
    error: str | None = None


@dataclass
class RequestState:
    prompt: str
    input_ids: list[int]
    params: SamplingParams
    request_id: str = field(default_factory=lambda: uuid4().hex)
    arrival_time: float = field(default_factory=time.monotonic)
    status: Status = Status.WAITING
    generated_ids: list[int] = field(default_factory=list)
    admitted_time: float | None = None
    first_token_time: float | None = None
    completion_time: float | None = None
    prefill_seconds: float | None = None
    token_times: list[float] = field(default_factory=list)
    kv_state: KVState | None = None
    finish_reason: str | None = None
    error: str | None = None
    output_text: str = ""
    cancel_requested: bool = False
    events: asyncio.Queue[TokenEvent] = field(init=False)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    rng: torch.Generator = field(init=False)

    def __post_init__(self) -> None:
        # A full output plus the terminal event fits even if the consumer stops reading.
        self.events = asyncio.Queue(maxsize=self.params.max_tokens + 1)
        self.rng = torch.Generator(device="cpu").manual_seed(self.params.seed)

    def transition(self, target: Status) -> None:
        if target not in TRANSITIONS.get(self.status, set()):
            raise ValueError(f"invalid request transition {self.status} -> {target}")
        self.status = target

    @property
    def reserved_tokens(self) -> int:
        return len(self.input_ids) + self.params.max_tokens
