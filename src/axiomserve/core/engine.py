"""Event-loop-owned lifecycle with one dedicated model execution thread."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from axiomserve.config import RuntimeConfig
from axiomserve.core.request import TERMINAL, RequestState, Status, TokenEvent
from axiomserve.core.sampling import SamplingParams, sample
from axiomserve.core.scheduler import make_policy
from axiomserve.metrics.collector import Metrics
from axiomserve.runtime.model_runner import Runner, StepResult

logger = logging.getLogger(__name__)


class OverloadedError(RuntimeError):
    """The bounded waiting queue has reached capacity."""


class Engine:
    def __init__(self, runner: Runner, config: RuntimeConfig) -> None:
        self.runner = runner
        self.config = config
        self.policy = make_policy(config.scheduler)
        self.metrics = Metrics(config.history_size)
        self.waiting: list[RequestState] = []
        self.active: list[RequestState] = []
        self.requests: dict[str, RequestState] = {}
        self._wake = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="axiom-model")
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

    @property
    def accepting(self) -> bool:
        return self._worker is not None and not self._closing and not self._worker.done()

    async def start(self) -> None:
        if self._closing:
            raise RuntimeError("engine has been closed")
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="axiom-inference")

    async def submit(self, prompt: str, params: SamplingParams) -> RequestState:
        arrival = time.monotonic()
        if self._worker is None or self._closing or self._worker.done():
            raise RuntimeError("engine is not accepting requests")
        if len(self.waiting) >= self.config.max_waiting:
            self.metrics.counters["rejected"] += 1
            raise OverloadedError("waiting queue is full")
        try:
            if len(prompt) > self.config.max_context * 64:
                raise ValueError("prompt exceeds character safety limit")
            if params.max_tokens > self.config.max_output:
                raise ValueError("max_tokens exceeds configured output limit")
            ids = self.runner.encode(prompt)
            request = RequestState(prompt, ids, params, arrival_time=arrival)
            context = min(self.config.max_context, self.runner.context_limit)
            if request.reserved_tokens > context:
                raise ValueError("prompt plus requested output exceeds context limit")
            if not self._fits([request]):
                raise ValueError("request cannot fit configured token/KV budget even alone")
        except ValueError:
            self.metrics.counters["rejected"] += 1
            raise
        self.waiting.append(request)
        self.requests[request.request_id] = request
        self.metrics.counters["accepted"] += 1
        self._wake.set()
        return request

    def cancel(self, request_id: str) -> bool:
        request = self.requests.get(request_id)
        if request is None or request.status in TERMINAL:
            return False
        request.cancel_requested = True
        self._wake.set()
        return True

    def _fits(self, requests: list[RequestState]) -> bool:
        # Reserve maximum output lengths, including padding, to avoid growth deadlock.
        tokens = len(requests) * max(r.reserved_tokens for r in requests)
        if tokens > self.config.max_batch_tokens:
            return False
        budget = self.config.max_kv_bytes
        # Persistent, packed, and newly returned caches can overlap in memory.
        return budget is None or 3 * tokens * self.runner.kv_bytes_per_token <= budget

    def _admit(self) -> list[RequestState]:
        if self.config.mode != "continuous" and self.active:
            return []
        maximum = 1 if self.config.mode == "sequential" else self.config.max_active
        admitted: list[RequestState] = []
        for request in self.policy.order(self.waiting):
            if len(self.active) + len(admitted) >= maximum:
                break
            if not self._fits(self.active + admitted + [request]):
                break  # Preserve policy order; no hidden backfilling.
            self.waiting.remove(request)
            request.admitted_time = time.monotonic()
            request.transition(Status.PREFILLING)
            admitted.append(request)
        self.active.extend(admitted)
        return admitted

    def _finish(
        self, request: RequestState, status: Status, reason: str, error: str | None = None
    ) -> None:
        if request.status in TERMINAL:
            return
        request.transition(status)
        request.completion_time = time.monotonic()
        request.finish_reason = reason
        request.error = error
        request.kv_state = None
        request.events.put_nowait(TokenEvent(finish_reason=reason, error=error))
        request.done.set()
        self.metrics.complete(request)
        self.requests.pop(request.request_id, None)
        logger.info(
            "request_terminal",
            extra={"request_id": request.request_id, "status": status.value, "reason": reason},
        )

    def _cleanup(self) -> None:
        for request in list(self.waiting) + self.active:
            if request.cancel_requested:
                self._finish(request, Status.CANCELLED, "cancelled")
        self.waiting[:] = [r for r in self.waiting if r.status not in TERMINAL]
        # Static mode retains finished slots until the original cohort completes.
        if self.config.mode != "static" or all(r.status in TERMINAL for r in self.active):
            self.active[:] = [r for r in self.active if r.status not in TERMINAL]

    async def _step(self, requests: list[RequestState], phase: str) -> None:
        loop = asyncio.get_running_loop()
        try:
            if phase == "prefill":
                result = await loop.run_in_executor(
                    self._executor, self.runner.prefill, [r.input_ids for r in requests]
                )
            else:
                states = [r.kv_state for r in requests]
                if any(state is None for state in states):
                    raise RuntimeError("running request missing KV state")
                result = await loop.run_in_executor(
                    self._executor,
                    self.runner.decode,
                    [r.generated_ids[-1] for r in requests],
                    [state for state in states if state is not None],
                )
            self._consume(requests, result, phase)
        except Exception as exc:
            logger.exception("model_step_failed", extra={"phase": phase})
            for request in requests:
                self._finish(request, Status.FAILED, "error", str(exc))

    def _consume(self, requests: list[RequestState], result: StepResult, phase: str) -> None:
        tokens_before = self.metrics.counters["tokens"]
        for row, request in enumerate(requests):
            if request.cancel_requested:
                self._finish(request, Status.CANCELLED, "cancelled")
                continue
            request.kv_state = result.caches[row]
            if phase == "prefill":
                request.prefill_seconds = result.seconds
                request.transition(Status.RUNNING)
            token = sample(result.logits[row], request.params, request.rng)
            now = time.monotonic()
            if request.first_token_time is None:
                request.first_token_time = now
            request.token_times.append(now)
            request.generated_ids.append(token)
            self.metrics.counters["tokens"] += 1
            eos = token in self.runner.eos_ids and not request.params.ignore_eos
            length = len(request.generated_ids) >= request.params.max_tokens
            text = self.runner.text(request.generated_ids)
            if not (eos or length):
                # Hold incomplete UTF-8 bytes until more tokens arrive.
                text = text.rstrip("\ufffd")
            if not text.startswith(request.output_text):
                raise RuntimeError("tokenizer changed an already emitted text prefix")
            delta = text[len(request.output_text) :]
            request.output_text = text
            request.events.put_nowait(TokenEvent(token_id=token, text=delta))
            if eos or length:
                self._finish(request, Status.FINISHED, "stop" if eos else "length")

        emitted = self.metrics.counters["tokens"] - tokens_before
        self.metrics.batch(phase, len(requests), result.seconds, result.packing_seconds, emitted)

    async def _run(self) -> None:
        try:
            while not self._closing:
                self._cleanup()
                if not self.active and not self.waiting:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                if self.config.mode == "static" and not self.active:
                    await asyncio.sleep(self.config.static_batch_wait_ms / 1000)
                    self._cleanup()
                admitted = self._admit()
                if admitted:
                    await self._step(admitted, "prefill")
                self._cleanup()
                running = [r for r in self.active if r.status == Status.RUNNING]
                if running:
                    await self._step(running, "decode")
                await asyncio.sleep(0)
        except Exception as exc:
            logger.exception("inference_worker_failed")
            for request in list(self.requests.values()):
                self._finish(request, Status.FAILED, "error", str(exc))
        finally:
            for request in list(self.requests.values()):
                self._finish(request, Status.CANCELLED, "shutdown")
            self.active.clear()
            self.waiting.clear()

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        if self._worker is not None:
            await self._worker  # Let an in-flight GPU call finish before releasing storage.
        self._executor.shutdown(wait=True)

    def snapshot(self) -> dict[str, Any]:
        running = sum(r.status not in TERMINAL for r in self.active)
        result = self.metrics.snapshot(running, len(self.waiting), self.runner.memory())
        result["occupied_slots"] = len(self.active)
        result["active_cache_bytes"] = sum(
            r.kv_state.nbytes for r in self.active if r.kv_state is not None
        )
        return result
