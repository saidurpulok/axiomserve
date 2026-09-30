"""Bounded histories and lifetime counters; latency summaries use recent records."""

import time
from collections import Counter, deque
from typing import Any

from axiomserve.core.request import RequestState, Status


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def request_metrics(request: RequestState) -> dict[str, Any]:
    end = request.completion_time
    first = request.first_token_time
    return {
        "request_id": request.request_id,
        "status": request.status.value,
        "arrival_time": request.arrival_time,
        "admitted_time": request.admitted_time,
        "first_token_time": first,
        "completion_time": end,
        "queue_seconds": (
            request.admitted_time - request.arrival_time
            if request.admitted_time is not None
            else None
        ),
        "prefill_seconds": request.prefill_seconds,
        "ttft_seconds": first - request.arrival_time if first is not None else None,
        "latency_seconds": end - request.arrival_time if end is not None else None,
        "tpot_seconds": (
            (request.token_times[-1] - first) / (len(request.token_times) - 1)
            if first is not None and len(request.token_times) > 1
            else None
        ),
        "inter_token_seconds": [
            b - a for a, b in zip(request.token_times, request.token_times[1:], strict=False)
        ],
        "prompt_tokens": len(request.input_ids),
        "output_tokens": len(request.generated_ids),
        "finish_reason": request.finish_reason,
        "error": request.error,
    }


class Metrics:
    def __init__(self, history_size: int) -> None:
        self.started = time.monotonic()
        self._last_batch_time = self.started
        self.counters: Counter[str] = Counter()
        self.records: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.batches: deque[dict[str, Any]] = deque(maxlen=history_size)

    def complete(self, request: RequestState) -> None:
        self.counters[request.status.value] += 1
        self.records.append(request_metrics(request))

    def batch(
        self, phase: str, size: int, seconds: float, packing: float, emitted: int = 0
    ) -> None:
        now = time.monotonic()
        interval = max(now - self._last_batch_time, 1e-9)
        self._last_batch_time = now
        self.batches.append(
            {
                "time": now - self.started,
                "emitted_tokens": emitted,
                "emission_tokens_per_second": emitted / interval,
                "tokens_total": self.counters["tokens"],
                "phase": phase,
                "size": size,
                "seconds": seconds,
                "packing_seconds": packing,
            }
        )

    def snapshot(self, active: int, queued: int, memory: dict[str, int]) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        finished = [r for r in self.records if r["status"] == Status.FINISHED.value]
        latency = [r["latency_seconds"] for r in finished]
        ttft = [r["ttft_seconds"] for r in finished if r["ttft_seconds"] is not None]
        return {
            "uptime_seconds": elapsed,
            "active": active,
            "queued": queued,
            "counters": dict(self.counters),
            "requests_per_second": self.counters["finished"] / elapsed,
            "tokens_per_second": self.counters["tokens"] / elapsed,
            "latency_seconds": {f"p{q}": percentile(latency, q / 100) for q in [50, 95, 99]},
            "ttft_seconds": {f"p{q}": percentile(ttft, q / 100) for q in [50, 95]},
            "gpu_memory_bytes": memory,
            "recent_requests": list(self.records),
            "recent_batches": list(self.batches),
            "history_capacity": self.records.maxlen,
        }
