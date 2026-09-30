"""In-process serving benchmarks: model load excluded, queue and inference included."""

import asyncio
import importlib.metadata
import platform
import time
from typing import Any

import torch

from axiomserve.bench.workload import BenchConfig, WorkItem, make_workload
from axiomserve.core.engine import Engine
from axiomserve.core.request import Status
from axiomserve.core.sampling import SamplingParams
from axiomserve.metrics.collector import Metrics, percentile, request_metrics
from axiomserve.runtime.model_runner import ModelRunner


async def run_case(
    config: BenchConfig, runner: ModelRunner, mode: str, concurrency: int, items: list[WorkItem]
) -> dict[str, Any]:
    settings = config.runtime.model_copy(update={"mode": mode})
    engine = Engine(runner, settings)
    await engine.start()
    try:
        for _ in range(config.warmup_requests):
            request = await engine.submit(
                items[0].prompt,
                SamplingParams(max_tokens=min(2, items[0].output_tokens), ignore_eos=True),
            )
            await request.done.wait()
            if request.status != Status.FINISHED:
                raise RuntimeError(f"warmup failed: {request.error}")
        engine.metrics = Metrics(settings.history_size)
        if runner.device.type == "cuda":
            torch.cuda.synchronize(runner.device)
            torch.cuda.reset_peak_memory_stats(runner.device)
        semaphore = asyncio.Semaphore(concurrency)
        records: list[dict[str, Any]] = []
        started = time.monotonic()

        async def send(item: WorkItem) -> None:
            planned = started + item.arrival_offset
            await asyncio.sleep(max(0, planned - time.monotonic()))
            async with semaphore:
                submitted = time.monotonic()
                try:
                    request = await engine.submit(
                        item.prompt,
                        SamplingParams(
                            max_tokens=item.output_tokens, seed=item.seed, ignore_eos=True
                        ),
                    )
                    await request.done.wait()
                    record = request_metrics(request)
                except (ValueError, RuntimeError) as exc:
                    record = {"status": "rejected", "error": str(exc), "output_tokens": 0}
                record.update(
                    {
                        "planned_arrival_offset": item.arrival_offset,
                        "client_wait_seconds": submitted - planned,
                        "client_latency_seconds": time.monotonic() - planned,
                        "target_prompt_tokens": item.target_prompt_tokens,
                    }
                )
                records.append(record)

        await asyncio.gather(*(send(item) for item in items))
        elapsed = time.monotonic() - started
        finished = [r for r in records if r["status"] == "finished"]
        ttft = [r["ttft_seconds"] for r in finished]
        latency = [r["latency_seconds"] for r in finished]
        return {
            "mode": mode,
            "scheduler": settings.scheduler,
            "concurrency": concurrency,
            "elapsed_seconds": elapsed,
            "successful_requests": len(finished),
            "unsuccessful_requests": len(records) - len(finished),
            "requests_per_second": len(finished) / elapsed,
            "tokens_per_second": sum(r["output_tokens"] for r in finished) / elapsed,
            "ttft_p50_seconds": percentile(ttft, 0.5),
            "ttft_p95_seconds": percentile(ttft, 0.95),
            "latency_p50_seconds": percentile(latency, 0.5),
            "latency_p95_seconds": percentile(latency, 0.95),
            "memory_bytes": runner.memory(),
            "requests": records,
            "system": engine.snapshot(),
        }
    finally:
        await engine.close()


async def benchmark(config: BenchConfig) -> dict[str, Any]:
    runner = await asyncio.to_thread(ModelRunner.load, config.runtime)
    items = make_workload(config, runner)
    cases = []
    for mode in config.modes:
        for concurrency in config.concurrencies:
            cases.append(await run_case(config, runner, mode, concurrency, items))
    return {
        "schema_version": 1,
        "kind": "in_process_real_model",
        "created_unix": time.time(),
        "config": config.model_dump(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
            "model_commit": getattr(runner.model.config, "_commit_hash", None),
            "transformers": importlib.metadata.version("transformers"),
            "device": str(runner.device),
            "dtype": str(next(runner.model.parameters()).dtype),
            "gpu": torch.cuda.get_device_name(runner.device)
            if runner.device.type == "cuda"
            else None,
        },
        "cases": cases,
    }
