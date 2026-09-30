"""Actual pretrained inference, with all three serving modes and heterogeneous requests."""

import argparse
import asyncio
import json

from axiomserve.config import RuntimeConfig
from axiomserve.core.engine import Engine
from axiomserve.core.request import Status
from axiomserve.core.sampling import SamplingParams
from axiomserve.runtime.model_runner import ModelRunner


async def smoke(model: str, device: str) -> None:
    settings = RuntimeConfig(model=model, device=device, max_active=2)
    runner = await asyncio.to_thread(ModelRunner.load, settings)
    reference: list[list[int]] | None = None
    for mode in ["sequential", "static", "continuous"]:
        engine = Engine(runner, settings.model_copy(update={"mode": mode}))
        await engine.start()
        try:
            requests = [
                await engine.submit(prompt, SamplingParams(max_tokens=tokens))
                for prompt, tokens in [
                    ("The capital of France is", 8),
                    ("In one sentence, explain a KV cache:", 16),
                    ("1 + 1 =", 4),
                ]
            ]
            await asyncio.wait_for(asyncio.gather(*(r.done.wait() for r in requests)), 120)
            if any(r.status != Status.FINISHED for r in requests):
                raise RuntimeError(str([r.error for r in requests]))
            actual = [request.generated_ids for request in requests]
            if reference is None:
                reference = actual
            elif actual != reference:
                raise RuntimeError(f"{mode} greedy outputs differ from sequential baseline")
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "device": str(runner.device),
                        "outputs": [r.output_text for r in requests],
                        "batch_sizes": [b["size"] for b in engine.metrics.batches],
                    }
                )
            )
        finally:
            await engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    asyncio.run(smoke(args.model, args.device))
