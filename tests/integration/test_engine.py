import asyncio

import pytest
from conftest import ControlledRunner

from axiomserve.config import RuntimeConfig
from axiomserve.core.engine import Engine, OverloadedError
from axiomserve.core.request import Status
from axiomserve.core.sampling import SamplingParams
from axiomserve.runtime.model_runner import ModelRunner


def config(**kwargs: object) -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {"max_context": 256, "max_output": 64, "max_active": 2, **kwargs}
    )


@pytest.mark.parametrize("mode", ["sequential", "static", "continuous"])
async def test_modes_real_model(tiny_runner: ModelRunner, mode: str) -> None:
    engine = Engine(tiny_runner, config(mode=mode))
    await engine.start()
    try:
        requests = [
            await engine.submit(prompt, SamplingParams(max_tokens=n, ignore_eos=True))
            for prompt, n in [("ab", 2), ("abcdef", 5), ("c", 3)]
        ]
        await asyncio.wait_for(asyncio.gather(*(r.done.wait() for r in requests)), 10)
        assert all(r.status == Status.FINISHED for r in requests)
        assert [len(r.generated_ids) for r in requests] == [2, 5, 3]
        for request in requests:
            result = tiny_runner.prefill([request.input_ids])
            expected = []
            for _ in range(request.params.max_tokens):
                token = int(result.logits[0].argmax())
                expected.append(token)
                result = tiny_runner.decode([token], result.caches)
            assert request.generated_ids == expected
        if mode == "sequential":
            assert all(batch["size"] == 1 for batch in engine.metrics.batches)
        else:
            assert any(batch["size"] == 2 for batch in engine.metrics.batches)
        assert not engine.requests
    finally:
        await engine.close()


@pytest.mark.parametrize("mode,early", [("continuous", True), ("static", False)])
async def test_admission_before_cohort_finishes(mode: str, early: bool) -> None:
    runner = ControlledRunner()
    engine = Engine(runner, config(mode=mode))
    await engine.start()
    try:
        short = await engine.submit("a", SamplingParams(max_tokens=1))
        long = await engine.submit("bb", SamplingParams(max_tokens=8))
        third = await engine.submit("c", SamplingParams(max_tokens=2))
        await asyncio.wait_for(third.done.wait(), 3)
        await asyncio.wait_for(long.done.wait(), 3)
        assert short.completion_time is not None
        assert third.admitted_time is not None and long.completion_time is not None
        assert (third.admitted_time < long.completion_time) == early
        if early:
            assert ("decode", 2) in runner.calls
    finally:
        await engine.close()


async def test_queue_limits_cancel_and_shutdown() -> None:
    engine = Engine(ControlledRunner(0.03), config(max_waiting=1))
    await engine.start()
    first = await engine.submit("a", SamplingParams(max_tokens=20))
    with pytest.raises(OverloadedError):
        await engine.submit("b", SamplingParams())
    assert engine.cancel(first.request_id)
    await asyncio.wait_for(first.done.wait(), 3)
    assert first.status == Status.CANCELLED
    second = await engine.submit("b", SamplingParams(max_tokens=20))
    while second.status == Status.WAITING:
        await asyncio.sleep(0.001)
    assert engine.cancel(second.request_id)
    await asyncio.wait_for(second.done.wait(), 3)
    assert second.status == Status.CANCELLED
    third = await engine.submit("c", SamplingParams(max_tokens=20))
    await engine.close()
    assert third.status == Status.CANCELLED
    assert not engine.requests and not engine.active and not engine.waiting
    with pytest.raises(RuntimeError):
        await engine.submit("d", SamplingParams())


async def test_admission_rejects_unserviceable_requests() -> None:
    engine = Engine(ControlledRunner(), config(max_batch_tokens=8, max_kv_bytes=192))
    await engine.start()
    try:
        with pytest.raises(ValueError, match="budget"):
            await engine.submit("abcdefghi", SamplingParams(max_tokens=1))
        with pytest.raises(ValueError, match="context"):
            await engine.submit("a" * 255, SamplingParams(max_tokens=2))
        with pytest.raises(ValueError, match="output"):
            await engine.submit("a", SamplingParams(max_tokens=65))
        request = await engine.submit("abc", SamplingParams(max_tokens=5))
        await asyncio.wait_for(request.done.wait(), 3)
        assert request.status == Status.FINISHED
    finally:
        await engine.close()


async def test_failed_forward_terminates_request() -> None:
    class FailingRunner(ControlledRunner):
        def prefill(self, sequences: list[list[int]]) -> object:
            raise RuntimeError("deliberate failure")

    engine = Engine(FailingRunner(), config())  # type: ignore[arg-type]
    await engine.start()
    try:
        request = await engine.submit("a", SamplingParams(max_tokens=2))
        await asyncio.wait_for(request.done.wait(), 3)
        assert request.status == Status.FAILED
        assert request.error == "deliberate failure"
    finally:
        await engine.close()


async def test_eos_stops_after_first_token() -> None:
    runner = ControlledRunner()
    runner.eos_ids = {2}
    engine = Engine(runner, config())
    await engine.start()
    try:
        request = await engine.submit("a", SamplingParams(max_tokens=8))
        await asyncio.wait_for(request.done.wait(), 3)
        assert request.finish_reason == "stop"
        assert len(request.generated_ids) == 1
        assert runner.calls == [("prefill", 1)]
    finally:
        await engine.close()


async def test_memory_budget_limits_joint_admission() -> None:
    # Each request reserves 8 tokens, 8 bytes/token and 3 copies = 192 bytes.
    engine = Engine(ControlledRunner(), config(max_kv_bytes=192))
    await engine.start()
    try:
        requests = [await engine.submit("abc", SamplingParams(max_tokens=5)) for _ in range(2)]
        await asyncio.wait_for(asyncio.gather(*(r.done.wait() for r in requests)), 3)
        assert all(batch["size"] == 1 for batch in engine.metrics.batches)
        assert requests[1].admitted_time >= requests[0].completion_time
    finally:
        await engine.close()


@pytest.mark.parametrize("mode", ["static", "continuous"])
async def test_seeded_output_independent_of_admission(tiny_runner: ModelRunner, mode: str) -> None:
    params = SamplingParams(
        max_tokens=6, temperature=0.8, top_k=12, top_p=0.9, seed=91, ignore_eos=True
    )
    outputs = []
    for execution in ["sequential", mode]:
        engine = Engine(tiny_runner, config(mode=execution))
        await engine.start()
        try:
            requests = [await engine.submit(prompt, params) for prompt in ["a", "abcdef", "bc"]]
            await asyncio.wait_for(asyncio.gather(*(r.done.wait() for r in requests)), 10)
            outputs.append([r.generated_ids for r in requests])
        finally:
            await engine.close()
    assert outputs[0] == outputs[1]
