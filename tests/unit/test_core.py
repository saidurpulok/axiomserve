import pytest
import torch
from pydantic import ValidationError

from axiomserve.config import RuntimeConfig
from axiomserve.core.request import RequestState, Status
from axiomserve.core.sampling import SamplingParams, sample
from axiomserve.core.scheduler import FCFS, ShortestPromptFirst
from axiomserve.metrics.collector import Metrics, percentile, request_metrics
from axiomserve.runtime.cache import KVState, pack, unpack


def test_sampling_greedy_topk_nucleus_and_seed() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    rng = torch.Generator().manual_seed(5)
    assert sample(scores, SamplingParams(), rng) == 3
    assert sample(scores, SamplingParams(temperature=1, top_k=1), rng) == 3
    assert sample(scores, SamplingParams(temperature=1, top_p=0.01), rng) == 3
    params = SamplingParams(temperature=0.8, top_k=3, top_p=0.9)
    first = torch.Generator().manual_seed(91)
    second = torch.Generator().manual_seed(91)
    assert [sample(scores, params, first) for _ in range(20)] == [
        sample(scores, params, second) for _ in range(20)
    ]
    with pytest.raises(ValueError, match="non-finite"):
        sample(torch.tensor([float("nan")]), params, first)
    with pytest.raises(ValidationError):
        SamplingParams(temperature=float("inf"))


def test_nucleus_keeps_threshold_crossing_token() -> None:
    # First token has probability .4; a .5 nucleus must also contain the second.
    logits = torch.tensor([0.4, 0.35, 0.25]).log()
    rng = torch.Generator().manual_seed(4)
    samples = {sample(logits, SamplingParams(temperature=1, top_p=0.5), rng) for _ in range(100)}
    assert samples == {0, 1}


def test_lifecycle_and_policies() -> None:
    long = RequestState("long", [2] * 4, SamplingParams(), arrival_time=1)
    short = RequestState("s", [2], SamplingParams(), arrival_time=2)
    assert FCFS().order([short, long]) == [long, short]
    assert ShortestPromptFirst().order([long, short]) == [short, long]
    with pytest.raises(ValueError):
        long.transition(Status.FINISHED)
    long.transition(Status.PREFILLING)
    long.transition(Status.RUNNING)
    long.transition(Status.FINISHED)
    with pytest.raises(ValueError):
        long.transition(Status.RUNNING)


def test_cache_roundtrip_clones_and_memory() -> None:
    states = [
        KVState(((torch.ones(1, 2, n, 4) * n, torch.ones(1, 2, n, 4) * (n + 1)),)) for n in [2, 5]
    ]
    packed = pack(states)
    restored = unpack(packed, [2, 5])
    for old, new in zip(states, restored, strict=True):
        assert new.length == old.length
        assert new.nbytes == 2 * 1 * 2 * old.length * 4 * 4
        torch.testing.assert_close(old.layers[0][0], new.layers[0][0])
        assert new.layers[0][0].untyped_storage().nbytes() == new.layers[0][0].numel() * 4


def test_metrics_and_bounded_history() -> None:
    request = RequestState("x", [2], SamplingParams(), arrival_time=1)
    request.admitted_time = 2
    request.first_token_time = 3
    request.token_times = [3, 5, 7]
    request.generated_ids = [2, 2, 2]
    request.completion_time = 8
    request.status = Status.FINISHED
    record = request_metrics(request)
    assert record["queue_seconds"] == 1
    assert record["ttft_seconds"] == 2
    assert record["tpot_seconds"] == 2
    assert record["latency_seconds"] == 7
    assert percentile([1, 3], 0.5) == 2
    assert percentile([], 0.5) is None
    metrics = Metrics(1)
    metrics.complete(request)
    metrics.complete(request)
    assert len(metrics.records) == 1
    assert metrics.counters["finished"] == 2


def test_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(max_active=0)
    with pytest.raises(ValidationError):
        RuntimeConfig(max_context=64, max_output=64)
