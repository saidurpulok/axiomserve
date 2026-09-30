from axiomserve.bench.reporting import save_report
from axiomserve.bench.runner import run_case
from axiomserve.bench.workload import BenchConfig, make_workload
from axiomserve.config import RuntimeConfig
from axiomserve.runtime.model_runner import ModelRunner


async def test_real_cpu_benchmark_case_and_report(tiny_runner: ModelRunner, tmp_path) -> None:
    import json

    config = BenchConfig(
        runtime=RuntimeConfig(max_context=256, max_output=16, max_active=2),
        requests=3,
        prompt_tokens=(2, 4),
        output_tokens=(2, 3),
        warmup_requests=1,
    )
    items = make_workload(config, tiny_runner)
    assert items == make_workload(config, tiny_runner)
    cases = [
        await run_case(config, tiny_runner, mode, 2, items)
        for mode in ["sequential", "static", "continuous"]
    ]
    assert all(case["successful_requests"] == 3 for case in cases)
    assert all(case["tokens_per_second"] > 0 for case in cases)
    assert all(case["memory_bytes"]["allocated"] == 0 for case in cases)
    path = tmp_path / "report.json"
    save_report({"cases": cases}, str(path))
    assert len(json.loads(path.read_text())["cases"]) == 3
