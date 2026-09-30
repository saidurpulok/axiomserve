# Initial measured results

Measured on 2026-09-30. Source: [`results/gpu-small.json`](results/gpu-small.json).
This is a small systems sanity check, not a statistical performance study.

## Environment and workload

- NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB reported device memory, WSL2.
- Python 3.12.13, PyTorch 2.14.0+cu130, Transformers 5.17.0, float16.
- Qwen/Qwen2.5-0.5B-Instruct, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- One CPU PyTorch thread, FCFS, max active 8; configured padded token budget 4096
  and approximate transient KV budget 1 GiB.
- 16 requests per case, seeded identical workloads, two bursts of eight separated
  by 100 ms; prompt targets 8–32 tokens and output targets 8–24 tokens.
- Two short single-request warmups per case. Model load excluded, EOS ignored to
  keep output lengths fixed. No HTTP transport in the timed path.
- All 192 measured requests across 12 cases finished successfully, with no
  rejections or failures. Actual retokenized prompt lengths were 8–30 tokens.

## Measurements

TTFT and latency are measured from engine submission. Client concurrency waiting
is recorded separately in the JSON. Percentiles interpolate the 16 successful
request observations. GPU memory is peak PyTorch allocated memory, including
weights and transient caches; it is not total device usage.

| Mode | Concurrency | Requests/s | Tokens/s | TTFT p50 (ms) | TTFT p95 (ms) | Latency p50 (ms) | Latency p95 (ms) | Peak allocated (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequential | 1 | 2.08 | 31.8 | 32 | 37 | 463 | 723 | 984.2 |
| sequential | 2 | 2.08 | 31.8 | 481 | 714 | 917 | 1381 | 984.2 |
| sequential | 4 | 2.16 | 33.1 | 1308 | 1623 | 1763 | 2148 | 984.2 |
| sequential | 8 | 2.20 | 33.7 | 2823 | 3478 | 3257 | 4052 | 984.2 |
| static | 1 | 2.10 | 32.2 | 35 | 40 | 449 | 710 | 984.2 |
| static | 2 | 2.72 | 41.7 | 180 | 341 | 670 | 1016 | 985.9 |
| static | 4 | 4.43 | 67.9 | 80 | 460 | 797 | 1320 | 989.1 |
| static | 8 | 7.08 | 108.5 | 57 | 503 | 908 | 1581 | 995.3 |
| continuous | 1 | 2.14 | 32.8 | 31 | 33 | 449 | 672 | 984.2 |
| continuous | 2 | 3.16 | 48.4 | 59 | 74 | 601 | 917 | 986.1 |
| continuous | 4 | 4.66 | 71.4 | 72 | 84 | 711 | 1130 | 989.7 |
| continuous | 8 | 6.56 | 100.5 | 62 | 119 | 948 | 1416 | 996.5 |

## Observations and limits

These short contexts kept peak allocation near 1 GiB. This workload does not
stress the full 8 GiB device budget or establish memory-admission behavior under
real GPU pressure. Admission limits are tested separately.

Batching increased token throughput in this run. At concurrency 8, static had
higher throughput than continuous, while continuous had lower p95 TTFT and
p95 latency. Continuous admission exposes a latency/throughput tradeoff here;
there is no universal winner supported by this experiment.

Across concurrency-8 cases, measured input/cache preparation and cache splitting
accounted for about 29–31% of batched runner time. These synchronized wall timers
include allocation/dispatch overhead and are not isolated GPU kernel timings.
This suggests cache movement is worth profiling, but it does not establish the
cause of a throughput difference.

Each point is one small run without confidence intervals. Small per-case warmups
do not exercise every future batch shape, and PyTorch/Triton can compile kernels
on first use. Case order is fixed, GPU clocks/thermals are uncontrolled, and the
allocator/kernel caches persist across cases. Different arrival rates, context
lengths, output distributions, repetitions, and randomized case order are needed
before stronger conclusions. Static mode compacts finished compute rows but
holds cohort slots; continuous uses separate prefill calls for new requests.

Plots generated from this JSON are in [`results/plots/`](results/plots/), with
PNG and PDF versions for throughput, median TTFT, p95 latency, and GPU memory.

## Reproduce this run

```bash
uv sync --python 3.12 --group dev --extra bench
uv run axiom bench --config configs/bench_small.yaml --output results/gpu-small.json
uv run python scripts/plot_benchmarks.py results/gpu-small.json
```

`configs/bench_small.yaml` sets `local_files_only: true`, so download the model
first (for example by starting `axiom serve` once) or set that flag to false. If
PyTorch's Triton helper fails to compile because Python development headers are
missing, install a Python build that includes them and point `CPATH` at its
`include/python3.12` directory.

## Validation evidence

- [`results/tests-final.txt`](results/tests-final.txt): 25 tests passed, including
  pretrained CUDA greedy equivalence against an explicitly neutral `generate()`
  oracle. A Starlette/httpx test-client deprecation warning remains.
- [`results/smoke-gpu.jsonl`](results/smoke-gpu.jsonl): all three modes produced
  identical token sequences, with two active slots and three requests; continuous
  admitted a fresh request while an older request remained running.
- [`results/types.txt`](results/types.txt): strict mypy passed for 22 source files.
- [`results/lint.txt`](results/lint.txt), [`results/format.txt`](results/format.txt):
  Ruff lint and formatting checks passed.

The checkpoint's generation defaults include a repetition penalty of 1.1. The
first oracle comparison used that default and differed from raw greedy decoding;
the final oracle explicitly uses repetition penalty 1.0. The serving engine never
calls `generate()`.
