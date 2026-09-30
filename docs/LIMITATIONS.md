# Limitations and interpretation

- **Model scope:** full-attention Qwen2 only. Qwen3, sliding-window/hybrid models,
  arbitrary Hugging Face architectures, and encoder-decoder models are unvalidated.
- **Cache bandwidth:** padding, concatenation, and cloning repeat each decode;
  cost grows with context and layers. There is no paged attention or persistent
  packed batch buffer. Transient allocations can greatly exceed live cache bytes.
- **Resource estimates:** token reservations bound configured padded capacity;
  KV estimates exclude weights, activations, workspaces, allocator fragmentation,
  and external GPU use. CUDA OOM is reported as failure of affected requests,
  not silently retried with a different mode. An unhealthy CUDA context may need
  process restart; no universal recovery guarantee is made.
- **Prefill interference:** new prompt prefill blocks decode for active requests.
  No chunked prefill, preemption, cache offload, or latency SLO enforcement exists.
- **Static interpretation:** cohort slots stay occupied, but finished rows are
  removed from model compute. Results describe this baseline explicitly.
- **Scheduling:** shortest-prompt-first can starve long prompts. No aging, fairness
  guarantee, or learned policy exists. Reservation ordering can block small work
  behind a large candidate that currently cannot fit.
- **Host overhead:** tokenization, sampling, and text decode run on the event loop.
  Prompt character limits bound work but do not eliminate event-loop stalls.
  Logits copying, synchronization, CPU RNGs, and repeated full-output text decode
  add overhead. No custom kernels, CUDA graphs, compile path, or fused sampling.
- **Streaming:** finite per-request output buffering protects memory but does not
  backpressure model compute based on network speed. Timings are emission timings,
  not observed HTTP delivery. Prefix stability is assumed only for the tested
  tokenizer; incomplete UTF-8 prefixes are withheld. EOS counts as an output token.
- **Protocol/security:** raw completions only; no chat conversion, auth, persistence,
  multi-model routing, or complete OpenAI API compatibility. Default binding is
  localhost. These are experimental local serving interfaces.
- **Metrics:** bounded windows mean recent percentiles differ from lifetime
  counters. GPU figures are PyTorch allocator observations, not total-device usage.
  TPOT is the average emission gap after the first token; it includes prefill
  interference and host work. First-token sampling is included in TTFT but not
  the runner's prefill timing. Failed/cancelled requests are excluded from latency
  summaries and must be considered separately.
- **Benchmark scope:** in-process workload sweeps omit HTTP/network overhead;
  concurrency limiting introduces a separate client wait. Fixed outputs ignore
  EOS. Token-target prompts may retokenize differently. Workload text is synthetic.
  A single small sweep is not a statistically controlled performance study;
  repeats, confidence intervals, varying arrivals, and profiler traces are needed
  before broad conclusions. Batch history can truncate on long runs.
- **Environment:** PyTorch 2.14's internal Triton GPU operations can require a C
  compiler and matching Python development headers even for eager forwards.
  If they are missing, install a Python build with headers and point `CPATH` at it.
- **Deployment:** one model on one GPU/CPU, one process, no distributed inference,
  quantization engine, multimodal path, speculative decode, or prefix caching.

Tests using tiny random weights establish cache/decoding mechanics, not language
quality. Tests using timing doubles establish control behavior, not GPU behavior.
Only actual pretrained GPU runs can support claims about the target hardware.
