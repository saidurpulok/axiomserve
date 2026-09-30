# Design decisions

## Backend scope

Full-attention Qwen2 is a deliberate scope limit. Qwen2.5-0.5B is small enough
for repeated constrained-GPU experiments and uses grouped-query attention and
standard autoregressive cache semantics. Checkpoint ID remains configurable,
but another architecture is rejected. No remote model code is trusted.

The adapter uses modern `DynamicCache` layer storage instead of obsolete tuple
cache conversion APIs. Installed source was inspected for constructor/update,
layer storage, and Qwen2 forward/logit interfaces. Dependency bounds and the lock
file capture the tested interface; cache API changes require rerunning correctness
tests rather than assuming compatibility.

References: [Qwen checkpoint card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
and [Transformers cache documentation](https://huggingface.co/docs/transformers/en/cache_explanation).
The code is an original small serving implementation, not copied from an engine.

## Dense caches and variable lengths

Independent, compact per-request cache tensors simplify ownership, removal, and
admission. Packing pads to the longest active cache and reconstructs one
Transformers batch. Logical position IDs preserve RoPE semantics when rows have
different lengths. Cloning slices avoids keeping unrelated rows alive.

An alternative is one persistent rectangular cache with row compaction and
complex position bookkeeping; another is grouping identical lengths. Paged KV
storage avoids much copying but needs a different attention implementation.
The dense adapter intentionally lets us measure the cost of a simple correct
approach before considering specialized kernels.

## Reservations

Admission reserves `batch_count × max(prompt_length + max_output)` tokens.
This bounds padded growth even under heterogeneous output lengths. Optional KV
budget checks multiply this by KV bytes/token and three for overlapping cache
storage. It is an admission estimate, not a total-device-memory guarantee.
It sacrifices utilization to avoid admitting work whose growth cannot fit the
configured token budget. No preemption or eviction is implemented.

## One worker, FastAPI, and synchronization

FastAPI supplies HTTP validation and SSE integration without owning inference.
One execution thread serializes use of one model. Async handlers remain available
while that thread executes forwards. Multiple model workers would complicate
memory contention and scheduler attribution before there is evidence of benefit.

CUDA-to-CPU logits, sampling, synchronization, input creation, and cache copying
are explicit sources of overhead. Timed model steps include tensor preparation,
forwards, splitting, and synchronization; they are not pure kernel durations.
CPU per-request RNGs isolate random streams from admission order.
The explicit `cpu_threads` setting defaults to one to avoid oversubscribing small
CPU logits operations; model load sets this process-wide PyTorch option and
benchmarks record its value. CPU model users can increase it for experimentation. Floating point
batched kernel differences can still alter sampled tokens; seeded RNG does not
promise bit-identical CPU/GPU text.

## Streaming and lifecycle

Request streams have a finite max-output budget, so their entire event sequence
fits a bounded queue even if a client stops consuming. HTTP disconnect requests
cancellation; cancellation completes after an in-flight forward. SSE reports
partial text and then a reason/error. Unicode decoding holds incomplete trailing
replacement characters and asserts emitted-prefix stability for the chosen
Qwen tokenizer rather than claiming arbitrary-tokenizer streaming support.

## Experimental baselines

Static mode retains cohort capacity but compacts compute to unfinished rows.
All three modes share dense cache mechanics, making scheduling comparisons
interpretable while giving static mode a somewhat stronger compute baseline than
fixed-row execution. Continuous prefill is not fused with decode. Benchmarking
must explain these choices rather than comparing mode names without mechanics.

## Production differences

Production engines can use block-managed caches, specialized attention, fused
sampling, chunked prefill, and sophisticated resource schedulers. AxiomServe's
purpose is to expose and measure the serving mechanics in a readable system.
No performance superiority or research novelty is claimed.
