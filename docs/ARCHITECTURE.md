# Architecture

## Ownership and concurrency

`Engine` owns waiting, active, and live-request registries on one asyncio event
loop. HTTP handlers enqueue work and consume request events. `submit` has no
internal await; admission checks and queue insertion are atomic with respect to
other handlers on that loop. Tokenization is currently synchronous and bounded
by a character limit; long prompts can still delay the event loop.

One `ThreadPoolExecutor` thread runs model forwards and cache transformations.
The event loop awaits the result without blocking networking. CUDA execution is
synchronized before timings end. Logits are returned as CPU float32 for sampling.
Sampling and lifecycle mutation occur on the event loop. The model is never
forwarded concurrently by two workers.

API lifespan loads the model, starts the engine, and closes it. Shutdown prevents
submission, waits for any current forward, and marks remaining requests cancelled.
Cancellation similarly takes effect at iteration boundaries, not by interrupting
an executing CUDA kernel.

## Control flow

1. Validate API sampling fields, prompt/output limits, and waiting capacity.
2. Tokenize, check logical context and whether the request can fit alone.
3. Append a `WAITING` request with a monotonic arrival timestamp and seeded RNG.
4. Policy orders candidates. Admission reserves worst-case padded tokens and
   optionally estimated transient KV storage before moving to `PREFILLING`.
5. Prefill runs a real batch, splits its caches, and samples the first token.
6. Remaining `RUNNING` requests supply their latest token and cached prefix to a
   real batched one-token decode. Cache length grows by one each iteration.
7. Sample, record token timings, emit text deltas, and check EOS/output limit.
8. Terminal transition records metrics, releases cache, emits terminal event,
   signals `done`, and removes the live registry entry.

A cache contains the prompt and generated tokens already forwarded. Immediately
after sampling, the latest generated token is not yet in it. For N generated
tokens, final logical cache length before release is prompt length + N − 1.

## Modes and policies

FCFS and shortest-prompt-first only order admissions. The engine implements all
execution mechanics. Admission stops when the next policy candidate cannot fit;
there is no hidden memory backfilling. FCFS preserves ordering but can incur
head-of-line blocking; shortest-prompt-first has no aging/fairness guarantee.

Sequential admits one request. Static waits briefly to collect a cohort and
keeps occupied slots until every original member is terminal, while avoiding
forwards on finished rows. Continuous removes terminal requests between steps
and admits into freed slots. New prompt prefill is a separate batch; the next
decode combines new and older requests. This is real iteration-level batching,
with blocking prefill interference as an explicit limitation.

## KV data flow

`KVState` stores per-layer key/value tensors `[1, kv_heads, L, head_dim]` without
padding. Keys are already RoPE-rotated for the request's logical positions.
Prefill left-pads IDs and uses mask-derived position IDs. Decode left-pads each
cache to batch maximum L, concatenates rows, supplies a padding mask, and passes
one latest token per row with logical position L. The model's causal masking
uses the physical cache span; the explicit mask hides left padding.

`pack` uses `DynamicCache.update`; `unpack` reads `cache.layers[*].keys/values`.
Slices are cloned so one short request cannot retain an entire batch allocation.
The adapter is confined to `cache.py` and checked against the tested Transformers
version. All supported layers must use full attention.

## Observability and bounded storage

Request event queues hold at most max-output + one terminal event. There is no
unbounded token queue and no requirement for a client to consume events before
a request can terminate. Waiting and active storage are limited by configuration.
Terminal state is retained by client handles and bounded metric histories, not
an unlimited engine registry. Metrics retain lifetime counters plus recent
request and batch records. Percentiles cover successful recent completions;
throughput counters divide by uptime. Benchmarks reset counters after warmup.

`/metrics` is JSON, not Prometheus exposition. Batch records expose size, phase,
wall duration, and cache/input packing duration. Packing time includes
input/cache preparation and output cache splitting, synchronized at the boundaries
on CUDA; it includes allocation, dispatch, and synchronization costs. System snapshots include live
cache bytes and CUDA allocator statistics. Token emission is timed at the engine;
network delivery and client rendering are not measured by these timestamps.
