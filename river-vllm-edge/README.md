# RiverEdge optimization switches

Use `scripts/benchmark_vllm_switches.py` for new switch-based comparisons, or
`river_vllm_ext.optimization_switches.create_engine` from Python. Every RiverEdge
switch defaults to **off**. Switches are selected before model construction;
restart the engine/process to change them, rather than mutating live KV history.

| Switch | Effect when enabled | Dependency |
| --- | --- | --- |
| `--ptq-tail` | FP prefix and HQQ PTQ decode tail; k=3 by default | unified FP+HQQ checkpoint |
| `--online-routing` | per-request externally supplied eligibility, mixed projection and explicit phase routes | PTQ tail; eager only |
| `--grouped-scheduling` | phase/route-separated scheduler | online routing |
| `--batch-aware-routing` | low/high batch hysteresis | online routing |
| `--conservative-fallback` | mixed-eligibility batch falls back to FP | online routing |
| `--int4-head` | optional HQQ head on PTQ decode rows only; quality-changing | online routing |
| `--decode-cudagraph` | static full-decode Graph configuration, compilation disabled | incompatible with online routing |

Every switch has a matching `--no-...` form. `--all-off` asserts that no RiverEdge
switch was enabled; contradictory arguments fail rather than silently ignoring
a requested optimization. Hysteresis and conservative fallback are independently
switchable. INT4 head is not selected by default and is not a new recommended
experiment.

## What “all off” guarantees

The factory selects the native unquantized Llama checkpoint, calls the ordinary
vLLM `LLM` constructor, installs **no** `OnlineRuntime`, scheduler monkeypatch,
row-routing hook, PTQ sidecar or head replacement. Native mode rejects custom
RiverEdge architecture/quantization metadata and indexed sidecar weights. It
does not map `full_fp` to a RiverEdge model: `full_fp` in the historical benchmark
still allocates a custom dual-tail model and is not this native baseline.

Upstream vLLM features (paged KV, continuous batching, its default compilation,
Graph and scheduler behavior) are preserved. “All off” means all **RiverEdge**
optimizations are off, not disabling stock vLLM's own optimizations. Ordinary
vLLM options can still be explicitly set for a controlled comparison. The smoke
CLI uses the same in-process harness, memory budget and batch limits for all
paths; `--enforce-eager` is an ordinary vLLM execution option, not a RiverEdge
optimization. Without it, native mode preserves the upstream execution default.

The existing local `Llama-3.1-8B-Instruct-native-view` contains only FP shards and
native config/index; it shares FP weight files with the unified checkpoint.
The original weight files are not rewritten. Alternatively pass an ordinary
unquantized native Llama checkpoint using `--native-model`.

## Commands (configured Docker environment)

Run from `/workspace/vllm/RiverEdge` inside `riveredge-vllm-src` with the extension
installed. Choose a fresh output path for every run.

```bash
# Native vLLM with no RiverEdge behavior. Keep upstream execution defaults.
python3 river-vllm-edge/scripts/benchmark_vllm_switches.py --all-off --output /tmp/native.json

# Native eager baseline with the same harness limits.
python3 river-vllm-edge/scripts/benchmark_vllm_switches.py --all-off --enforce-eager --output /tmp/native-eager.json

# Algorithm/static-tail ablation with native scheduling and static decode Graph.
python3 river-vllm-edge/scripts/benchmark_vllm_switches.py --ptq-tail --decode-cudagraph --output /tmp/static-graph.json

# Online mixed routing; no custom grouped scheduler or hysteresis.
python3 river-vllm-edge/scripts/benchmark_vllm_switches.py --ptq-tail --online-routing --output /tmp/online.json

# Independently enable scheduling and batch adaptation.
python3 river-vllm-edge/scripts/benchmark_vllm_switches.py --ptq-tail --online-routing --grouped-scheduling --batch-aware-routing --output /tmp/grouped.json
```

The new CLI emits switches, selected and actual model classes, hook presence,
source hash, completion checks and outputs. Its schema is separate from the
historical timed-arrival analyzer schema: native mode deliberately has no route
trace, and absence of that trace must not be mistaken for a broken online trace.
For Python integrations, `create_engine(..., **vllm_options)` exposes the same
switches without requiring this smoke harness.

## Boundaries

Historical benchmark/profiling scripts remain available with their original
explicit mode arguments to preserve reproduction commands; they do not consume
the new switches. Use the canonical entry above for all-off comparisons.
Profiler instrumentation itself remains optional and is not enabled by this
entry. No vendored vLLM source changes are required.

Confidence gating, split/dynamic CUDA Graph and custom kernels are not yet
implemented. They are not presented as functional switches. Online mode still
rejects preemption, prefix caching, speculation, TP/PP>1 and asynchronous
scheduling. Static PTQ without its Graph switch uses eager because the historical
PTQ Inductor experiment was reverted; this does not constrain the native path.
