# Static profiling and online reference milestone (2026-09-09)

## Scope

This milestone collects experiments 11–15, static FP/PTQ benchmark and profiling
tools, and the eager online execution reference. It does not claim that the
complete RiverEdge algorithm or optimized serving architecture is finished.

The current experiment deliberately forces FP/PTQ tails after the shared prefix
(normally k=3) to characterize performance. Confidence gating at Llama layer 3
will be added later. Online eligibility is externally assigned. No vLLM source
files are patched. Use the benchmark's `river_edge_mode=full_fp` override for the
online model: its all-FP branch otherwise inherits the static model's mode.

Completed evidence:

- Experiment 11: static FP/PTQ × eager/CUDA Graph × B=1/2/4/8/16;
  full GSM8K closed-batch evaluation, including paired quality analysis.
- Experiment 12: Inductor exploration failed multi-batch validation; model
  changes were reverted. Eager and full-decode Graph remain the static baseline.
- Experiments 13–14: reproduction, memory-traffic diagnostics, and single-request
  profiling. Instrumented times are attribution evidence, not serving results.
- Experiment 15: timed arrivals, eager mixed-route projection, grouped scheduling,
  delivery/SLO metrics, small real-data quality checks, and shared-rail energy.
  Grouping has not demonstrated a speedup. Historical INT4-head results are
  opt-in ablations, not the selected optimization or an authorized new experiment.

Pending: confidence gating, split/dynamic Graph execution, architecture speedup
over the same algorithm on native vLLM, numerical/KV equivalence diagnostics,
preemption route replay, and any custom-kernel work. The online reference is
restricted to synchronous, eager, in-process TP=PP=1 execution without prefix
caching or speculative decoding. Preemption is explicitly rejected.

## Analyzer contract

`river-vllm-edge/scripts/analyze_riveredge_online.py` is an offline artifact
checker, not part of inference. Schema 3 separates:

- `structural_checks_pass`: nonempty/unique cases and requests, planned case
  matrix when available, request manifest/counts, completion/timestamps/token
  totals, scheduler trace coverage, prefill-FP and grouped-step contracts.
- `all_contracts_pass`: compatibility alias for structural checks only.
- `analysis_pass`: structural checks plus complete, unambiguous, protocol-matched
  mixed/static baseline pairing. This does **not** assert numerical equivalence.
- `output_comparison.status`: `exact_match`, `mismatch`, `incomplete`, or
  `not_applicable`. Every pair identifies its case and baseline.

Default CLI exit is nonzero for structural failures or missing/incompatible
baselines. `--require-token-equality` additionally requires a complete, nonempty,
exact mixed/static comparison. A token mismatch alone is not proof of a routing
bug: batch-dependent numerical differences still need logits/KV diagnosis.

New replay results store `expected_request_ids` before analysis. Legacy synthetic
runs use the saved request count; legacy trace runs without a manifest can only
check summary counts and trace consistency, and emit an explicit warning. They
cannot prove that the original workload was fully recorded if both the summary
and underlying records were already truncated. The analyzer recognizes vLLM's
eight-hex internal request-ID suffix without removing arbitrary suffixes.

## Verification

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=river-vllm-edge python3 -m unittest discover -s river-vllm-edge/scripts -p 'test_online*.py' -v
bash -n experiments/14_single_request_end_to_end_profile/run_all.sh
git diff --check
python3 -B river-vllm-edge/scripts/analyze_riveredge_online.py experiments/15_resume_20260907/online_final_matrix.json --output /tmp/riveredge-online-analysis.json --require-token-equality
```

18 CPU tests passed (12 analyzer regressions and 6 existing reference tests).
All 19 changed/new Python files passed compilation without writing bytecode.
No new GPU experiments were run during this review.

Reanalysis results are saved separately as `*.review_20260909.json` under
experiment 15; historical raw results and analyses are unchanged:

| Artifact | Structure/pair coverage | Token comparison |
| --- | --- | --- |
| online_final_matrix | pass | 48/48 equal |
| mmlu_20260908 | pass; legacy trace warning | 186/192 equal |
| mmlu_reasoning_20260908 | pass; legacy trace warning | 42/48 equal |

The original online matrix benchmark source hash differs from the later source;
the reasoning run matched the pre-review source. Adding manifests changes the
benchmark hash again. Historical results are not advertised as executions of
the newly modified benchmark.

## Environment and artifact boundaries

Historical GPU runs used Jetson AGX Orin, vLLM 0.24.0, Torch 2.11.0 and
TorchAO 0.17 in `riveredge-vllm-src`, with model
`/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq`. Workload and runtime commands
are recorded in JSON configuration/argv fields and individual experiment docs.
Shared VDD_GPU_SOC measurements include other device workloads; they are not
isolated whole-board energy. No other services were stopped.

This source/evidence snapshot excludes generated package metadata, large binary
profiler traces/databases, the full available-metrics dump, and unrelated paper
or planning edits. Profiler CSV/JSON summaries are included; regenerating some
summaries requires the local raw traces or a new profiling run. Historical docs
may link to those local artifacts. Rejected autoresearch is not part of this PR.
