# Latest-code runtime smoke test — 2026-09-09

Tested source commit: `f9a662e` on
`research/static-profile-online-reference-20260909`.
This record validates the bounded online reference workflow, not every optional
ablation, full benchmark quality, or production readiness. No inference code was
changed during this test and no other services were stopped.

## Results

- 18 CPU unit tests passed; all 19 Python files introduced/changed by the source
  commit compiled successfully without writing bytecode.
- Profiling shell syntax and commit whitespace checks passed.
- The three historical online/MMLU artifacts passed default analysis. Historical
  MMLU token mismatches remain documented; they were not silently waived as exact
  matches.
- Real Orin GPU inference exited successfully: FP/PTQ/mixed × native/grouped,
  4 requests per configuration, B<=2, k=3, forced 8 output tokens, FP head,
  eager, simultaneous arrival, one warmup and one measured repeat.
- All 24 measured requests finished, producing 192 tokens in total.
- The new analyzer passed all 41 structural checks, found complete baseline
  coverage and 8/8 exact mixed/static token matches. Strict CLI mode exited 0.
  New request manifests were present; no legacy-completeness warnings occurred.

The environment emitted the known Torch CC 8.7 support warning and a first-use
Triton compilation warning but inference completed. This short run is functional
validation, not a new throughput or quality claim. CUDA Graph, confidence gating,
preemption replay and the INT4-head ablation were not exercised in this run.

## Environment and reproduction

Container: `riveredge-vllm-src`; Jetson AGX Orin; Torch 2.11.0+cu130,
vLLM 0.24.0, TorchAO 0.17.0. The imported extension was the workspace checkout.
Initial CUDA free memory was approximately 27 GiB. Explicit KV allocation was
128 MiB; the run did not rely on shared-device memory profiling for its KV budget.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=river-vllm-edge python3 -m unittest discover -s river-vllm-edge/scripts -p 'test_online*.py'
bash -n experiments/14_single_request_end_to_end_profile/run_all.sh
git diff HEAD^ HEAD --check
docker exec riveredge-vllm-src python3 /workspace/vllm/RiverEdge/river-vllm-edge/scripts/benchmark_riveredge_online.py --output /workspace/vllm/RiverEdge/reports/latest_smoke_I1JQ3Zkn/online.json --policies fp,ptq,mixed --schedulers native,grouped --requests 4 --batch-size 2 --max-tokens 8 --max-model-len 128 --warmup 1 --repeats 1 --arrivals simultaneous --force-length --kv-cache-bytes 134217728
python3 -B river-vllm-edge/scripts/analyze_riveredge_online.py reports/latest_smoke_I1JQ3Zkn/online.json --output reports/latest_smoke_I1JQ3Zkn/online.analysis.json --require-token-equality
```

Use a **new output directory** when repeating these commands; the benchmark
intentionally rejects overwriting existing results. Model default:
`/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq`.

Artifacts:

- [Raw results, source hashes, configuration and request manifests](../reports/latest_smoke_I1JQ3Zkn/online.json)
- [Strict analysis](../reports/latest_smoke_I1JQ3Zkn/online.analysis.json)
