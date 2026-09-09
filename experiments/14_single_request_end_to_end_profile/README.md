# Single-Request End-to-End Profile

This experiment profiles one RiverEdge request from a raw prompt through
tokenization, scheduling, GPU execution, sampling, detokenization, and final
output. It compares `full_fp` with `static_ptq_tail` at `k=3`.

## Measurement Layers

- `profile_single_request.py`: production CUDA Graph timing with an
  uninstrumented baseline, TTFT, strict TPOT, per-token arrivals, engine steps,
  CPU stages, and CUDA Event GPU stages.
- `trace_single_request.py`: low-overhead Nsys capture after model/graph warmup.
  Optional `--layer-nvtx` marks FP/PTQ layer, attention, and MLP boundaries.
- `analyze_profile.py`: creates all CSV tables, `analysis.json`, and the Chinese
  report in `summary.md`.

The main result uses 85 prompt tokens, 32 forced output tokens,
batch/concurrency 1, BF16, FULL_DECODE_ONLY CUDA Graph, and no Inductor. The
in-process endpoint deliberately excludes HTTP/SSE because the research target
is the GPU inference critical path.

## Reproduction

Run the complete matrix from the host:

```bash
cd /home/orin/zjc/vllm/RiverEdge/experiments/14_single_request_end_to_end_profile
bash run_all.sh
```

`run_all.sh` uses `riveredge-vllm-src`, copies the ARM64 Nsys target into the
container when needed, and imports `.qdstrm` on the host. It loads the 18.5 GiB
checkpoint once per case and can take several minutes. Only one model process
runs at a time.

For a narrow CUDA Graph run:

```bash
sudo docker exec riveredge-vllm-src python3 \
  /workspace/vllm/RiverEdge/experiments/14_single_request_end_to_end_profile/profile_single_request.py \
  --mode static_ptq_tail \
  --execution-mode cudagraph \
  --detail runtime
```

## Interpretation

Use baseline rows in `request_summary.csv` for latency comparisons. Nsys wall
time and eager layer-hook wall time are diagnostic only. CUDA Graph runtime
insertion adds less than 0.1% E2E overhead; PTQ eager layer hooks are much more
intrusive, so PTQ layer attribution comes from `nvtx_kern_sum`.
`critical_path.csv` is the compact stage table; `runtime_stage_summary.csv`
contains every recorded CPU/GPU stage and phase.

No vLLM source file is modified. Runtime wrappers and hooks are installed only
after warmup and disappear when the profiling process exits.
