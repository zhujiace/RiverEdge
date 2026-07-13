#!/usr/bin/env python3
"""Benchmark vLLM static RiverEdge modes with serialized torchao checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq",
    )
    parser.add_argument("--modes", default="full_fp,static_fp_tail,static_ptq_tail")
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--prompt-tokens-target", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    parser.add_argument(
        "--output-json",
        default="/workspace/vllm/RiverEdge/experiments/09_torchao_serialized_riveredge/p6_vllm_static_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        default="/workspace/vllm/RiverEdge/experiments/09_torchao_serialized_riveredge/p6_vllm_static_summary.csv",
    )
    parser.add_argument("--worker-mode", default="")
    parser.add_argument("--worker-output", default="")
    return parser.parse_args()


def make_prompt(target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog near the river edge. "
    return (base * max(1, target_tokens // 12 + 1)).strip()


def riveredge_phase_trace(llm) -> list[str]:
    """Read phase markers from an in-process vLLM V1 model when available."""
    try:
        engine_core_client = llm.llm_engine.engine_core
        engine_core = getattr(engine_core_client, "engine_core", engine_core_client)
        model = engine_core.model_executor.driver_worker.model_runner.model
    except AttributeError:
        return []
    while not hasattr(model, "_traced_phases") and hasattr(model, "model"):
        model = model.model
    return sorted(getattr(model, "_traced_phases", ()))


def run_mode(args: argparse.Namespace, mode: str) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    if mode not in {"full_fp", "static_fp_tail", "static_ptq_tail"}:
        raise ValueError(f"Unsupported mode: {mode}")
    model_path = args.model
    t0 = time.perf_counter()
    llm = LLM(
        model=model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len * args.max_num_seqs,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        trust_remote_code=False,
        disable_log_stats=True,
        hf_overrides={
            "river_edge_mode": mode,
            "river_edge_exit_layer": args.exit_layer,
        },
    )
    load_s = time.perf_counter() - t0
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    prompt = make_prompt(args.prompt_tokens_target)

    for _ in range(args.warmup):
        llm.generate([prompt], sampling)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    total_output_tokens = 0
    last_text = ""
    last_token_ids: list[int] = []
    for _ in range(args.iters):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        out = outputs[0].outputs[0]
        total_output_tokens += len(out.token_ids)
        latencies.append(elapsed)
        last_text = out.text
        last_token_ids = list(out.token_ids)

    phase_trace = riveredge_phase_trace(llm)

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_s = sum(latencies)
    return {
        "mode": mode,
        "exit_layer": args.exit_layer,
        "model_path": model_path,
        "enforce_eager": args.enforce_eager,
        "iters": args.iters,
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "total_output_tokens": total_output_tokens,
        "load_s": load_s,
        "total_s": total_s,
        "mean_latency_s": total_s / len(latencies),
        "output_tps": total_output_tokens / total_s if total_s > 0 else 0.0,
        "latencies_s": latencies,
        "last_text": last_text,
        "last_token_ids": last_token_ids,
        "phase_trace": phase_trace,
    }


def run_mode_subprocess(args: argparse.Namespace, mode: str) -> dict:
    worker_output = Path(args.output_json).with_name(f"{Path(args.output_json).stem}.{mode}.json")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--modes",
        mode,
        "--exit-layer",
        str(args.exit_layer),
        "--prompt-tokens-target",
        str(args.prompt_tokens_target),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--worker-mode",
        mode,
        "--worker-output",
        str(worker_output),
    ]
    cmd.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
    subprocess.run(cmd, env=env, check=True)
    with worker_output.open("r", encoding="utf-8") as handle:
        return json.load(handle)["result"]


def write_outputs(args: argparse.Namespace, results: list[dict]) -> None:
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "exit_layer",
                "enforce_eager",
                "iters",
                "warmup",
                "max_new_tokens",
                "total_output_tokens",
                "load_s",
                "total_s",
                "mean_latency_s",
                "output_tps",
                "model_path",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        result = run_mode(args, args.worker_mode)
        worker_output = Path(args.worker_output)
        worker_output.parent.mkdir(parents=True, exist_ok=True)
        worker_output.write_text(json.dumps({"result": result}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"result": result}, indent=2))
        return

    modes = [item for item in args.modes.replace(",", " ").split() if item]
    results = [run_mode_subprocess(args, mode) for mode in modes]
    write_outputs(args, results)
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
