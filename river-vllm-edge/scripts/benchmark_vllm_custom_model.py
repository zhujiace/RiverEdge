#!/usr/bin/env python3
"""Initial speed probe for RiverEdge vLLM custom models."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-root",
        default="/workspace/vllm/RiverEdge/experiments/07_vllm_custom_model/model_adapters_container",
    )
    parser.add_argument("--modes", default="full_fp,static_fp_tail,static_ptq_tail")
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--prompt-tokens-target", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--output-json", default="/workspace/vllm/RiverEdge/experiments/07_vllm_custom_model/summary.json")
    parser.add_argument("--output-csv", default="/workspace/vllm/RiverEdge/experiments/07_vllm_custom_model/summary.csv")
    parser.add_argument("--worker-mode", default="")
    parser.add_argument("--worker-output", default="")
    parser.add_argument("--in-process", action="store_true")
    return parser.parse_args()


def make_prompt(target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    return (base * max(1, target_tokens // 10 + 1)).strip()


def run_mode(args, mode: str):
    import torch
    from vllm import LLM, SamplingParams

    model_path = str(Path(args.adapter_root) / f"{mode}_k{args.exit_layer}")
    llm = LLM(
        model=model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        enable_chunked_prefill=False,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    prompt = make_prompt(args.prompt_tokens_target)

    for _ in range(args.warmup):
        llm.generate([prompt], sampling)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    total_output_tokens = 0
    last_text = ""
    for _ in range(args.iters):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        out = outputs[0].outputs[0]
        output_tokens = len(out.token_ids)
        total_output_tokens += output_tokens
        latencies.append(elapsed)
        last_text = out.text

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_s = sum(latencies)
    return {
        "mode": mode,
        "model_path": model_path,
        "iters": args.iters,
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "total_output_tokens": total_output_tokens,
        "total_s": total_s,
        "mean_latency_s": total_s / len(latencies),
        "output_tps": total_output_tokens / total_s if total_s > 0 else 0.0,
        "latencies_s": latencies,
        "last_text": last_text,
    }


def run_mode_subprocess(args, mode: str):
    worker_output = Path(args.output_json).with_name(f"{Path(args.output_json).stem}.{mode}.json")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--adapter-root",
        args.adapter_root,
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
        "--worker-mode",
        mode,
        "--worker-output",
        str(worker_output),
    ]
    env = os.environ.copy()
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
    subprocess.run(cmd, env=env, check=True)
    with worker_output.open() as f:
        return json.load(f)["result"]


def main() -> None:
    args = parse_args()
    modes = [m for m in args.modes.replace(",", " ").split() if m]
    if args.worker_mode:
        result = run_mode(args, args.worker_mode)
        worker_output = Path(args.worker_output)
        worker_output.parent.mkdir(parents=True, exist_ok=True)
        worker_output.write_text(json.dumps({"result": result}, indent=2) + "\n")
        print(json.dumps({"result": result}, indent=2))
        return

    if args.in_process:
        results = [run_mode(args, mode) for mode in modes]
    else:
        results = [run_mode_subprocess(args, mode) for mode in modes]

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"results": results}, indent=2) + "\n")

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "iters",
                "warmup",
                "max_new_tokens",
                "total_output_tokens",
                "total_s",
                "mean_latency_s",
                "output_tps",
                "model_path",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
