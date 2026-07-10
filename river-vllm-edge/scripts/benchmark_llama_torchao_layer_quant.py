#!/usr/bin/env python3
"""Smoke benchmark for FP Llama vs torchao-quantized Llama in vLLM."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--quant-model",
        default="/models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-online",
    )
    parser.add_argument("--quant-config", default="")
    parser.add_argument("--mode", choices=["fp", "quant"], default="quant")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    if args.mode == "fp":
        model_path = args.model
        quantization = None
        hf_overrides = None
    else:
        model_path = args.quant_model
        quantization = "torchao"
        quant_config = args.quant_config or str(Path(model_path) / "torchao_layer_quant_config.json")
        hf_overrides = {"quantization_config_file": quant_config}

    t0 = time.perf_counter()
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
        quantization=quantization,
        hf_overrides=hf_overrides,
    )
    load_s = time.perf_counter() - t0

    prompt = "Explain in one concise sentence what LLM inference does."
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    for _ in range(args.warmup):
        llm.generate([prompt], sampling)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    total_tokens = 0
    text = ""
    for _ in range(args.iters):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        out = outputs[0].outputs[0]
        total_tokens += len(out.token_ids)
        latencies.append(elapsed)
        text = out.text

    result = {
        "mode": args.mode,
        "model_path": model_path,
        "quantization": quantization,
        "hf_overrides": hf_overrides,
        "load_s": load_s,
        "iters": args.iters,
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "total_output_tokens": total_tokens,
        "total_s": sum(latencies),
        "mean_latency_s": sum(latencies) / len(latencies),
        "output_tps": total_tokens / sum(latencies),
        "latencies_s": latencies,
        "last_text": text,
    }

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
