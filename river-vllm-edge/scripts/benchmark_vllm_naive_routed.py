#!/usr/bin/env python3
"""Naive routed RiverEdge benchmark using separate vLLM FP/PTQ engines.

This is intentionally a baseline, not the final route-aware scheduler. Mixed
routes are executed as two sequential vLLM batches, one on the PTQ-tail model
and one on the FP model, then merged only for accounting.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-model", default="/models/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--ptq-model",
        default="/models/Llama-3.1-8B-Instruct-layer4-32-torchao-hqq-fused",
    )
    parser.add_argument("--policies", default="all_fp,all_ptq,random,ptq_heavy")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--prompt-tokens-target", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--single-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--dual-gpu-memory-utilization", type=float, default=0.32)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ptq-heavy-ratio", type=float, default=0.75)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output-json",
        default="/workspace/vllm/RiverEdge/experiments/09_torchao_serialized_riveredge/p7_naive_routed_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        default="/workspace/vllm/RiverEdge/experiments/09_torchao_serialized_riveredge/p7_naive_routed_summary.csv",
    )
    parser.add_argument(
        "--route-csv",
        default="/workspace/vllm/RiverEdge/experiments/09_torchao_serialized_riveredge/p7_naive_routed_routes.csv",
    )
    parser.add_argument("--worker-policy", default="")
    parser.add_argument("--worker-output", default="")
    return parser.parse_args()


def make_prompts(num_prompts: int, target_tokens: int) -> list[str]:
    base = "The quick brown fox jumps over the lazy dog near the river edge. "
    repeated = (base * max(1, target_tokens // 12 + 1)).strip()
    return [f"{repeated} Request {idx}." for idx in range(num_prompts)]


def assign_routes(policy: str, num_prompts: int, seed: int, ptq_heavy_ratio: float) -> list[dict]:
    rng = random.Random(seed)
    routes = []
    for idx in range(num_prompts):
        if policy == "all_fp":
            score = 0.0
            route = "fp"
        elif policy == "all_ptq":
            score = 1.0
            route = "ptq"
        elif policy == "random":
            score = rng.random()
            route = "ptq" if score >= 0.5 else "fp"
        elif policy == "ptq_heavy":
            score = rng.random()
            route = "ptq" if score < ptq_heavy_ratio else "fp"
        else:
            raise ValueError(f"Unsupported policy: {policy}")
        routes.append({"prompt_index": idx, "route": route, "score": score})
    return routes


def build_llm(model_path: str, args: argparse.Namespace, gpu_memory_utilization: float):
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=args.num_prompts,
        max_num_batched_tokens=args.max_model_len * args.num_prompts,
        enable_chunked_prefill=False,
        trust_remote_code=False,
        disable_log_stats=True,
    )


def run_policy(args: argparse.Namespace, policy: str) -> dict:
    import torch
    from vllm import SamplingParams

    prompts = make_prompts(args.num_prompts, args.prompt_tokens_target)
    routes = assign_routes(policy, args.num_prompts, args.seed, args.ptq_heavy_ratio)
    need_fp = any(item["route"] == "fp" for item in routes)
    need_ptq = any(item["route"] == "ptq" for item in routes)
    dual = need_fp and need_ptq
    gpu_util = args.dual_gpu_memory_utilization if dual else args.single_gpu_memory_utilization

    load_start = time.perf_counter()
    fp_llm = build_llm(args.fp_model, args, gpu_util) if need_fp else None
    ptq_llm = build_llm(args.ptq_model, args, gpu_util) if need_ptq else None
    load_s = time.perf_counter() - load_start

    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    if args.warmup:
        if fp_llm is not None:
            fp_llm.generate([prompts[0]], sampling)
        if ptq_llm is not None:
            ptq_llm.generate([prompts[0]], sampling)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    groups = {"fp": [], "ptq": []}
    for route in routes:
        groups[route["route"]].append((route["prompt_index"], prompts[route["prompt_index"]]))

    outputs_by_index = {}
    route_elapsed = {"fp": 0.0, "ptq": 0.0}
    start = time.perf_counter()
    if groups["ptq"]:
        ptq_start = time.perf_counter()
        outputs = ptq_llm.generate([prompt for _, prompt in groups["ptq"]], sampling)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        route_elapsed["ptq"] = time.perf_counter() - ptq_start
        for (idx, _), output in zip(groups["ptq"], outputs):
            outputs_by_index[idx] = output
    if groups["fp"]:
        fp_start = time.perf_counter()
        outputs = fp_llm.generate([prompt for _, prompt in groups["fp"]], sampling)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        route_elapsed["fp"] = time.perf_counter() - fp_start
        for (idx, _), output in zip(groups["fp"], outputs):
            outputs_by_index[idx] = output
    total_s = time.perf_counter() - start

    total_output_tokens = sum(
        len(outputs_by_index[idx].outputs[0].token_ids) for idx in sorted(outputs_by_index)
    )
    route_counts = {
        "fp": sum(1 for item in routes if item["route"] == "fp"),
        "ptq": sum(1 for item in routes if item["route"] == "ptq"),
    }

    del fp_llm, ptq_llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "policy": policy,
        "num_prompts": args.num_prompts,
        "max_new_tokens": args.max_new_tokens,
        "total_output_tokens": total_output_tokens,
        "route_counts": route_counts,
        "ptq_ratio": route_counts["ptq"] / args.num_prompts,
        "load_s": load_s,
        "route_elapsed_s": route_elapsed,
        "total_s": total_s,
        "output_tps": total_output_tokens / total_s if total_s > 0 else 0.0,
        "gpu_memory_utilization": gpu_util,
        "dual_engine": dual,
        "routes": routes,
    }


def run_policy_subprocess(args: argparse.Namespace, policy: str) -> dict:
    worker_output = Path(args.output_json).with_name(f"{Path(args.output_json).stem}.{policy}.json")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--fp-model",
        args.fp_model,
        "--ptq-model",
        args.ptq_model,
        "--policies",
        policy,
        "--num-prompts",
        str(args.num_prompts),
        "--prompt-tokens-target",
        str(args.prompt_tokens_target),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--single-gpu-memory-utilization",
        str(args.single_gpu_memory_utilization),
        "--dual-gpu-memory-utilization",
        str(args.dual_gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--ptq-heavy-ratio",
        str(args.ptq_heavy_ratio),
        "--warmup",
        str(args.warmup),
        "--worker-policy",
        policy,
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
    output_json.write_text(
        json.dumps(
            {
                "note": "Naive routed baseline: mixed policies execute PTQ and FP batches sequentially on separate vLLM engines.",
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy",
                "num_prompts",
                "max_new_tokens",
                "total_output_tokens",
                "fp_count",
                "ptq_count",
                "ptq_ratio",
                "dual_engine",
                "gpu_memory_utilization",
                "load_s",
                "fp_elapsed_s",
                "ptq_elapsed_s",
                "total_s",
                "output_tps",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "policy": row["policy"],
                    "num_prompts": row["num_prompts"],
                    "max_new_tokens": row["max_new_tokens"],
                    "total_output_tokens": row["total_output_tokens"],
                    "fp_count": row["route_counts"]["fp"],
                    "ptq_count": row["route_counts"]["ptq"],
                    "ptq_ratio": row["ptq_ratio"],
                    "dual_engine": row["dual_engine"],
                    "gpu_memory_utilization": row["gpu_memory_utilization"],
                    "load_s": row["load_s"],
                    "fp_elapsed_s": row["route_elapsed_s"]["fp"],
                    "ptq_elapsed_s": row["route_elapsed_s"]["ptq"],
                    "total_s": row["total_s"],
                    "output_tps": row["output_tps"],
                }
            )

    route_csv = Path(args.route_csv)
    with route_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", "prompt_index", "route", "score"])
        writer.writeheader()
        for row in results:
            for route in row["routes"]:
                writer.writerow({"policy": row["policy"], **route})


def main() -> None:
    args = parse_args()
    if args.worker_policy:
        result = run_policy(args, args.worker_policy)
        worker_output = Path(args.worker_output)
        worker_output.parent.mkdir(parents=True, exist_ok=True)
        worker_output.write_text(json.dumps({"result": result}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"result": result}, indent=2))
        return

    policies = [item for item in args.policies.replace(",", " ").split() if item]
    results = [run_policy_subprocess(args, policy) for policy in policies]
    write_outputs(args, results)
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
