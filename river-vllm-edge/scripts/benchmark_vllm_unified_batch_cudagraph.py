#!/usr/bin/env python3
"""Benchmark RiverEdge static decode modes across batch sizes and CUDA Graph."""

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


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq",
    )
    parser.add_argument("--modes", default="full_fp,static_ptq_tail")
    parser.add_argument(
        "--execution-modes",
        default="eager,cudagraph",
    )
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--prompt-tokens-target", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument(
        "--output-json",
        default=(
            "/workspace/vllm/RiverEdge/experiments/11_vllm_batch_cudagraph/"
            "batch_cudagraph_matrix.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "/workspace/vllm/RiverEdge/experiments/11_vllm_batch_cudagraph/"
            "batch_cudagraph_matrix.csv"
        ),
    )
    parser.add_argument("--worker-mode", default="")
    parser.add_argument("--worker-execution-mode", default="")
    parser.add_argument("--worker-output", default="")
    return parser.parse_args()


def split_values(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def make_prompt(target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog near the river edge. "
    return (base * max(1, target_tokens // 12 + 1)).strip()


def riveredge_phase_trace(llm) -> list[str]:
    try:
        engine_core_client = llm.llm_engine.engine_core
        engine_core = getattr(engine_core_client, "engine_core", engine_core_client)
        model = engine_core.model_executor.driver_worker.model_runner.model
    except AttributeError:
        return []
    while not hasattr(model, "_traced_phases") and hasattr(model, "model"):
        model = model.model
    return sorted(getattr(model, "_traced_phases", ()))


def make_compilation_config(execution_mode: str, batch_sizes: list[int]):
    if execution_mode == "eager":
        return True, None
    if execution_mode != "cudagraph":
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=sorted(set(batch_sizes)),
        cudagraph_num_of_warmups=1,
    )
    return False, config


def run_worker(args: argparse.Namespace) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    mode = args.worker_mode
    execution_mode = args.worker_execution_mode
    if mode not in {"full_fp", "static_fp_tail", "static_ptq_tail"}:
        raise ValueError(f"Unsupported RiverEdge mode: {mode}")

    batch_sizes = sorted(set(args.batch_sizes))
    max_batch_size = max(batch_sizes)
    enforce_eager, compilation_config = make_compilation_config(
        execution_mode, batch_sizes
    )

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=max_batch_size,
        max_num_batched_tokens=args.max_model_len * max_batch_size,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        trust_remote_code=False,
        disable_log_stats=True,
        hf_overrides={
            "river_edge_mode": mode,
            "river_edge_exit_layer": args.exit_layer,
        },
    )
    load_s = time.perf_counter() - started

    prompt = make_prompt(args.prompt_tokens_target)
    tokenizer = llm.get_tokenizer()
    prompt_tokens = len(tokenizer.encode(prompt))
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    rows = []
    for batch_size in batch_sizes:
        prompts = [prompt] * batch_size
        for _ in range(args.warmup):
            llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()

        latencies = []
        total_output_tokens = 0
        last_first_token_ids: list[int] = []
        for _ in range(args.iters):
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            output_tokens = sum(
                len(request.outputs[0].token_ids) for request in outputs
            )
            latencies.append(latency)
            total_output_tokens += output_tokens
            last_first_token_ids = list(outputs[0].outputs[0].token_ids)

        total_s = sum(latencies)
        rows.append(
            {
                "mode": mode,
                "execution_mode": execution_mode,
                "enforce_eager": enforce_eager,
                "exit_layer": args.exit_layer,
                "batch_size": batch_size,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "warmup": args.warmup,
                "iters": args.iters,
                "total_output_tokens": total_output_tokens,
                "total_s": total_s,
                "mean_batch_latency_s": total_s / len(latencies),
                "output_tps": total_output_tokens / total_s,
                "latencies_s": latencies,
                "last_first_token_ids": last_first_token_ids,
            }
        )

    result = {
        "model_path": args.model,
        "mode": mode,
        "execution_mode": execution_mode,
        "enforce_eager": enforce_eager,
        "compilation": (
            "none"
            if enforce_eager
            else "mode=NONE,cudagraph_mode=FULL_DECODE_ONLY"
        ),
        "cudagraph_capture_sizes": [] if enforce_eager else batch_sizes,
        "load_s": load_s,
        "phase_trace": riveredge_phase_trace(llm),
        "rows": rows,
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def worker_command(
    args: argparse.Namespace,
    mode: str,
    execution_mode: str,
    worker_output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--modes",
        mode,
        "--execution-modes",
        execution_mode,
        "--batch-sizes",
        ",".join(str(item) for item in args.batch_sizes),
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
        "--worker-execution-mode",
        execution_mode,
        "--worker-output",
        str(worker_output),
    ]


def run_parent(args: argparse.Namespace) -> list[dict]:
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for execution_mode in split_values(args.execution_modes):
        for mode in split_values(args.modes):
            worker_output = output_json.with_name(
                f"{output_json.stem}.{execution_mode}.{mode}.json"
            )
            subprocess.run(
                worker_command(args, mode, execution_mode, worker_output),
                env=os.environ.copy(),
                check=True,
            )
            result = json.loads(worker_output.read_text(encoding="utf-8"))["result"]
            results.append(result)
            write_outputs(args, results)
    return results


def write_outputs(args: argparse.Namespace, results: list[dict]) -> None:
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"results": results}, indent=2) + "\n")

    fieldnames = [
        "mode",
        "execution_mode",
        "enforce_eager",
        "exit_layer",
        "batch_size",
        "prompt_tokens",
        "max_new_tokens",
        "warmup",
        "iters",
        "total_output_tokens",
        "total_s",
        "mean_batch_latency_s",
        "output_tps",
        "load_s",
        "phase_trace",
        "model_path",
    ]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result["rows"]:
                writer.writerow(
                    {
                        **{key: row.get(key) for key in fieldnames},
                        "load_s": result["load_s"],
                        "phase_trace": ",".join(result["phase_trace"]),
                        "model_path": result["model_path"],
                    }
                )


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        result = run_worker(args)
        worker_output = Path(args.worker_output)
        worker_output.parent.mkdir(parents=True, exist_ok=True)
        worker_output.write_text(json.dumps({"result": result}, indent=2) + "\n")
        print(json.dumps({"result": result}, indent=2))
        return

    results = run_parent(args)
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
