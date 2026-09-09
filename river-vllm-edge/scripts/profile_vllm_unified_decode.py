#!/usr/bin/env python3
"""Profile steady-state RiverEdge generation after model and graph warmup."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.replace(",", " ").split()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq",
    )
    parser.add_argument(
        "--mode",
        choices=("full_fp", "static_ptq_tail"),
        default="full_fp",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "cudagraph"),
        default="cudagraph",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=parse_int_list)
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--prompt-tokens-target", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument(
        "--profile-start-delay",
        type=float,
        default=0.0,
        help="Seconds to wait after warmup so an external sampler can attach.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/vllm/RiverEdge/experiments/"
            "13_low_batch_memory_validation/profile_run.json"
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.warmup < 1:
        parser.error("--warmup must be positive")
    if args.profile_iters < 1:
        parser.error("--profile-iters must be positive")
    if args.profile_start_delay < 0:
        parser.error("--profile-start-delay cannot be negative")
    return args


def make_prompt(target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog near the river edge. "
    return (base * max(1, target_tokens // 12 + 1)).strip()


def make_compilation_config(execution_mode: str, batch_sizes: list[int]):
    if execution_mode == "eager":
        return True, None

    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=batch_sizes,
        cudagraph_num_of_warmups=1,
    )
    return False, config


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


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    args = parse_args()
    batch_sizes = args.batch_sizes or [args.batch_size]
    max_batch_size = max(batch_sizes)
    enforce_eager, compilation_config = make_compilation_config(
        args.execution_mode, batch_sizes
    )
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
            "river_edge_mode": args.mode,
            "river_edge_exit_layer": args.exit_layer,
        },
    )

    prompt = make_prompt(args.prompt_tokens_target)
    prompt_tokens = len(llm.get_tokenizer().encode(prompt))
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    for batch_size in batch_sizes:
        prompts = [prompt] * batch_size
        for _ in range(args.warmup):
            llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()

    print("RIVEREDGE_PROFILE_READY", flush=True)
    if args.profile_start_delay:
        time.sleep(args.profile_start_delay)

    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()

    rows = []
    outer_range_name = f"riveredge_{args.mode}_{args.execution_mode}"
    torch.cuda.nvtx.range_push(outer_range_name)
    for batch_size in batch_sizes:
        prompts = [prompt] * batch_size
        latencies = []
        total_output_tokens = 0
        first_output_token_ids: list[int] = []
        range_name = f"{outer_range_name}_batch{batch_size}"
        torch.cuda.nvtx.range_push(range_name)
        for index in range(args.profile_iters):
            torch.cuda.nvtx.range_push(f"{range_name}_iter{index}")
            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            torch.cuda.nvtx.range_pop()
            total_output_tokens += sum(
                len(request.outputs[0].token_ids) for request in outputs
            )
            first_output_token_ids = list(outputs[0].outputs[0].token_ids)
        torch.cuda.nvtx.range_pop()
        total_s = sum(latencies)
        rows.append(
            {
                "batch_size": batch_size,
                "total_output_tokens": total_output_tokens,
                "latencies_s": latencies,
                "total_s": total_s,
                "output_tps": total_output_tokens / total_s,
                "first_output_token_ids": first_output_token_ids,
                "nvtx_range": range_name,
            }
        )
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()

    result = {
        "model": args.model,
        "mode": args.mode,
        "execution_mode": args.execution_mode,
        "batch_sizes": batch_sizes,
        "exit_layer": args.exit_layer,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "profile_iters": args.profile_iters,
        "rows": rows,
        "phase_trace": riveredge_phase_trace(llm),
        "cuda_profiler_api": args.cuda_profiler_api,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
