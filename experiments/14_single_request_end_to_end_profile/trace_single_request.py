#!/usr/bin/env python3
"""Collect a low-overhead Nsys trace for one warmed-up RiverEdge request."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")


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
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--prompt-tokens-target", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument(
        "--layer-nvtx",
        action="store_true",
        help="Add lightweight layer/component NVTX ranges after warmup.",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/vllm/RiverEdge/experiments/"
            "14_single_request_end_to_end_profile/nsys_run.json"
        ),
    )
    return parser.parse_args()


def make_prompt(target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog near the river edge. "
    return (base * max(1, target_tokens // 12 + 1)).strip()


def make_compilation_config(execution_mode: str) -> tuple[bool, Any]:
    if execution_mode == "eager":
        return True, None

    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    return False, CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[1],
        cudagraph_num_of_warmups=1,
    )


def locate_model_runner(llm: Any) -> Any:
    engine_core_client = llm.llm_engine.engine_core
    engine_core = getattr(engine_core_client, "engine_core", engine_core_client)
    return engine_core.model_executor.driver_worker.model_runner


def install_layer_nvtx(model_runner: Any, torch: Any) -> list[Any]:
    handles: list[Any] = []

    def add_range(module: Any, name: str) -> None:
        def push(_module: Any, _inputs: Any) -> None:
            torch.cuda.nvtx.range_push(name)

        def pop(_module: Any, _inputs: Any, _output: Any) -> None:
            torch.cuda.nvtx.range_pop()

        handles.append(module.register_forward_pre_hook(push))
        handles.append(module.register_forward_hook(pop))

    outer_model = model_runner.model
    inner_model = getattr(outer_model, "model", None)
    fp_layers = getattr(inner_model, "layers", None)
    if fp_layers is None:
        raise RuntimeError("cannot locate RiverEdge FP layers")

    for index, layer in enumerate(fp_layers, start=1):
        prefix = f"layer.{index:02d}.fp"
        add_range(layer, f"{prefix}.total")
        for attr, suffix in (
            ("input_layernorm", "input_norm"),
            ("self_attn", "attention"),
            ("post_attention_layernorm", "post_attn_norm"),
            ("mlp", "mlp"),
        ):
            child = getattr(layer, attr, None)
            if child is not None:
                add_range(child, f"{prefix}.{suffix}")

    ptq_layers = getattr(inner_model, "ptq_layers", None)
    if ptq_layers is not None:
        for index, layer in enumerate(ptq_layers, start=1):
            if layer.__class__.__name__ == "Identity":
                continue
            prefix = f"layer.{index:02d}.ptq"
            add_range(layer, f"{prefix}.total")
            for attr, suffix in (("self_attn", "attention"), ("mlp", "mlp")):
                child = getattr(layer, attr, None)
                if child is not None:
                    add_range(child, f"{prefix}.{suffix}")
    return handles


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    args = parse_args()
    enforce_eager, compilation_config = make_compilation_config(
        args.execution_mode
    )
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
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

    for _ in range(args.warmup):
        llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()

    hook_handles = (
        install_layer_nvtx(locate_model_runner(llm), torch)
        if args.layer_nvtx
        else []
    )
    range_name = f"single_request.{args.mode}.{args.execution_mode}"
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(range_name)
    started = time.perf_counter()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    torch.cuda.nvtx.range_pop()

    output_tokens = len(outputs[0].outputs[0].token_ids)
    result = {
        "model": args.model,
        "mode": args.mode,
        "execution_mode": args.execution_mode,
        "exit_layer": args.exit_layer,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed_s_under_nsys": elapsed_s,
        "output_tps_under_nsys": output_tokens / elapsed_s,
        "nvtx_range": range_name,
        "layer_nvtx": args.layer_nvtx,
        "note": "Nsys wall time is perturbed and is not a serving benchmark.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    torch.cuda.cudart().cudaProfilerStop()
    for handle in reversed(hook_handles):
        handle.remove()


if __name__ == "__main__":
    main()
