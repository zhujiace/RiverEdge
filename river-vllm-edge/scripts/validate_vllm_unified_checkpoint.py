#!/usr/bin/env python3
"""Validate one unified RiverEdge checkpoint across multiple exit layers."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--exit-layers", default="1,3,8,16,31")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument(
        "--output",
        default="/workspace/vllm/RiverEdge/experiments/10_unified_fp_hqq_checkpoint/multi_k_validation.json",
    )
    return parser.parse_args()


def get_riveredge_model(llm):
    engine_core_client = llm.llm_engine.engine_core
    engine_core = getattr(engine_core_client, "engine_core", engine_core_client)
    model = engine_core.model_executor.driver_worker.model_runner.model
    while not hasattr(model, "river_edge_mode") and hasattr(model, "model"):
        model = model.model
    if not hasattr(model, "river_edge_mode"):
        raise RuntimeError("Could not find RiverEdge unified model in vLLM engine.")
    return model


def generate(llm, sampling_params, prompt: str) -> tuple[list[int], str, float]:
    import torch

    start = time.perf_counter()
    output = llm.generate([prompt], sampling_params)[0].outputs[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return list(output.token_ids), output.text, elapsed


def main() -> None:
    from vllm import LLM, SamplingParams

    args = parse_args()
    exit_layers = [
        int(item) for item in args.exit_layers.replace(",", " ").split() if item
    ]
    if not exit_layers:
        raise ValueError("At least one exit layer is required.")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        trust_remote_code=False,
        disable_log_stats=True,
        hf_overrides={
            "river_edge_mode": "static_ptq_tail",
            "river_edge_exit_layer": exit_layers[0],
        },
    )
    model = get_riveredge_model(llm)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    prompt = "Explain in one sentence why edge inference benefits from quantization."

    model.river_edge_mode = "full_fp"
    model._traced_phases.clear()
    baseline_ids, baseline_text, baseline_s = generate(
        llm, sampling_params, prompt
    )
    rows = []
    for exit_layer in exit_layers:
        model.river_edge_mode = "static_ptq_tail"
        model.river_edge_exit_layer = exit_layer
        model._traced_phases.clear()
        token_ids, text, elapsed = generate(llm, sampling_params, prompt)
        rows.append(
            {
                "exit_layer": exit_layer,
                "token_ids": token_ids,
                "text": text,
                "elapsed_s": elapsed,
                "phase_trace": sorted(model._traced_phases),
                "first_token_matches_fp": bool(
                    baseline_ids and token_ids and baseline_ids[0] == token_ids[0]
                ),
            }
        )

    result = {
        "model": args.model,
        "prefill_policy": "full_fp",
        "baseline": {
            "mode": "full_fp",
            "token_ids": baseline_ids,
            "text": baseline_text,
            "elapsed_s": baseline_s,
        },
        "rows": rows,
        "all_first_tokens_match_fp": all(
            row["first_token_matches_fp"] for row in rows
        ),
        "all_ptq_phases_observed": all(
            row["phase_trace"] == ["decode_ptq", "prefill_fp"] for row in rows
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
