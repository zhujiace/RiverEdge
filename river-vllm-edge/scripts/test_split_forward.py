#!/usr/bin/env python3
"""Smoke and speed test for the RiverEdge PyTorch routed reference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/vllm")
    parser.add_argument(
        "--model-path",
        default="/models/Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16",
    )
    parser.add_argument("--output-json", default="/workspace/vllm/RiverEdge/experiments/04_split_reference/routed_reference_result.json")
    parser.add_argument("--output-csv", default="/workspace/vllm/RiverEdge/experiments/04_split_reference/routed_reference_steps.csv")
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--quant-backend", choices=("n", "d", "g", "t"), default="t")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force-route", choices=("auto", "ptq", "fp"), default="auto")
    parser.add_argument("--modes", default="", help="Comma/space separated modes from auto,ptq,fp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    ext_root = repo_root / "RiverEdge" / "river-vllm-edge"
    sys.path.insert(0, str(ext_root))

    from river_vllm_ext.models.split_model_reference import RiverRoutedDualTailModel, load_river_model

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = load_river_model(
        args.model_path,
        repo_root=repo_root,
        dtype=dtype,
        quant_backend=args.quant_backend,
    )
    routed = RiverRoutedDualTailModel(
        model,
        exit_layer=args.exit_layer,
        route_threshold=args.route_threshold,
    )

    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (args.batch_size, args.prompt_len),
        device="cuda:0",
        generator=generator,
    )
    attention_mask = torch.ones((args.batch_size, args.prompt_len), dtype=torch.long, device="cuda:0")

    modes = [item for item in args.modes.replace(",", " ").split() if item] or [args.force_route]
    invalid_modes = [mode for mode in modes if mode not in {"auto", "ptq", "fp"}]
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {invalid_modes}")

    result = {
        "model_path": args.model_path,
        "exit_layer": args.exit_layer,
        "route_threshold": args.route_threshold,
        "modes": modes,
        "prompt_len": args.prompt_len,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "quant_backend": args.quant_backend,
        "dtype": args.dtype,
        "runs": [],
    }

    step_rows = []
    for mode in modes:
        force_route = None if mode == "auto" else mode
        output_tokens, stats = routed.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            attention_mask=attention_mask,
            force_route=force_route,
        )
        run = {
            "mode": mode,
            "generated_shape": list(output_tokens.shape),
            "prefill_s": stats.prefill_s,
            "decode_s": stats.decode_s,
            "decode_steps": stats.decode_steps,
            "decode_tps": stats.decode_tps,
            "route_counts": stats.route_counts,
            "avg_route_score": sum(stats.route_scores) / len(stats.route_scores) if stats.route_scores else None,
            "min_route_score": min(stats.route_scores) if stats.route_scores else None,
            "max_route_score": max(stats.route_scores) if stats.route_scores else None,
        }
        result["runs"].append(run)
        for index, score in enumerate(stats.route_scores):
            step_rows.append({"mode": mode, "step": index, "route_score": score})

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "step", "route_score"])
        writer.writeheader()
        writer.writerows(step_rows)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
