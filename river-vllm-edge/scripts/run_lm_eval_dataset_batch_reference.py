#!/usr/bin/env python3
"""Batch sweep for RiverEdge routed reference on lm-evaluation prompts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/vllm")
    parser.add_argument(
        "--model-path",
        default="/models/Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16",
    )
    parser.add_argument("--tasks", default="mmlu_abstract_algebra")
    parser.add_argument("--split", choices=("auto", "validation", "test", "train"), default="auto")
    parser.add_argument("--limit", type=int, default=0, help="0 means full selected split.")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--route-threshold", type=float, default=0.8)
    parser.add_argument("--modes", default="auto,ptq,fp")
    parser.add_argument("--quant-backend", choices=("n", "d", "g", "t"), default="t")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output-json", default="/workspace/vllm/RiverEdge/experiments/04_split_reference/mmlu_abstract_algebra_batch_sweep_k3_thr08_summary.json")
    parser.add_argument("--output-csv", default="/workspace/vllm/RiverEdge/experiments/04_split_reference/mmlu_abstract_algebra_batch_sweep_k3_thr08_batches.csv")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def summarize(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def token_match_rate(candidate: List[int], reference: List[int]) -> float:
    if not reference:
        return 0.0
    matches = sum(1 for a, b in zip(candidate, reference) if a == b)
    return matches / len(reference)


def select_docs(task, split: str):
    if split == "validation":
        return task.validation_docs()
    if split == "test":
        return task.test_docs()
    if split == "train":
        return task.training_docs()
    if task.has_validation_docs():
        return task.validation_docs()
    if task.has_test_docs():
        return task.test_docs()
    return task.training_docs()


def load_task_prompts(repo_root: Path, task_names: List[str], split: str, limit: int) -> List[Dict[str, object]]:
    lm_eval_dir = repo_root / "rivier" / "lm-evaluation"
    sys.path.insert(0, str(lm_eval_dir))
    from lm_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager("ERROR")
    tasks = get_task_dict(task_names, task_manager=task_manager)
    prompts = []
    for task_name, task in tasks.items():
        docs = list(select_docs(task, split))
        if limit > 0:
            docs = docs[:limit]
        for doc_id, doc in enumerate(docs):
            prompts.append({"task": task_name, "doc_id": doc_id, "prompt": str(task.doc_to_text(doc))})
    return prompts


def get_tokenizer(repo_root: Path, model_path: str):
    lm_eval_dir = repo_root / "rivier" / "lm-evaluation"
    sys.path.insert(0, str(lm_eval_dir))
    from lm_eval.transformers_extra.tokenizer_utils import load_tokenizer_with_fallback

    tokenizer = load_tokenizer_with_fallback(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def encode_batch(tokenizer, prompts: List[str], max_prompt_tokens: int):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens if max_prompt_tokens > 0 else None,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.shape[1] == 0:
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        input_ids = torch.full((len(prompts), 1), eos_id, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
    prompt_lengths = attention_mask.sum(dim=1).cpu().tolist()
    return input_ids.to("cuda:0"), attention_mask.to("cuda:0"), [int(x) for x in prompt_lengths]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    repo_root = Path(args.repo_root).resolve()
    ext_root = repo_root / "RiverEdge" / "river-vllm-edge"
    sys.path.insert(0, str(ext_root))

    from river_vllm_ext.models.split_model_reference import RiverRoutedDualTailModel, load_river_model

    task_names = [item for item in args.tasks.replace(",", " ").split() if item]
    modes = [item for item in args.modes.replace(",", " ").split() if item]
    batch_sizes = parse_int_list(args.batch_sizes)
    for mode in modes:
        if mode not in {"auto", "ptq", "fp"}:
            raise ValueError(f"Unsupported mode: {mode}")

    prompts = load_task_prompts(repo_root, task_names, args.split, args.limit)
    tokenizer = get_tokenizer(repo_root, args.model_path)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = load_river_model(
        args.model_path,
        repo_root=repo_root,
        dtype=dtype,
        quant_backend=args.quant_backend,
    )
    routed = RiverRoutedDualTailModel(model, exit_layer=args.exit_layer, route_threshold=args.route_threshold)

    rows = []
    summary = {
        "tasks": task_names,
        "split": args.split,
        "num_samples": len(prompts),
        "model_path": args.model_path,
        "exit_layer": args.exit_layer,
        "route_threshold": args.route_threshold,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "batch_sizes": batch_sizes,
        "modes": modes,
        "quant_backend": args.quant_backend,
        "dtype": args.dtype,
        "batch_results": {},
    }

    wall_start = time.perf_counter()
    for batch_size in batch_sizes:
        per_mode = {
            mode: {
                "prefill_s": 0.0,
                "decode_s": 0.0,
                "decode_steps": 0,
                "decode_tokens": 0,
                "route_steps": {"ptq": 0, "fp": 0},
                "route_tokens": {"ptq": 0, "fp": 0},
                "route_scores": [],
                "outputs": {},
            }
            for mode in modes
        }
        batch_count = math.ceil(len(prompts) / batch_size)
        print(f"batch_size={batch_size}: {batch_count} batches", flush=True)
        for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
            batch_items = prompts[start : start + batch_size]
            actual_batch_size = len(batch_items)
            input_ids, attention_mask, prompt_lengths = encode_batch(
                tokenizer,
                [item["prompt"] for item in batch_items],
                args.max_prompt_tokens,
            )
            for mode in modes:
                force_route = None if mode == "auto" else mode
                output_tokens, stats = routed.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    attention_mask=attention_mask,
                    force_route=force_route,
                )
                data = per_mode[mode]
                decode_tokens = actual_batch_size * stats.decode_steps
                data["prefill_s"] += stats.prefill_s
                data["decode_s"] += stats.decode_s
                data["decode_steps"] += stats.decode_steps
                data["decode_tokens"] += decode_tokens
                for route_name in ("ptq", "fp"):
                    route_steps = stats.route_counts.get(route_name, 0)
                    data["route_steps"][route_name] += route_steps
                    data["route_tokens"][route_name] += route_steps * actual_batch_size
                data["route_scores"].extend(stats.route_scores)
                for local_index in range(actual_batch_size):
                    sample_index = start + local_index
                    data["outputs"][sample_index] = output_tokens[local_index].detach().cpu().tolist()

                rows.append(
                    {
                        "batch_size": batch_size,
                        "batch_index": batch_index,
                        "actual_batch_size": actual_batch_size,
                        "mode": mode,
                        "prompt_tokens_mean": statistics.mean(prompt_lengths),
                        "prompt_tokens_max": max(prompt_lengths),
                        "prefill_s": stats.prefill_s,
                        "decode_s": stats.decode_s,
                        "decode_steps": stats.decode_steps,
                        "decode_tokens": decode_tokens,
                        "decode_step_tps": stats.decode_steps / stats.decode_s if stats.decode_s > 0 else 0.0,
                        "decode_token_tps": decode_tokens / stats.decode_s if stats.decode_s > 0 else 0.0,
                        "ptq_route_steps": stats.route_counts.get("ptq", 0),
                        "fp_route_steps": stats.route_counts.get("fp", 0),
                        "avg_route_score": (
                            sum(stats.route_scores) / len(stats.route_scores)
                            if stats.route_scores
                            else ""
                        ),
                    }
                )
            if (batch_index + 1) % 5 == 0 or batch_index + 1 == batch_count:
                print(f"  batch_size={batch_size}: processed {batch_index + 1}/{batch_count} batches", flush=True)

        summary["batch_results"][str(batch_size)] = {"mode_summary": {}, "token_match_vs_fp": {}}
        for mode, data in per_mode.items():
            route_token_total = data["route_tokens"]["ptq"] + data["route_tokens"]["fp"]
            summary["batch_results"][str(batch_size)]["mode_summary"][mode] = {
                "prefill_s": data["prefill_s"],
                "decode_s": data["decode_s"],
                "decode_steps": data["decode_steps"],
                "decode_tokens": data["decode_tokens"],
                "decode_step_tps": data["decode_steps"] / data["decode_s"] if data["decode_s"] > 0 else 0.0,
                "decode_token_tps": data["decode_tokens"] / data["decode_s"] if data["decode_s"] > 0 else 0.0,
                "route_steps": data["route_steps"],
                "route_tokens": data["route_tokens"],
                "ptq_route_token_ratio": (
                    data["route_tokens"]["ptq"] / route_token_total if route_token_total else 0.0
                ),
                "route_score": summarize(data["route_scores"]),
            }

        if "fp" in per_mode:
            fp_outputs = per_mode["fp"]["outputs"]
            for mode, data in per_mode.items():
                if mode == "fp":
                    continue
                rates = [
                    token_match_rate(data["outputs"][idx], fp_outputs[idx])
                    for idx in fp_outputs.keys()
                    if idx in data["outputs"]
                ]
                exact = [
                    1.0 if data["outputs"][idx] == fp_outputs[idx] else 0.0
                    for idx in fp_outputs.keys()
                    if idx in data["outputs"]
                ]
                summary["batch_results"][str(batch_size)]["token_match_vs_fp"][mode] = {
                    "token_match_rate": summarize(rates),
                    "exact_sequence_match_rate": statistics.mean(exact) if exact else 0.0,
                }

    summary["total_wall_s"] = time.perf_counter() - wall_start
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "batch_index",
                "actual_batch_size",
                "mode",
                "prompt_tokens_mean",
                "prompt_tokens_max",
                "prefill_s",
                "decode_s",
                "decode_steps",
                "decode_tokens",
                "decode_step_tps",
                "decode_token_tps",
                "ptq_route_steps",
                "fp_route_steps",
                "avg_route_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
