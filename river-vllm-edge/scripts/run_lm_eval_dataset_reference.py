#!/usr/bin/env python3
"""Run RiverEdge routed reference on prompts from lm-evaluation tasks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--modes", default="auto,ptq,fp")
    parser.add_argument("--quant-backend", choices=("n", "d", "g", "t"), default="t")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output-json", default="/workspace/vllm/RiverEdge/experiments/04_split_reference/lm_eval_dataset_routed_summary.json")
    parser.add_argument("--output-csv", default="/workspace/vllm/RiverEdge/experiments/04_split_reference/lm_eval_dataset_routed_samples.csv")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


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
    if str(lm_eval_dir) not in sys.path:
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
            prompts.append(
                {
                    "task": task_name,
                    "doc_id": doc_id,
                    "prompt": str(task.doc_to_text(doc)),
                }
            )
    return prompts


def get_tokenizer(repo_root: Path, model_path: str):
    lm_eval_dir = repo_root / "rivier" / "lm-evaluation"
    if str(lm_eval_dir) not in sys.path:
        sys.path.insert(0, str(lm_eval_dir))
    from lm_eval.transformers_extra.tokenizer_utils import load_tokenizer_with_fallback

    tokenizer = load_tokenizer_with_fallback(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_prompt(tokenizer, prompt: str, max_prompt_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] == 0:
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        input_ids = torch.tensor([[eos_id]], dtype=torch.long)
    if max_prompt_tokens > 0 and input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
    attention_mask = torch.ones_like(input_ids)
    prompt_tokens = int(input_ids.shape[1])
    return input_ids.to("cuda:0"), attention_mask.to("cuda:0"), prompt_tokens


def token_match_rate(candidate: List[int], reference: List[int]) -> float:
    if not reference:
        return 0.0
    matches = sum(1 for a, b in zip(candidate, reference) if a == b)
    return matches / len(reference)


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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    repo_root = Path(args.repo_root).resolve()
    ext_root = repo_root / "RiverEdge" / "river-vllm-edge"
    sys.path.insert(0, str(ext_root))

    from river_vllm_ext.models.split_model_reference import RiverRoutedDualTailModel, load_river_model

    task_names = [item for item in args.tasks.replace(",", " ").split() if item]
    modes = [item for item in args.modes.replace(",", " ").split() if item]
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
    routed = RiverRoutedDualTailModel(
        model,
        exit_layer=args.exit_layer,
        route_threshold=args.route_threshold,
    )

    rows = []
    per_mode = {
        mode: {
            "decode_s": 0.0,
            "prefill_s": 0.0,
            "decode_steps": 0,
            "route_counts": {"ptq": 0, "fp": 0},
            "route_scores": [],
            "outputs": {},
        }
        for mode in modes
    }

    start_time = time.perf_counter()
    for sample_index, item in enumerate(prompts):
        input_ids, attention_mask, prompt_tokens = encode_prompt(
            tokenizer,
            item["prompt"],
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
            output_list = output_tokens[0].detach().cpu().tolist()
            per_mode[mode]["decode_s"] += stats.decode_s
            per_mode[mode]["prefill_s"] += stats.prefill_s
            per_mode[mode]["decode_steps"] += stats.decode_steps
            per_mode[mode]["route_counts"]["ptq"] += stats.route_counts.get("ptq", 0)
            per_mode[mode]["route_counts"]["fp"] += stats.route_counts.get("fp", 0)
            per_mode[mode]["route_scores"].extend(stats.route_scores)
            per_mode[mode]["outputs"][sample_index] = output_list
            rows.append(
                {
                    "sample_index": sample_index,
                    "task": item["task"],
                    "doc_id": item["doc_id"],
                    "mode": mode,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(output_list),
                    "prefill_s": stats.prefill_s,
                    "decode_s": stats.decode_s,
                    "decode_steps": stats.decode_steps,
                    "decode_tps": stats.decode_tps,
                    "ptq_routes": stats.route_counts.get("ptq", 0),
                    "fp_routes": stats.route_counts.get("fp", 0),
                    "avg_route_score": (
                        sum(stats.route_scores) / len(stats.route_scores)
                        if stats.route_scores
                        else ""
                    ),
                }
            )
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(prompts):
            print(f"processed {sample_index + 1}/{len(prompts)} samples", flush=True)

    total_wall_s = time.perf_counter() - start_time
    summary = {
        "tasks": task_names,
        "split": args.split,
        "num_samples": len(prompts),
        "model_path": args.model_path,
        "exit_layer": args.exit_layer,
        "route_threshold": args.route_threshold,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "modes": modes,
        "quant_backend": args.quant_backend,
        "dtype": args.dtype,
        "total_wall_s": total_wall_s,
        "prompt_tokens": summarize(row["prompt_tokens"] for row in rows if row["mode"] == modes[0]),
        "mode_summary": {},
        "token_match_vs_fp": {},
    }

    for mode, data in per_mode.items():
        decode_steps = data["decode_steps"]
        total_routes = data["route_counts"]["ptq"] + data["route_counts"]["fp"]
        summary["mode_summary"][mode] = {
            "prefill_s": data["prefill_s"],
            "decode_s": data["decode_s"],
            "decode_steps": decode_steps,
            "decode_tps": decode_steps / data["decode_s"] if data["decode_s"] > 0 else 0.0,
            "route_counts": data["route_counts"],
            "ptq_route_ratio": data["route_counts"]["ptq"] / total_routes if total_routes else 0.0,
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
            summary["token_match_vs_fp"][mode] = {
                "token_match_rate": summarize(rates),
                "exact_sequence_match_rate": statistics.mean(exact) if exact else 0.0,
            }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "task",
                "doc_id",
                "mode",
                "prompt_tokens",
                "generated_tokens",
                "prefill_s",
                "decode_s",
                "decode_steps",
                "decode_tps",
                "ptq_routes",
                "fp_routes",
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
