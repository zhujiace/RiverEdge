#!/usr/bin/env python3
"""Run full GSM8K with natural-length outputs on RiverEdge vLLM modes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

NUMBER_PATTERN = r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?"
STRICT_ANSWER_RE = re.compile(
    rf"The final answer is\s+({NUMBER_PATTERN})", re.IGNORECASE
)
FLEXIBLE_ANSWER_RE = re.compile(NUMBER_PATTERN)
QUESTION_TEMPLATE = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" where [answer] '
    "is the response to the problem.\n"
)


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
    parser.add_argument(
        "--dataset-jsonl",
        default=(
            "/workspace/vllm/RiverEdge/experiments/11_vllm_batch_cudagraph/"
            "gsm8k_full/gsm8k_test.jsonl"
        ),
    )
    parser.add_argument(
        "--task-yaml",
        default=(
            "/workspace/vllm/rivier/lm-evaluation/lm_eval/tasks/gsm8k/"
            "gsm8k-cot-llama.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/workspace/vllm/RiverEdge/experiments/11_vllm_batch_cudagraph/"
            "gsm8k_full"
        ),
    )
    parser.add_argument("--run-name", default="gsm8k_full_cudagraph")
    parser.add_argument("--modes", default="full_fp,static_ptq_tail")
    parser.add_argument("--execution-mode", choices=["eager", "cudagraph"], default="cudagraph")
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup-size", type=int, default=16)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument(
        "--cudagraph-capture-sizes",
        type=parse_int_list,
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--inspect-prompts", action="store_true")
    parser.add_argument("--worker-mode", default="")
    return parser.parse_args()


def split_values(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def load_records(path: Path, max_samples: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if max_samples > 0 and len(records) >= max_samples:
                    break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def load_fewshot_samples(path: Path) -> list[dict[str, str]]:
    import yaml

    task = yaml.safe_load(path.read_text(encoding="utf-8"))
    samples = task["fewshot_config"]["samples"]
    num_fewshot = int(task.get("num_fewshot", len(samples)))
    if len(samples) < num_fewshot:
        raise ValueError("Task YAML has fewer samples than num_fewshot")
    return samples[:num_fewshot]


def make_prompts(
    tokenizer: Any,
    records: list[dict[str, Any]],
    fewshot_samples: list[dict[str, str]],
) -> list[str]:
    base_messages = []
    for sample in fewshot_samples:
        base_messages.extend(
            [
                {
                    "role": "user",
                    "content": QUESTION_TEMPLATE.format(question=sample["question"]),
                },
                {"role": "assistant", "content": sample["target"]},
            ]
        )

    prompts = []
    for record in records:
        messages = list(base_messages)
        messages.append(
            {
                "role": "user",
                "content": QUESTION_TEMPLATE.format(question=record["question"]),
            }
        )
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def percentile(values: list[float] | list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def normalize_number(value: str | None) -> Decimal | None:
    if value is None:
        return None
    cleaned = value.replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def reference_answer(answer: str) -> str:
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def extract_answers(text: str) -> tuple[str | None, str | None]:
    strict_matches = STRICT_ANSWER_RE.findall(text)
    flexible_matches = FLEXIBLE_ANSWER_RE.findall(text)
    strict = strict_matches[-1] if strict_matches else None
    flexible = flexible_matches[-1] if flexible_matches else None
    return strict, flexible


def riveredge_phase_trace(llm: Any) -> list[str]:
    try:
        engine_core_client = llm.llm_engine.engine_core
        engine_core = getattr(engine_core_client, "engine_core", engine_core_client)
        model = engine_core.model_executor.driver_worker.model_runner.model
    except AttributeError:
        return []
    while not hasattr(model, "_traced_phases") and hasattr(model, "model"):
        model = model.model
    return sorted(getattr(model, "_traced_phases", ()))


def make_compilation_config(args: argparse.Namespace) -> tuple[bool, Any]:
    if args.execution_mode == "eager":
        return True, None

    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=sorted(set(args.cudagraph_capture_sizes)),
        cudagraph_num_of_warmups=1,
    )
    return False, config


def request_timing(request: Any, output_tokens: int) -> dict[str, float | None]:
    metrics = getattr(request, "metrics", None)
    if metrics is None:
        return {"e2e_s": None, "ttft_s": None, "tpot_s": None}
    arrival = getattr(metrics, "arrival_time", None)
    first = getattr(metrics, "first_token_time", None)
    last = getattr(metrics, "last_token_time", None)
    if last is None:
        last = getattr(metrics, "finished_time", None)
    e2e = last - arrival if last is not None and arrival is not None else None
    ttft = first - arrival if first is not None and arrival is not None else None
    tpot = None
    if last is not None and first is not None and output_tokens > 1:
        tpot = (last - first) / (output_tokens - 1)
    return {"e2e_s": e2e, "ttft_s": ttft, "tpot_s": tpot}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_prompts(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    records = load_records(Path(args.dataset_jsonl), args.max_samples)
    fewshot = load_fewshot_samples(Path(args.task_yaml))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = make_prompts(tokenizer, records, fewshot)
    lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    print(
        json.dumps(
            {
                "samples": len(records),
                "fewshot": len(fewshot),
                "prompt_tokens": distribution(lengths),
                "max_model_len": args.max_model_len,
                "max_tokens_safety_cap": args.max_tokens,
                "longest_prompt_plus_cap": max(lengths) + args.max_tokens,
            },
            indent=2,
        )
    )


def summarize_mode(
    args: argparse.Namespace,
    mode: str,
    load_s: float,
    warmup_s: float,
    phase_trace: list[str],
    samples: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    total_s = sum(float(chunk["latency_s"]) for chunk in chunks)
    input_lengths = [int(sample["prompt_tokens"]) for sample in samples]
    output_lengths = [int(sample["output_tokens"]) for sample in samples]
    total_input = sum(input_lengths)
    total_output = sum(output_lengths)
    strict_correct = sum(bool(sample["strict_correct"]) for sample in samples)
    flexible_correct = sum(bool(sample["flexible_correct"]) for sample in samples)
    finish_reasons = Counter(str(sample["finish_reason"]) for sample in samples)
    cap_hits = sum(
        sample["finish_reason"] == "length" and sample["output_tokens"] >= args.max_tokens
        for sample in samples
    )
    e2e_values = [sample["e2e_s"] for sample in samples if sample["e2e_s"] is not None]
    ttft_values = [sample["ttft_s"] for sample in samples if sample["ttft_s"] is not None]
    tpot_values = [sample["tpot_s"] for sample in samples if sample["tpot_s"] is not None]

    return {
        "model_path": args.model,
        "dataset": "gsm8k/main/test",
        "task": "gsm8k_cot_llama",
        "mode": mode,
        "execution_mode": args.execution_mode,
        "compilation": (
            "enforce_eager"
            if args.execution_mode == "eager"
            else "mode=NONE,cudagraph_mode=FULL_DECODE_ONLY"
        ),
        "exit_layer": args.exit_layer,
        "batch_size": args.batch_size,
        "num_samples": len(samples),
        "num_chunks": len(chunks),
        "fewshot": 8,
        "max_model_len": args.max_model_len,
        "max_tokens_safety_cap": args.max_tokens,
        "natural_eos_enabled": True,
        "ignore_eos": False,
        "min_tokens": 0,
        "prefix_caching": False,
        "chunked_prefill": False,
        "inductor": False,
        "cudagraph_capture_sizes": (
            [] if args.execution_mode == "eager" else args.cudagraph_capture_sizes
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "load_s": load_s,
        "warmup_s": warmup_s,
        "benchmark_s": total_s,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "output_tps": total_output / total_s,
        "total_token_tps": (total_input + total_output) / total_s,
        "requests_per_s": len(samples) / total_s,
        "input_output_token_ratio": total_input / total_output,
        "strict_correct": strict_correct,
        "strict_accuracy": strict_correct / len(samples),
        "flexible_correct": flexible_correct,
        "flexible_accuracy": flexible_correct / len(samples),
        "strict_extract_failures": sum(sample["strict_prediction"] is None for sample in samples),
        "flexible_extract_failures": sum(sample["flexible_prediction"] is None for sample in samples),
        "finish_reasons": dict(finish_reasons),
        "max_tokens_cap_hits": cap_hits,
        "max_tokens_cap_hit_rate": cap_hits / len(samples),
        "prompt_tokens": distribution(input_lengths),
        "output_tokens": distribution(output_lengths),
        "request_e2e_s": distribution(e2e_values),
        "request_ttft_s": distribution(ttft_values),
        "request_tpot_s": distribution(tpot_values),
        "chunk_latency_s": distribution([chunk["latency_s"] for chunk in chunks]),
        "chunk_output_tps": distribution([chunk["output_tps"] for chunk in chunks]),
        "phase_trace": phase_trace,
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams

    mode = args.worker_mode
    if mode not in {"full_fp", "static_ptq_tail"}:
        raise ValueError(f"Unsupported mode: {mode}")

    records = load_records(Path(args.dataset_jsonl), args.max_samples)
    fewshot = load_fewshot_samples(Path(args.task_yaml))
    enforce_eager, compilation_config = make_compilation_config(args)

    load_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_model_len * args.batch_size,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        trust_remote_code=False,
        disable_log_stats=True,
        hf_overrides={
            "river_edge_mode": mode,
            "river_edge_exit_layer": args.exit_layer,
        },
    )
    load_s = time.perf_counter() - load_started

    tokenizer = llm.get_tokenizer()
    prompts = make_prompts(tokenizer, records, fewshot)
    prompt_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    too_long = [
        (index, length)
        for index, length in enumerate(prompt_lengths)
        if length + args.max_tokens > args.max_model_len
    ]
    if too_long:
        raise ValueError(
            f"{len(too_long)} prompts exceed max_model_len with safety cap; "
            f"longest prompt={max(prompt_lengths)}, max_model_len={args.max_model_len}"
        )

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    warmup_sampling = SamplingParams(
        max_tokens=min(args.warmup_max_tokens, args.max_tokens), temperature=0.0
    )
    warmup_count = min(args.warmup_size, len(prompts))
    warmup_started = time.perf_counter()
    if warmup_count:
        llm.generate(prompts[:warmup_count], warmup_sampling, use_tqdm=False)
        torch.cuda.synchronize()
    warmup_s = time.perf_counter() - warmup_started

    output_dir = Path(args.output_dir)
    sample_path = output_dir / f"{args.run_name}.{mode}.samples.jsonl"
    chunk_path = output_dir / f"{args.run_name}.{mode}.chunks.csv"
    summary_path = output_dir / f"{args.run_name}.{mode}.summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    with sample_path.open("w", encoding="utf-8") as sample_handle:
        for chunk_index, start_index in enumerate(range(0, len(records), args.batch_size)):
            end_index = min(start_index + args.batch_size, len(records))
            chunk_prompts = prompts[start_index:end_index]
            torch.cuda.synchronize()
            chunk_started = time.perf_counter()
            outputs = llm.generate(chunk_prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            chunk_latency = time.perf_counter() - chunk_started

            chunk_sample_rows = []
            for offset, request in enumerate(outputs):
                record_index = start_index + offset
                record = records[record_index]
                completion = request.outputs[0]
                output_tokens = len(completion.token_ids)
                strict_prediction, flexible_prediction = extract_answers(completion.text)
                target = reference_answer(record["answer"])
                target_number = normalize_number(target)
                strict_number = normalize_number(strict_prediction)
                flexible_number = normalize_number(flexible_prediction)
                timing = request_timing(request, output_tokens)
                row = {
                    "dataset_index": record.get("dataset_index", record_index),
                    "chunk_index": chunk_index,
                    "position_in_chunk": offset,
                    "mode": mode,
                    "question": record["question"],
                    "reference_reasoning": record["answer"],
                    "reference_answer": target,
                    "prompt_tokens": prompt_lengths[record_index],
                    "output_tokens": output_tokens,
                    "generated_text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "strict_prediction": strict_prediction,
                    "flexible_prediction": flexible_prediction,
                    "strict_correct": strict_number is not None and strict_number == target_number,
                    "flexible_correct": flexible_number is not None and flexible_number == target_number,
                    **timing,
                }
                chunk_sample_rows.append(row)
                sample_rows.append(row)
                sample_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            sample_handle.flush()

            chunk_output_tokens = sum(row["output_tokens"] for row in chunk_sample_rows)
            chunk_input_tokens = sum(row["prompt_tokens"] for row in chunk_sample_rows)
            finish_counts = Counter(str(row["finish_reason"]) for row in chunk_sample_rows)
            chunk_row = {
                "chunk_index": chunk_index,
                "start_index": start_index,
                "end_index": end_index,
                "batch_size": len(chunk_sample_rows),
                "input_tokens": chunk_input_tokens,
                "output_tokens": chunk_output_tokens,
                "latency_s": chunk_latency,
                "output_tps": chunk_output_tokens / chunk_latency,
                "total_token_tps": (chunk_input_tokens + chunk_output_tokens) / chunk_latency,
                "strict_accuracy": sum(row["strict_correct"] for row in chunk_sample_rows) / len(chunk_sample_rows),
                "flexible_accuracy": sum(row["flexible_correct"] for row in chunk_sample_rows) / len(chunk_sample_rows),
                "min_output_tokens": min(row["output_tokens"] for row in chunk_sample_rows),
                "max_output_tokens": max(row["output_tokens"] for row in chunk_sample_rows),
                "finish_reasons": json.dumps(dict(finish_counts), sort_keys=True),
            }
            chunk_rows.append(chunk_row)
            write_csv(chunk_path, chunk_rows)

            phase_trace = riveredge_phase_trace(llm)
            partial_summary = summarize_mode(
                args, mode, load_s, warmup_s, phase_trace, sample_rows, chunk_rows
            )
            write_json(summary_path, {"complete": end_index == len(records), "result": partial_summary})
            print(
                f"[{mode}] chunk {chunk_index + 1}/{math.ceil(len(records) / args.batch_size)} "
                f"samples={end_index}/{len(records)} output_tps={chunk_row['output_tps']:.2f} "
                f"output_tokens={chunk_output_tokens}",
                flush=True,
            )

    result = summarize_mode(
        args,
        mode,
        load_s,
        warmup_s,
        riveredge_phase_trace(llm),
        sample_rows,
        chunk_rows,
    )
    write_json(summary_path, {"complete": True, "result": result})

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--dataset-jsonl",
        args.dataset_jsonl,
        "--task-yaml",
        args.task_yaml,
        "--output-dir",
        args.output_dir,
        "--run-name",
        args.run_name,
        "--modes",
        mode,
        "--execution-mode",
        args.execution_mode,
        "--exit-layer",
        str(args.exit_layer),
        "--batch-size",
        str(args.batch_size),
        "--max-samples",
        str(args.max_samples),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-size",
        str(args.warmup_size),
        "--warmup-max-tokens",
        str(args.warmup_max_tokens),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cudagraph-capture-sizes",
        ",".join(str(item) for item in args.cudagraph_capture_sizes),
        "--worker-mode",
        mode,
    ]
    return command


def write_combined(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    output_dir = Path(args.output_dir)
    write_json(output_dir / f"{args.run_name}.json", {"results": results})
    rows = []
    for result in results:
        rows.append(
            {
                "mode": result["mode"],
                "num_samples": result["num_samples"],
                "batch_size": result["batch_size"],
                "benchmark_s": result["benchmark_s"],
                "total_input_tokens": result["total_input_tokens"],
                "total_output_tokens": result["total_output_tokens"],
                "output_tps": result["output_tps"],
                "total_token_tps": result["total_token_tps"],
                "requests_per_s": result["requests_per_s"],
                "strict_accuracy": result["strict_accuracy"],
                "flexible_accuracy": result["flexible_accuracy"],
                "max_tokens_cap_hits": result["max_tokens_cap_hits"],
                "prompt_tokens_mean": result["prompt_tokens"]["mean"],
                "prompt_tokens_p90": result["prompt_tokens"]["p90"],
                "output_tokens_mean": result["output_tokens"]["mean"],
                "output_tokens_p90": result["output_tokens"]["p90"],
                "phase_trace": ",".join(result["phase_trace"]),
            }
        )
    write_csv(output_dir / f"{args.run_name}.csv", rows)


def run_parent(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in split_values(args.modes):
        subprocess.run(worker_command(args, mode), env=os.environ.copy(), check=True)
        summary_path = output_dir / f"{args.run_name}.{mode}.summary.json"
        result = json.loads(summary_path.read_text(encoding="utf-8"))["result"]
        results.append(result)
        write_combined(args, results)
    return results


def main() -> None:
    args = parse_args()
    if args.inspect_prompts:
        inspect_prompts(args)
        return
    if args.worker_mode:
        result = run_worker(args)
        print(json.dumps({"result": result}, indent=2, ensure_ascii=False))
        return
    results = run_parent(args)
    print(json.dumps({"results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
