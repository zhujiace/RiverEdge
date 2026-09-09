#!/usr/bin/env python3
"""Aggregate single-request JSON and Nsys CSV artifacts into readable tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROFILE_FILES = {
    "full_fp_cudagraph": "full_fp.cudagraph.json",
    "static_ptq_tail_cudagraph": "static_ptq_tail.cudagraph.json",
    "full_fp_eager_layer": "full_fp.eager_layer.json",
    "static_ptq_tail_eager_layer": "static_ptq_tail.eager_layer.json",
}
GRAPH_MODES = ("full_fp_cudagraph", "static_ptq_tail_cudagraph")
LAYER_MODES = ("full_fp_eager_layer", "static_ptq_tail_eager_layer")
LAYER_RE = re.compile(
    r"^(?:gpu\.)?layer\.(?P<layer>\d+)\.(?P<path>fp|ptq)"
    r"\.(?P<component>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_profiles(root: Path) -> dict[str, dict[str, Any]]:
    return {
        label: json.loads((root / filename).read_text(encoding="utf-8"))
        for label, filename in PROFILE_FILES.items()
    }


def summary_value(profile: dict[str, Any], source: str, metric: str) -> float:
    return float(profile[source]["summary"][metric]["median"])


def request_rows(profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, profile in profiles.items():
        baseline_e2e = summary_value(profile, "baseline", "e2e_ms")
        for source in ("baseline", "profiled"):
            summary = profile[source]["summary"]
            e2e = float(summary["e2e_ms"]["median"])
            rows.append(
                {
                    "profile": label,
                    "mode": profile["mode"],
                    "execution_mode": profile["execution_mode"],
                    "detail": profile["detail"],
                    "source": source,
                    "requests": summary["requests"],
                    "prompt_tokens": profile["prompt_tokens"],
                    "output_tokens": profile["max_new_tokens"],
                    "add_request_ms": summary["add_request_ms"]["median"],
                    "e2e_ms": e2e,
                    "ttft_ms": summary["ttft_ms"]["median"],
                    "strict_tpot_ms": summary["strict_tpot_ms"]["median"],
                    "inter_token_p99_ms": summary["inter_token_p99_ms"][
                        "median"
                    ],
                    "end_to_end_output_tps": summary[
                        "end_to_end_output_tps"
                    ]["median"],
                    "instrumentation_overhead_pct": (
                        100.0 * (e2e / baseline_e2e - 1.0)
                        if source == "profiled"
                        else 0.0
                    ),
                }
            )
    return rows


def event_rows(profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, profile in profiles.items():
        for event_type in ("cpu", "gpu"):
            grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
            for event in profile["profiled"][f"{event_type}_events"]:
                grouped[(event["phase"], event["stage"])].append(
                    float(event["duration_ms"])
                )
            for (phase, stage), values in sorted(grouped.items()):
                rows.append(
                    {
                        "profile": label,
                        "mode": profile["mode"],
                        "execution_mode": profile["execution_mode"],
                        "detail": profile["detail"],
                        "event_type": event_type,
                        "phase": phase,
                        "stage": stage,
                        **describe(values),
                    }
                )
    return rows


def token_timeline_rows(
    profiles: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in GRAPH_MODES:
        profile = profiles[label]
        for source in ("baseline", "profiled"):
            for request in profile[source]["requests"]:
                previous = None
                for token_index, arrival_ms in enumerate(
                    request["token_arrival_ms"], start=1
                ):
                    rows.append(
                        {
                            "profile": label,
                            "source": source,
                            "request_index": request["request_index"],
                            "token_index": token_index,
                            "arrival_ms": arrival_ms,
                            "inter_token_ms": (
                                None
                                if previous is None
                                else arrival_ms - previous
                            ),
                            "token_id": request["token_ids"][token_index - 1],
                        }
                    )
                    previous = arrival_ms
    return rows


def engine_step_rows(
    profiles: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in GRAPH_MODES:
        profile = profiles[label]
        for source in ("baseline", "profiled"):
            for request in profile[source]["requests"]:
                for step in request["step_rows"]:
                    rows.append(
                        {
                            "profile": label,
                            "source": source,
                            "request_index": request["request_index"],
                            **step,
                        }
                    )
    return rows


def layer_event_rows(
    profiles: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in LAYER_MODES:
        profile = profiles[label]
        grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(
            list
        )
        for event in profile["profiled"]["gpu_events"]:
            match = LAYER_RE.match(event["stage"])
            if match is None:
                continue
            key = (
                event["phase"],
                int(match.group("layer")),
                match.group("path"),
                match.group("component"),
            )
            grouped[key].append(float(event["duration_ms"]))
        for (phase, layer, path, component), values in sorted(
            grouped.items()
        ):
            rows.append(
                {
                    "profile": label,
                    "mode": profile["mode"],
                    "phase": phase,
                    "layer": layer,
                    "path": path,
                    "component": component,
                    **describe(values),
                }
            )
    return rows


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "tinygemm" in lowered or "int4" in lowered:
        return "int4_weight_only_gemm"
    if "gemm" in lowered:
        return "bf16_gemm"
    if "flash_fwd" in lowered or "attention" in lowered:
        return "attention"
    if "rms_norm" in lowered:
        return "normalization"
    if "rotary_embedding" in lowered:
        return "rope"
    if "act_and_mul" in lowered or "silu" in lowered:
        return "activation"
    if "reshape_and_cache" in lowered:
        return "kv_cache"
    if "argmax" in lowered or "topk" in lowered or "sampling" in lowered:
        return "sampling"
    if "index" in lowered or "elementwise" in lowered or "slot_mapping" in lowered:
        return "index_and_elementwise"
    return "other"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def global_kernel_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    category_values: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"time_ns": 0.0, "instances": 0.0}
    )
    totals: dict[str, float] = {}
    for label in GRAPH_MODES:
        path = root / "nsys" / "stats" / f"{label}_cuda_gpu_kern_sum.csv"
        source = read_csv(path)
        totals[label] = sum(float(row["Total Time (ns)"]) for row in source)
        for row in source:
            category = classify_kernel(row["Name"])
            total_ns = float(row["Total Time (ns)"])
            instances = int(row["Instances"])
            raw_rows.append(
                {
                    "profile": label,
                    "category": category,
                    "time_pct_reported": float(row["Time (%)"]),
                    "total_time_ms": total_ns / 1e6,
                    "instances": instances,
                    "avg_us": float(row["Avg (ns)"]) / 1e3,
                    "kernel": row["Name"],
                }
            )
            values = category_values[(label, category)]
            values["time_ns"] += total_ns
            values["instances"] += instances

    category_rows = []
    for (label, category), values in sorted(category_values.items()):
        category_rows.append(
            {
                "profile": label,
                "category": category,
                "total_time_ms": values["time_ns"] / 1e6,
                "kernel_time_pct": 100.0
                * values["time_ns"]
                / totals[label],
                "instances": int(values["instances"]),
            }
        )
    return raw_rows, category_rows


def memory_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in GRAPH_MODES:
        path = (
            root
            / "nsys"
            / "stats"
            / f"{label}_cuda_gpu_mem_time_sum.csv"
        )
        for row in read_csv(path):
            rows.append(
                {
                    "profile": label,
                    "operation": row["Operation"],
                    "total_time_ms": float(row["Total Time (ns)"]) / 1e6,
                    "count": int(row["Count"]),
                    "avg_us": float(row["Avg (ns)"]) / 1e3,
                }
            )
    return rows


def split_phase(total_ns: float, instances: int, decode_steps: int) -> Iterable[
    tuple[str, float, float]
]:
    if instances == decode_steps:
        yield "decode", total_ns, float(instances)
    elif instances == 1:
        yield "prefill", total_ns, 1.0
    elif instances == decode_steps + 1:
        yield "prefill", total_ns / instances, 1.0
        yield (
            "decode",
            total_ns * decode_steps / instances,
            float(decode_steps),
        )
    else:
        yield "mixed", total_ns, float(instances)


def layer_kernel_rows(root: Path, decode_steps: int) -> list[dict[str, Any]]:
    path = (
        root
        / "nsys"
        / "stats"
        / "static_ptq_tail_eager_layer_nvtx_kern_sum.csv"
    )
    grouped: dict[
        tuple[str, int, str, str, str], dict[str, float]
    ] = defaultdict(lambda: {"time_ns": 0.0, "kernel_instances": 0.0})
    for row in read_csv(path):
        match = LAYER_RE.match(row["NVTX Range"].lstrip(":"))
        if match is None:
            continue
        layer = int(match.group("layer"))
        path_name = match.group("path")
        component = match.group("component")
        total_ns = float(row["Total Time (ns)"])
        range_instances = int(row["NVTX Inst"])
        kernel_instances = int(row["Kern Inst"])
        for phase, phase_ns, phase_range_instances in split_phase(
            total_ns, range_instances, decode_steps
        ):
            fraction = phase_ns / total_ns if total_ns else 0.0
            key = (
                phase,
                layer,
                path_name,
                component,
                classify_kernel(row["Kernel Name"]),
            )
            values = grouped[key]
            values["time_ns"] += phase_ns
            values["kernel_instances"] += kernel_instances * fraction
            values["range_instances"] = phase_range_instances

    rows = []
    for (phase, layer, path_name, component, category), values in sorted(
        grouped.items()
    ):
        instances = values["range_instances"]
        rows.append(
            {
                "phase": phase,
                "layer": layer,
                "path": path_name,
                "component": component,
                "kernel_category": category,
                "total_kernel_time_ms": values["time_ns"] / 1e6,
                "range_instances": instances,
                "per_iteration_kernel_ms": (
                    values["time_ns"] / instances / 1e6
                    if instances
                    else None
                ),
                "kernel_instances": values["kernel_instances"],
            }
        )
    return rows


def stage_mean(
    event_summary: list[dict[str, Any]],
    profile: str,
    event_type: str,
    phase: str,
    stage: str,
) -> float:
    for row in event_summary:
        if (
            row["profile"] == profile
            and row["event_type"] == event_type
            and row["phase"] == phase
            and row["stage"] == stage
        ):
            return float(row["mean_ms"])
    return 0.0


def category_time(
    categories: list[dict[str, Any]], profile: str, category: str
) -> float:
    for row in categories:
        if row["profile"] == profile and row["category"] == category:
            return float(row["total_time_ms"])
    return 0.0


def critical_path_rows(
    profiles: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    definitions = (
        (
            "request_admission",
            "request",
            "wall",
            "",
            "Raw prompt admission including input processing",
        ),
        (
            "input_processing",
            "request",
            "cpu",
            "cpu.request.input_processing",
            "Tokenize and validate request",
        ),
        (
            "engine_preprocess",
            "request",
            "cpu",
            "cpu.request.engine_preprocess",
            "Build engine request state",
        ),
        (
            "scheduler",
            "unknown",
            "cpu",
            "cpu.scheduler.schedule",
            "Scheduler decision; phase is assigned after this call",
        ),
        (
            "runner_update_state",
            "decode",
            "cpu",
            "cpu.runner.update_states",
            "Update persistent request state",
        ),
        (
            "prepare_inputs_cpu",
            "decode",
            "cpu",
            "runtime.prepare_inputs",
            "CPU launch/preparation wall time",
        ),
        (
            "select_execution_shape",
            "decode",
            "cpu",
            "cpu.runner.select_execution_shape",
            "Select batch shape and graph",
        ),
        (
            "slot_mapping",
            "decode",
            "cpu",
            "runtime.slot_mappings",
            "Build KV-cache slot mapping",
        ),
        (
            "attention_metadata",
            "decode",
            "cpu",
            "runtime.prepare_attn_metadata",
            "Build attention metadata",
        ),
        (
            "model_input_preprocess",
            "decode",
            "cpu",
            "runtime.preprocess_model_inputs",
            "Final model input preprocessing",
        ),
        (
            "executor_submit",
            "decode",
            "cpu",
            "cpu.executor.submit_model",
            "Parent range containing CPU preparation and model launch",
        ),
        (
            "prepare_inputs_gpu",
            "decode",
            "gpu",
            "runtime.prepare_inputs",
            "GPU input preparation",
        ),
        (
            "model_forward_gpu",
            "decode",
            "gpu",
            "gpu.model_forward",
            "Transformer forward",
        ),
        (
            "logits_gpu",
            "decode",
            "gpu",
            "gpu.compute_logits",
            "LM-head vocabulary projection",
        ),
        (
            "sampler_gpu",
            "decode",
            "gpu",
            "gpu.sampler",
            "Greedy sampler",
        ),
        (
            "gpu_to_cpu_wait",
            "decode",
            "cpu",
            "cpu.output.wait_gpu_to_cpu",
            "Blocking wait that overlaps GPU execution",
        ),
        (
            "scheduler_update",
            "decode",
            "cpu",
            "cpu.scheduler.update_output",
            "Commit sampled output to scheduler state",
        ),
        (
            "detokenize_output",
            "decode",
            "cpu",
            "cpu.output.process_and_detokenize",
            "Detokenize and build delta output",
        ),
    )
    rows: list[dict[str, Any]] = []
    for label in GRAPH_MODES:
        for name, phase, domain, stage, note in definitions:
            if name == "request_admission":
                value = summary_value(
                    profiles[label], "baseline", "add_request_ms"
                )
            else:
                value = stage_mean(
                    events, label, domain, phase, stage
                )
            rows.append(
                {
                    "profile": label,
                    "stage": name,
                    "phase": phase,
                    "timing_domain": domain,
                    "mean_ms": value,
                    "note": note,
                }
            )
    return rows


def layer_component_per_decode(
    rows: list[dict[str, Any]],
    *,
    path: str,
    component: str,
    decode_steps: int,
) -> float:
    return sum(
        float(row["total_kernel_time_ms"])
        for row in rows
        if row["phase"] == "decode"
        and row["path"] == path
        and row["component"] == component
    ) / decode_steps


def render_summary(
    root: Path,
    profiles: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    layer_kernels: list[dict[str, Any]],
) -> str:
    fp = profiles["full_fp_cudagraph"]
    ptq = profiles["static_ptq_tail_cudagraph"]
    fp_tpot = summary_value(fp, "baseline", "strict_tpot_ms")
    ptq_tpot = summary_value(ptq, "baseline", "strict_tpot_ms")
    fp_ttft = summary_value(fp, "baseline", "ttft_ms")
    ptq_ttft = summary_value(ptq, "baseline", "ttft_ms")
    fp_e2e = summary_value(fp, "baseline", "e2e_ms")
    ptq_e2e = summary_value(ptq, "baseline", "e2e_ms")
    fp_tps = summary_value(fp, "baseline", "end_to_end_output_tps")
    ptq_tps = summary_value(ptq, "baseline", "end_to_end_output_tps")
    fp_eager = summary_value(
        profiles["full_fp_eager_layer"], "baseline", "strict_tpot_ms"
    )
    ptq_eager = summary_value(
        profiles["static_ptq_tail_eager_layer"],
        "baseline",
        "strict_tpot_ms",
    )

    fp_model = stage_mean(
        events, "full_fp_cudagraph", "gpu", "decode", "gpu.model_forward"
    )
    ptq_model = stage_mean(
        events,
        "static_ptq_tail_cudagraph",
        "gpu",
        "decode",
        "gpu.model_forward",
    )
    fp_logits = stage_mean(
        events, "full_fp_cudagraph", "gpu", "decode", "gpu.compute_logits"
    )
    ptq_logits = stage_mean(
        events,
        "static_ptq_tail_cudagraph",
        "gpu",
        "decode",
        "gpu.compute_logits",
    )
    fp_prefill_model = stage_mean(
        events, "full_fp_cudagraph", "gpu", "prefill", "gpu.model_forward"
    )
    ptq_prefill_model = stage_mean(
        events,
        "static_ptq_tail_cudagraph",
        "gpu",
        "prefill",
        "gpu.model_forward",
    )
    fp_prefill_logits = stage_mean(
        events,
        "full_fp_cudagraph",
        "gpu",
        "prefill",
        "gpu.compute_logits",
    )
    ptq_prefill_logits = stage_mean(
        events,
        "static_ptq_tail_cudagraph",
        "gpu",
        "prefill",
        "gpu.compute_logits",
    )
    fp_scheduler = stage_mean(
        events,
        "full_fp_cudagraph",
        "cpu",
        "unknown",
        "cpu.scheduler.schedule",
    )
    ptq_scheduler = stage_mean(
        events,
        "static_ptq_tail_cudagraph",
        "cpu",
        "unknown",
        "cpu.scheduler.schedule",
    )
    fp_prepare_cpu = stage_mean(
        events,
        "full_fp_cudagraph",
        "cpu",
        "decode",
        "runtime.prepare_inputs",
    )
    ptq_prepare_cpu = stage_mean(
        events,
        "static_ptq_tail_cudagraph",
        "cpu",
        "decode",
        "runtime.prepare_inputs",
    )
    fp_wait = stage_mean(
        events,
        "full_fp_cudagraph",
        "cpu",
        "decode",
        "cpu.output.wait_gpu_to_cpu",
    )
    ptq_wait = stage_mean(
        events,
        "static_ptq_tail_cudagraph",
        "cpu",
        "decode",
        "cpu.output.wait_gpu_to_cpu",
    )
    fp_detokenize = stage_mean(
        events,
        "full_fp_cudagraph",
        "cpu",
        "decode",
        "cpu.output.process_and_detokenize",
    )
    ptq_detokenize = stage_mean(
        events,
        "static_ptq_tail_cudagraph",
        "cpu",
        "decode",
        "cpu.output.process_and_detokenize",
    )
    ptq_int4 = category_time(
        categories, "static_ptq_tail_cudagraph", "int4_weight_only_gemm"
    )
    fp_bf16 = category_time(categories, "full_fp_cudagraph", "bf16_gemm")
    ptq_bf16 = category_time(
        categories, "static_ptq_tail_cudagraph", "bf16_gemm"
    )
    fp_kernel_total = sum(
        float(row["total_time_ms"])
        for row in categories
        if row["profile"] == "full_fp_cudagraph"
    )
    ptq_kernel_total = sum(
        float(row["total_time_ms"])
        for row in categories
        if row["profile"] == "static_ptq_tail_cudagraph"
    )
    ptq_tail = layer_component_per_decode(
        layer_kernels,
        path="ptq",
        component="total",
        decode_steps=fp["max_new_tokens"] - 1,
    )
    shared_fp = sum(
        float(row["total_kernel_time_ms"])
        for row in layer_kernels
        if row["phase"] == "decode"
        and row["path"] == "fp"
        and row["component"] == "total"
        and int(row["layer"]) <= 3
    ) / (fp["max_new_tokens"] - 1)
    ptq_attention = layer_component_per_decode(
        layer_kernels,
        path="ptq",
        component="attention",
        decode_steps=fp["max_new_tokens"] - 1,
    )
    ptq_mlp = layer_component_per_decode(
        layer_kernels,
        path="ptq",
        component="mlp",
        decode_steps=fp["max_new_tokens"] - 1,
    )
    fp_layer_overhead = (
        summary_value(
            profiles["full_fp_eager_layer"], "profiled", "e2e_ms"
        )
        / summary_value(
            profiles["full_fp_eager_layer"], "baseline", "e2e_ms"
        )
        - 1.0
    ) * 100.0
    ptq_layer_overhead = (
        summary_value(
            profiles["static_ptq_tail_eager_layer"],
            "profiled",
            "e2e_ms",
        )
        / summary_value(
            profiles["static_ptq_tail_eager_layer"],
            "baseline",
            "e2e_ms",
        )
        - 1.0
    ) * 100.0

    return f"""# 单请求端到端效率剖析

## 结论摘要

- 配置：Llama-3.1-8B-Instruct、RiverEdge `k=3`、85 input tokens、32 output tokens、batch/concurrency=1、BF16、FULL_DECODE_ONLY CUDA Graph、Inductor 关闭。
- FP：TTFT `{fp_ttft:.2f} ms`，稳态 TPOT `{fp_tpot:.2f} ms/token`，端到端 `{fp_e2e:.2f} ms`，输出吞吐 `{fp_tps:.2f} tok/s`。
- PTQ tail：TTFT `{ptq_ttft:.2f} ms`，稳态 TPOT `{ptq_tpot:.2f} ms/token`，端到端 `{ptq_e2e:.2f} ms`，输出吞吐 `{ptq_tps:.2f} tok/s`。
- PTQ 将 TPOT 降低 `{100.0 * (1.0 - ptq_tpot / fp_tpot):.1f}%`（`{fp_tpot / ptq_tpot:.2f}x`），端到端输出 TPS 提高 `{ptq_tps / fp_tps:.2f}x`。TPS 提升略小，因为约 115 ms 的全 FP prefill/TTFT 没有变化。
- 结论：当前单请求瓶颈不是调度或采样，而是 decode 每步读取并执行大规模线性层权重；PTQ 正在优化正确的主要路径。

## 关键路径

| Decode 阶段 | FP (ms/token) | PTQ tail (ms/token) | PTQ 占 TPOT |
|---|---:|---:|---:|
| Model forward | {fp_model:.3f} | {ptq_model:.3f} | {100.0 * ptq_model / ptq_tpot:.1f}% |
| LM-head / logits | {fp_logits:.3f} | {ptq_logits:.3f} | {100.0 * ptq_logits / ptq_tpot:.1f}% |
| 其余关键路径 | {fp_tpot - fp_model - fp_logits:.3f} | {ptq_tpot - ptq_model - ptq_logits:.3f} | {100.0 * (ptq_tpot - ptq_model - ptq_logits) / ptq_tpot:.1f}% |

CUDA Graph 下，FP model forward 从 `{fp_model:.2f}` 降到 PTQ 的 `{ptq_model:.2f} ms/token`。LM head 仍为约 `{ptq_logits:.2f} ms/token`，从 FP 中的 `{100.0 * fp_logits / fp_tpot:.1f}%` 上升到 PTQ 中的 `{100.0 * ptq_logits / ptq_tpot:.1f}%`，成为下一阶段明确的优化对象。

CPU 的 `output.wait_gpu_to_cpu` 数字表示阻塞等待 GPU，不是 CPU 在做同等时长的计算。调度、状态更新、detokenize 等 CPU 工作均远小于 1 ms/step；GPU 输入准备约 0.06 ms/step，sampler 约 0.08 ms/step。

| 细分阶段 | FP (ms) | PTQ tail (ms) | 解释 |
|---|---:|---:|---|
| Prefill model forward | {fp_prefill_model:.3f} | {ptq_prefill_model:.3f} | 85-token prefill，全 FP |
| Prefill logits | {fp_prefill_logits:.3f} | {ptq_prefill_logits:.3f} | 首 token LM head |
| Scheduler / step | {fp_scheduler:.3f} | {ptq_scheduler:.3f} | CPU 决策 |
| Prepare inputs / step | {fp_prepare_cpu:.3f} | {ptq_prepare_cpu:.3f} | CPU wall，包含若干子阶段 |
| GPU-to-CPU wait / step | {fp_wait:.3f} | {ptq_wait:.3f} | 与 GPU 执行重叠，不可再次相加 |
| Detokenize / token | {fp_detokenize:.3f} | {ptq_detokenize:.3f} | CPU 输出处理 |

`critical_path.csv` 继续列出 state update、shape selection、slot mapping、attention metadata、model launch、sampling 和 scheduler update。CPU parent range、子 range及 GPU range存在嵌套或异步重叠，不能直接全部求和；可相加的 GPU 主路径已在上表“关键路径”中单独给出。

## Kernel 归因

- FP Nsys：总 kernel 时间 `{fp_kernel_total:.1f} ms/请求`，其中 BF16 GEMM `{fp_bf16:.1f} ms`（`{100.0 * fp_bf16 / fp_kernel_total:.1f}%`）。
- PTQ Nsys：总 kernel 时间 `{ptq_kernel_total:.1f} ms/请求`；INT4 tinygemm `{ptq_int4:.1f} ms`，BF16 GEMM `{ptq_bf16:.1f} ms`，二者合计 `{100.0 * (ptq_int4 + ptq_bf16) / ptq_kernel_total:.1f}%`。
- PTQ eager NVTX 的 kernel 归因估算：每个 decode step 的 4-32 层 PTQ tail 约 `{ptq_tail:.2f} ms`，1-3 层 shared FP 约 `{shared_fp:.2f} ms`；PTQ tail 内 attention 约 `{ptq_attention:.2f} ms`，MLP 约 `{ptq_mlp:.2f} ms`。这些是 kernel 累计时间，不包含 CPU launch gap。
- PTQ 的 Device-to-Device 拷贝虽增加到 4683 次，但整请求仅约 5.8 ms；它不是当前 1.29 s 端到端时间的主因。

## CUDA Graph

- FP eager TPOT `{fp_eager:.2f} ms`，CUDA Graph `{fp_tpot:.2f} ms`，仅改善 `{100.0 * (1.0 - fp_tpot / fp_eager):.1f}%`。FP GEMM 很长，kernel launch 开销占比较低。
- PTQ eager TPOT `{ptq_eager:.2f} ms`，CUDA Graph `{ptq_tpot:.2f} ms`，改善 `{100.0 * (1.0 - ptq_tpot / ptq_eager):.1f}%`。量化后 kernel 更短且更碎，CUDA Graph 的价值明显增大。

## 优化优先级

1. **LM head / logits**：约 `{ptq_logits:.2f} ms/token`，占 PTQ TPOT `{100.0 * ptq_logits / ptq_tpot:.1f}%`。研究量化 LM head、低比特 vocab projection，或在保持语义的前提下优化 top-1 logits 路径。
2. **INT4 tail kernel**：验证 tinygemm 在 Orin `M=1` 下的内存带宽、tensor-core 利用率和反量化融合；定制 kernel 应优先融合 scale/zero-point、GEMM、bias/activation，减少中间写回。
3. **Shared FP 1-3 层**：约 `{shared_fp:.2f} ms/token` 的不可退出固定成本。评估减小 `k`、只保留必要 shared state，或对 shared 层采用更温和的量化。
4. **保持 full decode CUDA Graph**：PTQ 下 Graph 已贡献约 `{100.0 * (1.0 - ptq_tpot / ptq_eager):.1f}%` TPOT 改善。动态 route 应按有限的静态路径/桶 capture，避免逐 token graph break。
5. **Prefill 单独优化**：当前按设计全 FP，所以 TTFT 没有改善。若目标包含 TTFT，再研究 prefill 权重量化或 chunked prefill；不要与 decode TPOT 结论混合。

## 测量边界

主结果是 vLLM 进程内从 raw prompt 请求接收、tokenize、schedule、GPU 执行、sample、detokenize 到最终输出的端到端时间，刻意排除了 loopback HTTP/SSE，以免网络栈掩盖模型关键路径。Nsys 只用于 kernel 构成，不能把 profiler 下 wall time当作正常性能。

CUDA Event runtime 插桩对 CUDA Graph E2E 的扰动低于 0.1%。eager 层级 hook 对 FP 的扰动为 `{fp_layer_overhead:.1f}%`，对 PTQ 为 `{ptq_layer_overhead:.1f}%`；因此 PTQ 层级 JSON 的绝对 wall time不作性能结论，层级归因采用更轻的 Nsys NVTX kernel 统计。

完整数据见 `request_summary.csv`、`critical_path.csv`、`runtime_stage_summary.csv`、`token_timeline.csv`、`engine_steps.csv`、`layer_event_summary.csv`、`nsys_kernel_categories.csv`、`nsys_layer_kernel_summary.csv` 和 `nsys/` 下原始 trace。
"""


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    profiles = load_profiles(root)
    requests = request_rows(profiles)
    events = event_rows(profiles)
    token_rows = token_timeline_rows(profiles)
    steps = engine_step_rows(profiles)
    layers = layer_event_rows(profiles)
    kernels, categories = global_kernel_rows(root)
    memops = memory_rows(root)
    decode_steps = profiles["full_fp_cudagraph"]["max_new_tokens"] - 1
    layer_kernels = layer_kernel_rows(root, decode_steps)
    critical_path = critical_path_rows(profiles, events)

    write_csv(root / "request_summary.csv", requests)
    write_csv(root / "critical_path.csv", critical_path)
    write_csv(root / "runtime_stage_summary.csv", events)
    write_csv(root / "token_timeline.csv", token_rows)
    write_csv(root / "engine_steps.csv", steps)
    write_csv(root / "layer_event_summary.csv", layers)
    write_csv(root / "nsys_kernel_summary.csv", kernels)
    write_csv(root / "nsys_kernel_categories.csv", categories)
    write_csv(root / "nsys_memory_operations.csv", memops)
    write_csv(root / "nsys_layer_kernel_summary.csv", layer_kernels)

    summary = render_summary(
        root, profiles, events, categories, layer_kernels
    )
    (root / "summary.md").write_text(summary, encoding="utf-8")
    analysis = {
        "profiles": requests,
        "runtime_stages": events,
        "kernel_categories": categories,
        "memory_operations": memops,
        "layer_kernel_summary": layer_kernels,
    }
    (root / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
