#!/usr/bin/env python3
"""Profile one RiverEdge request from input processing to final output.

The script uses vLLM's in-process engine so it can time scheduler, model-runner,
sampling, detokenization, and per-token delivery without modifying vLLM.
Production timings use CUDA Graph. Optional eager-only layer hooks provide a
diagnostic layer breakdown and must not be interpreted as production latency.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import statistics
import time
from collections.abc import Callable
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
    parser.add_argument(
        "--detail",
        choices=("runtime", "layer"),
        default="runtime",
        help="Layer detail is eager-only and adds substantial overhead.",
    )
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--prompt-tokens-target", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--baseline-requests", type=int, default=5)
    parser.add_argument("--profile-requests", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Bracket detailed requests with cudaProfilerStart/Stop for nsys.",
    )
    parser.add_argument(
        "--output",
        default=(
            "/workspace/vllm/RiverEdge/experiments/"
            "14_single_request_end_to_end_profile/profile.json"
        ),
    )
    args = parser.parse_args()
    if args.detail == "layer" and args.execution_mode != "eager":
        parser.error("--detail layer requires --execution-mode eager")
    for name in (
        "exit_layer",
        "prompt_tokens_target",
        "max_new_tokens",
        "warmup",
        "baseline_requests",
        "profile_requests",
        "max_model_len",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def make_prompt(target_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog near the river edge. "
    return (base * max(1, target_tokens // 12 + 1)).strip()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


class StageProfiler:
    """Runtime monkeypatch profiler with CPU timers and CUDA event pairs."""

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.origin_ns = time.perf_counter_ns()
        self.current_request = -1
        self.current_step = -1
        self.current_phase = "idle"
        self.cpu_events: list[dict[str, Any]] = []
        self.gpu_events: list[dict[str, Any]] = []
        self._gpu_pending: list[tuple[dict[str, Any], Any, Any]] = []
        self._restorers: list[Callable[[], None]] = []
        self._hook_handles: list[Any] = []
        self.warnings: list[str] = []

    def _context(self) -> dict[str, Any]:
        return {
            "request_index": self.current_request,
            "step_index": self.current_step,
            "phase": self.current_phase,
        }

    def _record_cpu(
        self, stage: str, started_ns: int, ended_ns: int, context: dict[str, Any]
    ) -> None:
        self.cpu_events.append(
            {
                **context,
                "stage": stage,
                "start_ms": (started_ns - self.origin_ns) / 1e6,
                "duration_ms": (ended_ns - started_ns) / 1e6,
            }
        )

    def _record_gpu(
        self, stage: str, start_event: Any, end_event: Any, context: dict[str, Any]
    ) -> None:
        row = {**context, "stage": stage}
        self._gpu_pending.append((row, start_event, end_event))

    def wrap(
        self,
        obj: Any,
        attr: str,
        stage: str,
        *,
        cpu: bool = True,
        gpu: bool = False,
        after: Callable[[Any], None] | None = None,
    ) -> None:
        if obj is None or not hasattr(obj, attr):
            self.warnings.append(f"missing stage target: {stage}")
            return
        original = getattr(obj, attr)
        profiler = self

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            context = profiler._context()
            started_ns = time.perf_counter_ns()
            start_event = end_event = None
            profiler.torch.cuda.nvtx.range_push(stage)
            if gpu:
                start_event = profiler.torch.cuda.Event(enable_timing=True)
                end_event = profiler.torch.cuda.Event(enable_timing=True)
                start_event.record()
            try:
                result = original(*args, **kwargs)
                if after is not None:
                    after(result)
                return result
            finally:
                if gpu and start_event is not None and end_event is not None:
                    end_event.record()
                    profiler._record_gpu(stage, start_event, end_event, context)
                ended_ns = time.perf_counter_ns()
                if cpu:
                    profiler._record_cpu(stage, started_ns, ended_ns, context)
                profiler.torch.cuda.nvtx.range_pop()

        try:
            setattr(obj, attr, wrapped)
        except (AttributeError, TypeError) as exc:
            self.warnings.append(f"cannot wrap {stage}: {exc}")
            return

        def restore() -> None:
            setattr(obj, attr, original)

        self._restorers.append(restore)

    def wrap_scheduler(self, scheduler: Any) -> None:
        def classify(output: Any) -> None:
            counts = list(getattr(output, "num_scheduled_tokens", {}).values())
            if not counts:
                self.current_phase = "drain"
            elif any(count > 1 for count in counts):
                self.current_phase = "prefill"
            else:
                self.current_phase = "decode"

        self.wrap(
            scheduler,
            "schedule",
            "cpu.scheduler.schedule",
            cpu=True,
            after=classify,
        )

    def _module_hook(self, module: Any, stage: str) -> None:
        stack: list[tuple[Any, int, dict[str, Any]]] = []

        def pre_hook(_module: Any, _inputs: Any) -> None:
            context = self._context()
            started_ns = time.perf_counter_ns()
            start_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
            stack.append((start_event, started_ns, context))
            self.torch.cuda.nvtx.range_push(stage)

        def post_hook(_module: Any, _inputs: Any, _output: Any) -> None:
            end_event = self.torch.cuda.Event(enable_timing=True)
            end_event.record()
            started_event, started_ns, context = stack.pop()
            self._record_gpu(stage, started_event, end_event, context)
            self._record_cpu(stage, started_ns, time.perf_counter_ns(), context)
            self.torch.cuda.nvtx.range_pop()

        self._hook_handles.append(module.register_forward_pre_hook(pre_hook))
        self._hook_handles.append(module.register_forward_hook(post_hook))

    def install_layer_hooks(self, model_runner: Any) -> None:
        outer_model = model_runner.model
        inner_model = getattr(outer_model, "model", None)
        fp_layers = getattr(inner_model, "layers", None)
        if fp_layers is None:
            self.warnings.append("cannot locate FP layer list")
            return

        for index, layer in enumerate(fp_layers, start=1):
            prefix = f"gpu.layer.{index:02d}.fp"
            self._module_hook(layer, f"{prefix}.total")
            for attr, suffix in (
                ("input_layernorm", "input_norm"),
                ("self_attn", "attention"),
                ("post_attention_layernorm", "post_attn_norm"),
                ("mlp", "mlp"),
            ):
                child = getattr(layer, attr, None)
                if child is not None:
                    self._module_hook(child, f"{prefix}.{suffix}")

        ptq_layers = getattr(inner_model, "ptq_layers", None)
        if ptq_layers is None:
            return
        for index, layer in enumerate(ptq_layers, start=1):
            if layer.__class__.__name__ == "Identity":
                continue
            prefix = f"gpu.layer.{index:02d}.ptq"
            self._module_hook(layer, f"{prefix}.total")
            for attr, suffix in (("self_attn", "attention"), ("mlp", "mlp")):
                child = getattr(layer, attr, None)
                if child is not None:
                    self._module_hook(child, f"{prefix}.{suffix}")

    def resolve_gpu_events(self) -> None:
        self.torch.cuda.synchronize()
        for row, start_event, end_event in self._gpu_pending:
            row["duration_ms"] = start_event.elapsed_time(end_event)
            self.gpu_events.append(row)
        self._gpu_pending.clear()

    def close(self) -> None:
        for handle in reversed(self._hook_handles):
            handle.remove()
        for restore in reversed(self._restorers):
            restore()


def locate_runtime(llm: Any) -> tuple[Any, Any, Any, Any]:
    llm_engine = llm.llm_engine
    engine_core_client = llm_engine.engine_core
    engine_core = getattr(engine_core_client, "engine_core", engine_core_client)
    executor = engine_core.model_executor
    worker = executor.driver_worker
    model_runner = worker.model_runner
    return llm_engine, engine_core, executor, model_runner


def install_runtime_profile(
    profiler: StageProfiler,
    llm_engine: Any,
    engine_core: Any,
    executor: Any,
    model_runner: Any,
    *,
    layer_detail: bool,
) -> None:
    profiler.wrap(
        llm_engine.input_processor,
        "process_inputs",
        "cpu.request.input_processing",
    )
    profiler.wrap(
        engine_core,
        "preprocess_add_request",
        "cpu.request.engine_preprocess",
    )
    profiler.wrap_scheduler(engine_core.scheduler)
    profiler.wrap(
        engine_core.scheduler,
        "get_grammar_bitmask",
        "cpu.scheduler.grammar_mask",
    )
    profiler.wrap(
        engine_core.scheduler,
        "update_from_output",
        "cpu.scheduler.update_output",
    )
    profiler.wrap(
        engine_core,
        "_process_aborts_queue",
        "cpu.engine.process_aborts",
    )
    profiler.wrap(
        llm_engine.output_processor,
        "process_outputs",
        "cpu.output.process_and_detokenize",
    )
    profiler.wrap(
        executor,
        "execute_model",
        "cpu.executor.submit_model",
    )
    profiler.wrap(
        executor,
        "sample_tokens",
        "cpu.executor.sample_and_sync",
    )
    try:
        from vllm.v1.worker.gpu.async_utils import AsyncOutput
        from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

        profiler.wrap(
            AsyncGPUModelRunnerOutput,
            "get_output",
            "cpu.output.wait_gpu_to_cpu",
        )
        profiler.wrap(
            AsyncOutput,
            "get_output",
            "cpu.output.wait_gpu_to_cpu_v2",
        )
    except ImportError as exc:
        profiler.warnings.append(f"cannot instrument async output wait: {exc}")

    for attr, stage in (
        ("update_pp_decode_requests", "cpu.runner.update_pp"),
        ("finish_requests", "cpu.runner.finish_requests"),
        ("free_states", "cpu.runner.free_states"),
        ("add_requests", "cpu.runner.add_requests"),
        ("update_requests", "cpu.runner.update_requests"),
        ("_update_states", "cpu.runner.update_states"),
        (
            "_determine_batch_execution_and_padding",
            "cpu.runner.select_execution_shape",
        ),
        ("_get_slot_mappings", "runtime.slot_mappings"),
        ("_build_attention_metadata", "runtime.prepare_attn_metadata"),
        ("_preprocess", "runtime.preprocess_model_inputs"),
    ):
        profiler.wrap(model_runner, attr, stage)

    block_tables = getattr(model_runner, "block_tables", None)
    profiler.wrap(
        block_tables,
        "apply_staged_writes",
        "runtime.block_table_writes",
        gpu=True,
    )
    profiler.wrap(
        model_runner,
        "prepare_inputs",
        "runtime.prepare_inputs",
        gpu=True,
    )
    profiler.wrap(
        model_runner,
        "prepare_attn",
        "runtime.prepare_attn_metadata",
        gpu=True,
    )
    profiler.wrap(
        model_runner,
        "_prepare_inputs",
        "runtime.prepare_inputs",
        gpu=True,
    )
    model_state = getattr(model_runner, "model_state", None)
    profiler.wrap(
        model_state,
        "prepare_attn",
        "runtime.model_attention_metadata",
        gpu=True,
    )

    if hasattr(model_runner, "_model_forward"):
        profiler.wrap(
            model_runner,
            "_model_forward",
            "gpu.model_forward",
            gpu=True,
        )
    else:
        cudagraph_manager = getattr(model_runner, "cudagraph_manager", None)
        if cudagraph_manager is not None:
            profiler.wrap(
                cudagraph_manager,
                "run_fullgraph",
                "gpu.model_forward",
                gpu=True,
            )
            profiler.wrap(
                cudagraph_manager,
                "run_pw_graph",
                "gpu.model_forward_piecewise",
                gpu=True,
            )
        else:
            profiler.wrap(
                model_runner.model,
                "forward",
                "gpu.model_forward",
                gpu=True,
            )

    profiler.wrap(
        model_runner.model,
        "compute_logits",
        "gpu.compute_logits",
        gpu=True,
    )
    if hasattr(model_runner, "_sample"):
        profiler.wrap(
            model_runner,
            "_sample",
            "gpu.sampler",
            gpu=True,
        )
    else:
        sampler = getattr(model_runner, "sampler", None)
        if sampler is not None:
            profiler.wrap(sampler, "forward", "gpu.sampler", gpu=True)
    profiler.wrap(
        model_runner,
        "postprocess_sampled",
        "gpu.sample_postprocess",
        gpu=True,
    )
    profiler.wrap(
        model_runner,
        "_update_states_after_model_execute",
        "gpu.sample_postprocess",
        gpu=True,
    )

    if layer_detail:
        profiler.install_layer_hooks(model_runner)


def make_sampling(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )


def run_request(
    *,
    llm_engine: Any,
    prompt: str,
    args: argparse.Namespace,
    request_index: int,
    label: str,
    profiler: StageProfiler | None,
) -> dict[str, Any]:
    if profiler is not None:
        profiler.current_request = request_index
        profiler.current_step = -1
        profiler.current_phase = "request"

    request_id = f"{label}-{request_index}"
    sampling = make_sampling(args)
    request_started_ns = time.perf_counter_ns()
    if profiler is not None:
        profiler.torch.cuda.nvtx.range_push(f"request.{label}.{request_index}")

    add_started_ns = time.perf_counter_ns()
    llm_engine.add_request(request_id, prompt, sampling)
    add_finished_ns = time.perf_counter_ns()

    step_rows: list[dict[str, Any]] = []
    token_arrival_ns: list[int] = []
    token_ids: list[int] = []
    text_parts: list[str] = []
    finished_ns: int | None = None
    step_index = 0

    while llm_engine.has_unfinished_requests():
        if profiler is not None:
            profiler.current_step = step_index
            profiler.current_phase = "unknown"
            profiler.torch.cuda.nvtx.range_push(f"engine_step.{step_index}")
        step_started_ns = time.perf_counter_ns()
        outputs = llm_engine.step()
        step_finished_ns = time.perf_counter_ns()
        if profiler is not None:
            profiler.torch.cuda.nvtx.range_pop()

        emitted = 0
        output_finished = False
        for output in outputs:
            if output.request_id != request_id:
                continue
            for completion in output.outputs:
                new_ids = list(completion.token_ids)
                emitted += len(new_ids)
                token_ids.extend(new_ids)
                token_arrival_ns.extend([step_finished_ns] * len(new_ids))
                if completion.text:
                    text_parts.append(completion.text)
            if output.finished:
                output_finished = True
                finished_ns = step_finished_ns

        phase = (
            profiler.current_phase
            if profiler is not None
            else ("prefill" if step_index == 0 else "decode")
        )
        step_rows.append(
            {
                "step_index": step_index,
                "phase": phase,
                "wall_ms": (step_finished_ns - step_started_ns) / 1e6,
                "emitted_tokens": emitted,
                "finished": output_finished,
            }
        )
        step_index += 1

    if finished_ns is None:
        finished_ns = time.perf_counter_ns()
    if profiler is not None:
        profiler.torch.cuda.nvtx.range_pop()

    first_token_ns = token_arrival_ns[0] if token_arrival_ns else None
    inter_token_ms = [
        (right - left) / 1e6
        for left, right in zip(token_arrival_ns, token_arrival_ns[1:])
    ]
    e2e_ms = (finished_ns - request_started_ns) / 1e6
    output_tokens = len(token_ids)
    result = {
        "request_index": request_index,
        "request_id": request_id,
        "add_request_ms": (add_finished_ns - add_started_ns) / 1e6,
        "e2e_ms": e2e_ms,
        "ttft_ms": (
            None
            if first_token_ns is None
            else (first_token_ns - request_started_ns) / 1e6
        ),
        "strict_tpot_ms": (
            statistics.mean(inter_token_ms) if inter_token_ms else None
        ),
        "inter_token_p50_ms": percentile(inter_token_ms, 0.50),
        "inter_token_p90_ms": percentile(inter_token_ms, 0.90),
        "inter_token_p99_ms": percentile(inter_token_ms, 0.99),
        "finalization_after_last_token_ms": (
            None
            if not token_arrival_ns
            else (finished_ns - token_arrival_ns[-1]) / 1e6
        ),
        "output_tokens": output_tokens,
        "end_to_end_output_tps": output_tokens / (e2e_ms / 1000.0),
        "token_ids": token_ids,
        "text": "".join(text_parts),
        "step_rows": step_rows,
        "token_arrival_ms": [
            (timestamp - request_started_ns) / 1e6
            for timestamp in token_arrival_ns
        ],
    }
    return result


def summarize_requests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(name: str) -> list[float]:
        return [
            float(row[name])
            for row in rows
            if row.get(name) is not None
        ]

    summary: dict[str, Any] = {"requests": len(rows)}
    for name in (
        "add_request_ms",
        "e2e_ms",
        "ttft_ms",
        "strict_tpot_ms",
        "inter_token_p99_ms",
        "end_to_end_output_tps",
    ):
        values = collect(name)
        summary[name] = {
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def make_compilation_config(execution_mode: str) -> tuple[bool, Any]:
    if execution_mode == "eager":
        return True, None
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[1],
        cudagraph_num_of_warmups=1,
    )
    return False, config


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    args = parse_args()
    initialization_started = time.perf_counter()
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
    initialization_s = time.perf_counter() - initialization_started
    llm_engine, engine_core, executor, model_runner = locate_runtime(llm)

    prompt = make_prompt(args.prompt_tokens_target)
    prompt_tokens = len(llm.get_tokenizer().encode(prompt))
    warmup_sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        min_tokens=args.max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    for _ in range(args.warmup):
        llm.generate([prompt], warmup_sampling, use_tqdm=False)
    torch.cuda.synchronize()

    baseline_rows = [
        run_request(
            llm_engine=llm_engine,
            prompt=prompt,
            args=args,
            request_index=index,
            label="baseline",
            profiler=None,
        )
        for index in range(args.baseline_requests)
    ]
    torch.cuda.synchronize()

    profiler = StageProfiler(torch)
    install_runtime_profile(
        profiler,
        llm_engine,
        engine_core,
        executor,
        model_runner,
        layer_detail=args.detail == "layer",
    )

    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()
    profile_rows = [
        run_request(
            llm_engine=llm_engine,
            prompt=prompt,
            args=args,
            request_index=index,
            label="profile",
            profiler=profiler,
        )
        for index in range(args.profile_requests)
    ]
    profiler.resolve_gpu_events()
    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()
    profiler.close()

    result = {
        "schema_version": 1,
        "model": args.model,
        "mode": args.mode,
        "execution_mode": args.execution_mode,
        "detail": args.detail,
        "exit_layer": args.exit_layer,
        "dtype": args.dtype,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "initialization_s": initialization_s,
        "cuda_profiler_api": args.cuda_profiler_api,
        "baseline": {
            "summary": summarize_requests(baseline_rows),
            "requests": baseline_rows,
        },
        "profiled": {
            "summary": summarize_requests(profile_rows),
            "requests": profile_rows,
            "cpu_events": profiler.cpu_events,
            "gpu_events": profiler.gpu_events,
            "warnings": profiler.warnings,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "mode": args.mode,
                "execution_mode": args.execution_mode,
                "detail": args.detail,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "baseline_summary": result["baseline"]["summary"],
                "profiled_summary": result["profiled"]["summary"],
                "cpu_event_count": len(profiler.cpu_events),
                "gpu_event_count": len(profiler.gpu_events),
                "warnings": profiler.warnings,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
