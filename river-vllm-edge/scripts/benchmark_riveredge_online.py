#!/usr/bin/env python3
"""Replay timed arrivals through a real in-process vLLM engine.

Reports delivery timestamps (not CUDA kernel times), strict per-request TPOT,
SLO goodput, actual scheduler routes, and optional board-rail energy. Trace
rows accept arrival_s, prompt, allow_ptq, max_tokens, answer, and task.
This is not HTTP/SSE and synthetic eligibility is not River confidence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def score_answer(text, row):
    """Use an explicit final-answer pattern when reasoning precedes the answer."""
    if row.get("answer_regex"):
        matches = list(re.finditer(row["answer_regex"], text))
        if not matches:
            return None, False
        predicted = matches[-1].group(1)
    else:
        predicted = text
    normalize = lambda value: re.sub(r"\s+", " ", str(value).strip().casefold())
    return predicted, normalize(predicted) == normalize(row["answer"])


def summarize(requests, elapsed_s, ttft_slo, tpot_slo):
    for row in requests:
        stamps = row["token_times_s"]
        row["ttft_s"] = stamps[0] - row["arrival_s"] if stamps else None
        row["tpot_s"] = (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else None
        row["e2e_s"] = row["finished_s"] - row["arrival_s"]
        row["slo_pass"] = (row["ttft_s"] is not None and row["ttft_s"] <= ttft_slo
                           and (row["tpot_s"] is None or row["tpot_s"] <= tpot_slo))
    summary = {"requests": len(requests), "elapsed_s": elapsed_s,
               "output_tokens": sum(len(r["token_ids"]) for r in requests)}
    for name in ("ttft_s", "tpot_s", "e2e_s", "admission_lag_s"):
        values = [r[name] for r in requests if r[name] is not None]
        summary[name] = {f"p{p}": percentile(values, p / 100) for p in (50, 90, 95, 99)}
    summary["output_tps"] = summary["output_tokens"] / elapsed_s
    summary["slo_requests_per_s"] = sum(r["slo_pass"] for r in requests) / elapsed_s
    summary["slo_tokens_per_s"] = sum(len(r["token_ids"]) for r in requests if r["slo_pass"]) / elapsed_s
    labeled = [r for r in requests if r.get("correct") is not None]
    summary["quality"] = {
        task: {"count": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows)}
        for task in sorted({r.get("task", "unspecified") for r in labeled})
        if (rows := [r for r in labeled if r.get("task", "unspecified") == task])
    }
    summary["quality_slo_requests_per_s"] = (
        sum(r["correct"] and r["slo_pass"] for r in labeled) / elapsed_s
        if labeled else None)
    return summary


class PowerSampler:
    def __init__(self, label):
        self.samples = []
        self.stop_event = threading.Event()
        self.sensor = None
        for path in Path("/sys/bus/i2c/drivers/ina3221").glob("*/hwmon/hwmon*/in*_label"):
            if path.read_text().strip() == label:
                index = path.name[2:].split("_")[0]
                voltage = path.with_name(f"in{index}_input")
                current = path.with_name(f"curr{index}_input")
                if voltage.exists() and current.exists():
                    self.sensor = voltage, current
                    break
        self.thread = None
        self.error = None

    def sample_once(self):
        voltage, current = self.sensor
        watts = float(voltage.read_text()) * float(current.read_text()) / 1e6
        self.samples.append((time.perf_counter(), watts))

    def start(self):
        if self.sensor is None:
            return
        try:
            self.sample_once()
        except (OSError, ValueError) as error:
            self.error = str(error)
            return
        def sample():
            try:
                while not self.stop_event.is_set():
                    self.sample_once()
                    self.stop_event.wait(0.1)
            except (OSError, ValueError) as error:
                self.error = str(error)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def finish(self, start, end, tokens):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        if self.sensor is not None and not self.error:
            try:
                self.sample_once()
            except (OSError, ValueError) as error:
                self.error = str(error)
        joules = None
        if len(self.samples) >= 2 and not self.error:
            joules = 0.0
            for a, b in zip(self.samples, self.samples[1:]):
                left, right = max(start, a[0]), min(end, b[0])
                if right <= left or b[0] <= a[0]:
                    continue
                slope = (b[1] - a[1]) / (b[0] - a[0])
                left_w = a[1] + slope * (left - a[0])
                right_w = a[1] + slope * (right - a[0])
                joules += (right - left) * (left_w + right_w) / 2
        return {"sensor": [str(p) for p in self.sensor] if self.sensor else None,
                "error": self.error, "samples": self.samples,
                "sampled_joules": joules,
                "sample_coverage_fraction": ((min(end, self.samples[-1][0]) - max(start, self.samples[0][0])) / (end - start)
                                             if len(self.samples) >= 2 else 0),
                "sampled_joules_per_output_token": joules / tokens if joules is not None and tokens else None,
                "scope": "whole rail, shared-device load included, no idle subtraction"}


def workload(args):
    if args.trace:
        rows = [json.loads(line) for line in Path(args.trace).read_text().splitlines() if line.strip()]
    else:
        rng = random.Random(args.seed)
        arrival = 0.0
        rows = []
        for index in range(args.requests):
            if index and args.arrivals == "poisson":
                arrival += rng.expovariate(args.rate)
            elif args.arrivals == "bursty":
                arrival = (index // args.batch_size) / args.rate
            rows.append({"arrival_s": arrival, "prompt": "Explain why the sky is blue in simple terms.",
                         "allow_ptq": index % 2 == 0,
                         "max_tokens": args.max_tokens})
    if not rows:
        raise ValueError("Empty workload")
    for index, row in enumerate(rows):
        row.setdefault("arrival_s", 0.0)
        row.setdefault("allow_ptq", False)
        row.setdefault("max_tokens", args.max_tokens)
        if row["arrival_s"] < 0 or row["max_tokens"] < 1 or not isinstance(row.get("prompt"), str):
            raise ValueError(f"Invalid trace row {index}")
        if row.get("answer_regex") and re.compile(row["answer_regex"]).groups != 1:
            raise ValueError("answer_regex must contain exactly one capture group")
        row["request_id"] = f"request-{index}"
    return sorted(rows, key=lambda row: row["arrival_s"])


def replay(llm, rows, args, runtime):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    engine = llm.llm_engine
    pending, results = list(rows), {}
    power = PowerSampler(args.power_label)
    power.start()
    started = time.perf_counter()
    first_step = len(runtime.steps)
    while pending or engine.has_unfinished_requests():
        now = time.perf_counter() - started
        while pending and pending[0]["arrival_s"] <= now:
            source = pending.pop(0)
            row = dict(source, token_times_s=[], token_ids=[], text="",
                       admission_lag_s=time.perf_counter() - started - source["arrival_s"])
            results[row["request_id"]] = row
            params = SamplingParams(temperature=0, max_tokens=row["max_tokens"],
                                    ignore_eos=args.force_length,
                                    output_kind=RequestOutputKind.DELTA,
                                    extra_args={"riveredge_allow_ptq": row["allow_ptq"],
                                                "riveredge_ttft_slo": args.ttft_slo,
                                                "riveredge_tpot_slo": args.tpot_slo})
            engine.add_request(row["request_id"], row["prompt"], params)
        if not engine.has_unfinished_requests():
            time.sleep(min(0.01, max(0, pending[0]["arrival_s"] - now)))
            continue
        outputs = engine.step()
        stamp = time.perf_counter() - started
        for output in outputs:
            row = results[output.request_id]
            for completion in output.outputs:
                if len(completion.token_ids) > 1:
                    raise RuntimeError("Multiple tokens per delivery: strict timing unavailable")
                row["token_ids"].extend(completion.token_ids)
                row["token_times_s"].extend([stamp] * len(completion.token_ids))
                row["text"] += completion.text
            if output.finished:
                row["finished_s"] = stamp
                if "answer" in row:
                    row["scored_answer"], row["correct"] = score_answer(row["text"], row)
    ended = time.perf_counter()
    requests = list(results.values())
    summary = summarize(requests, ended - started, args.ttft_slo, args.tpot_slo)
    energy = power.finish(started, ended, summary["output_tokens"])
    steps = runtime.steps[first_step:]
    return {"summary": summary, "expected_request_ids": [row['request_id'] for row in rows],
            "requests": requests, "steps": steps, "energy": energy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--latency-probe", action="store_true",
                        help="Also measure pure PTQ, B=1, 64 output tokens, five repeats per head")
    parser.add_argument("--arrival-patterns", default="",
                        help="Run comma-separated synthetic patterns in one model process")
    parser.add_argument("--policies", default="fp,ptq,mixed,batch")
    parser.add_argument("--schedulers", default="native,grouped")
    parser.add_argument("--exit-layers", default="3")
    parser.add_argument("--head-variants", default="fp", help="fp,int4: opt-in quality-changing LM-head ablation")
    parser.add_argument("--arrivals", choices=("simultaneous", "poisson", "bursty"), default="bursty")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rate", type=float, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    parser.add_argument("--kv-cache-bytes", type=int, default=268435456,
                        help="Explicit KV budget prevents shared-device profiling from overallocating")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--low-batch", type=int, default=8)
    parser.add_argument("--high-batch", type=int, default=16)
    parser.add_argument("--ttft-slo", type=float, default=1)
    parser.add_argument("--tpot-slo", type=float, default=0.1)
    parser.add_argument("--force-length", action="store_true")
    parser.add_argument("--power-label", default="VDD_GPU_SOC")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new path to preserve artifacts")
    if min(args.requests, args.batch_size, args.max_tokens, args.rate, args.repeats,
           args.ttft_slo, args.tpot_slo) <= 0 or args.warmup < 1:
        parser.error("Counts, rates and SLOs must be positive; warmup >= 1")
    policies, schedulers = args.policies.split(","), args.schedulers.split(",")
    if set(schedulers) - {"native", "grouped"}:
        parser.error("Unknown scheduler")
    heads = args.head_variants.split(",")
    if set(heads) - {"fp", "int4"}:
        parser.error("Unknown head variant")
    from river_vllm_ext.online_runtime import OnlineRuntime, RoutePolicy
    for policy in policies:
        RoutePolicy(policy, args.low_batch, args.high_batch)
    rows = workload(args)
    workloads = {"trace" if args.trace else args.arrivals: rows}
    if args.arrival_patterns:
        if args.trace:
            parser.error("--arrival-patterns cannot be combined with --trace")
        workloads = {}
        original_pattern = args.arrivals
        for pattern in args.arrival_patterns.split(","):
            if pattern not in {"simultaneous", "poisson", "bursty"}:
                parser.error("Unknown arrival pattern")
            args.arrivals = pattern
            workloads[pattern] = workload(args)
        args.arrivals = original_pattern
    from vllm import LLM
    import torch
    import torchao
    import vllm
    extension = Path(__file__).resolve().parents[1] / "river_vllm_ext"
    sources = [Path(__file__), extension / "online_runtime.py",
               extension / "models/online_llama.py", extension / "head_ablation.py"]
    environment = {"vllm": vllm.__version__, "torch": torch.__version__,
                   "torchao": torchao.__version__, "device": torch.cuda.get_device_name(),
                   "argv": sys.argv, "started_unix_s": time.time(),
                   "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              async_scheduling=False, max_model_len=args.max_model_len,
              max_num_seqs=args.batch_size,
              max_num_batched_tokens=args.max_model_len * args.batch_size,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              gpu_memory_utilization=args.memory_fraction, disable_log_stats=True,
              kv_cache_memory_bytes=args.kv_cache_bytes,
              hf_overrides={"architectures": ["RiverEdgeOnlineForCausalLM"],
                            "river_edge_mode": "full_fp", "river_edge_exit_layer": 3})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    if args.chat_template:
        tokenizer = llm.get_tokenizer()
        for workload_rows in workloads.values():
            for row in workload_rows:
                row["raw_prompt"] = row["prompt"]
                row["prompt"] = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False, add_generation_prompt=True)
    core_client = llm.llm_engine.engine_core
    core = getattr(core_client, "engine_core", core_client)
    model = core.model_executor.driver_worker.model_runner.model
    original_head = model.lm_head.quant_method
    head_candidate = None
    if "int4" in heads:
        from river_vllm_ext.head_ablation import Int4HeadCandidate
        head_candidate = Int4HeadCandidate(model)
    configs = [(int(k), policy, scheduler, repeat, head, pattern)
               for k in args.exit_layers.split(",") for policy in policies
               for scheduler in schedulers for repeat in range(args.repeats)
               for head in heads for pattern in workloads]
    if args.latency_probe:
        workloads["latency_probe"] = [{"request_id": "latency-probe", "arrival_s": 0,
                                        "prompt": "Explain why the sky is blue in simple terms.",
                                        "allow_ptq": True, "max_tokens": 64}]
        configs.extend((3, "ptq", "native", repeat, head, "latency_probe")
                       for head in heads for repeat in range(5))
    random.Random(args.seed).shuffle(configs)
    for k, policy, scheduler, repeat, head, pattern in configs:
        rows = workloads[pattern]
        force_length = args.force_length
        if pattern == "latency_probe":
            args.force_length = True
        runtime = OnlineRuntime(llm, RoutePolicy(policy, args.low_batch, args.high_batch),
                                grouped=scheduler == "grouped")
        try:
            if not 1 <= k < runtime.model.config.num_hidden_layers:
                raise ValueError("Invalid exit layer")
            runtime.model.river_edge_exit_layer = k
            model.lm_head.quant_method = head_candidate if head == "int4" else original_head
            warm_rows = [dict(row, arrival_s=0, max_tokens=64 if pattern == "latency_probe" else 4)
                         for row in rows[:args.batch_size]]
            for _ in range(args.warmup):
                replay(llm, warm_rows, args, runtime)
            case = replay(llm, rows, args, runtime)
            case.update(exit_layer=k, policy=policy, scheduler=scheduler, repeat=repeat, head=head,
                        workload=pattern, force_length=args.force_length)
            cases.append(case)
            args.force_length = force_length
            output.write_text(json.dumps({"schema_version": 2, "environment": environment,
                                          "config": vars(args), "cases": cases,
                                          "timing_scope": "in-process delivery; excludes HTTP/SSE"}, indent=2))
            print(json.dumps({"case": [k, policy, scheduler, repeat, head, pattern], **case["summary"]}), flush=True)
        finally:
            args.force_length = force_length
            runtime.close()
            model.lm_head.quant_method = original_head
    llm.llm_engine.engine_core.shutdown()
    import torch.distributed
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
