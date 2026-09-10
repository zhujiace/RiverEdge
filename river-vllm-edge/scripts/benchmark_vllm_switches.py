#!/usr/bin/env python3
"""Canonical switch-based benchmark. No enabled switches means stock vLLM."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from river_vllm_ext.optimization_switches import OptimizationSwitches, create_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in OptimizationSwitches.__dataclass_fields__:
        parser.add_argument('--' + name.replace('_', '-'), action=argparse.BooleanOptionalAction,
                            default=False)
    parser.add_argument('--all-off', action='store_true', help='Assert that every RiverEdge switch is off')
    parser.add_argument('--native-model', default='/models/Llama-3.1-8B-Instruct-native-view')
    parser.add_argument('--riveredge-model', default='/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq')
    parser.add_argument('--exit-layer', type=int, default=3)
    parser.add_argument('--low-batch', type=int, default=8)
    parser.add_argument('--high-batch', type=int, default=16)
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--max-model-len', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--prompt', default='Explain why the sky is blue in simple terms.')
    parser.add_argument('--enforce-eager', action=argparse.BooleanOptionalAction, default=None,
                        help='Ordinary vLLM option; omitted means upstream/default path behavior')
    parser.add_argument('--kv-cache-bytes', type=int, default=134217728)
    parser.add_argument('--memory-fraction', type=float, default=0.36)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    switches = OptimizationSwitches(**{key: getattr(args, key) for key in OptimizationSwitches.__dataclass_fields__})
    try:
        switches.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.all_off and not switches.all_off:
        parser.error('--all-off cannot be combined with enabled RiverEdge switches')
    if min(args.requests, args.batch_size, args.max_tokens, args.max_model_len,
           args.warmup, args.kv_cache_bytes) < 1:
        parser.error('Counts and memory budget must be positive')
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a new path')
    # Same harness setting for native and RiverEdge; not an engine mutation.
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    from vllm import SamplingParams
    import torch
    kwargs = dict(dtype='bfloat16', max_model_len=args.max_model_len,
                  max_num_seqs=args.batch_size, max_num_batched_tokens=args.max_model_len * args.batch_size,
                  kv_cache_memory_bytes=args.kv_cache_bytes, gpu_memory_utilization=args.memory_fraction,
                  disable_log_stats=True)
    if args.enforce_eager is not None:
        kwargs['enforce_eager'] = args.enforce_eager
    engine = create_engine(switches, native_model=args.native_model, riveredge_model=args.riveredge_model,
                           exit_layer=args.exit_layer, low_batch=args.low_batch, high_batch=args.high_batch,
                           capture_sizes=tuple(range(1, args.batch_size + 1)), **kwargs)
    try:
        # Alternating eligibility exercises both tails without claiming confidence.
        prompts = [args.prompt] * args.requests
        sampling = [SamplingParams(temperature=0, max_tokens=args.max_tokens, min_tokens=args.max_tokens,
                                   ignore_eos=True, extra_args={'riveredge_allow_ptq': i % 2 == 0}
                                   if switches.online_routing else None) for i in range(args.requests)]
        for _ in range(args.warmup):
            engine.llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        first_step = len(engine.runtime.steps) if engine.runtime else 0
        started = time.perf_counter()
        results = engine.llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        core = engine.llm.llm_engine.engine_core
        core = getattr(core, 'engine_core', core)
        model = core.model_executor.driver_worker.model_runner.model
        # Eager mode gives direct type evidence; compiled/Graph wrappers may wrap it.
        model_class = f'{type(model).__module__}.{type(model).__name__}'
        rows = [dict(token_ids=list(row.outputs[0].token_ids), text=row.outputs[0].text,
                     finished=row.finished) for row in results]
        valid = len(rows) == args.requests and all(r['finished'] and len(r['token_ids']) == args.max_tokens for r in rows)
        source = Path(__file__).resolve().parents[1] / 'river_vllm_ext/optimization_switches.py'
        payload = dict(schema_version=1, switches=asdict(switches), native_path=not switches.ptq_tail,
                       model_class=model_class, selected_architecture=switches.architecture,
                       runtime_hooks_installed=engine.runtime is not None, args=vars(args),
                       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       request_contract_pass=valid, elapsed_s=elapsed, requests=rows,
                       output_tps=sum(len(row['token_ids']) for row in rows) / elapsed,
                       steps=engine.runtime.steps[first_step:] if engine.runtime else [],
                       note='Functional closed-workload smoke; not confidence routing or a quality benchmark')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        print(json.dumps({key: payload[key] for key in ('model_class', 'runtime_hooks_installed', 'request_contract_pass', 'output_tps')}))
        if not valid:
            raise RuntimeError('Request completion/token-count check failed')
    finally:
        engine.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
