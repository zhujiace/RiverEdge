#!/usr/bin/env python3
"""PyTorch reference benchmark for River dual-tail inference.

This script does not patch River/vLLM internals. It loads the existing
SidewayLlama checkpoint and manually executes:

1. full_fp: full FP backbone layers 0..31
2. fp_tail: FP prefix 0..k-1 plus FP tail k..31
3. ptq_tail: FP prefix 0..k-1 plus HQQ PTQ tail k..31
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


def add_lm_eval_to_path(repo_root: Path) -> None:
    lm_eval_dir = repo_root / "rivier" / "lm-evaluation"
    hqq_dir = lm_eval_dir / "HQQ-test-from-git"
    sys.path.insert(0, str(lm_eval_dir))
    sys.path.insert(0, str(hqq_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/models/Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16",
    )
    parser.add_argument("--repo-root", default="/workspace/vllm")
    parser.add_argument("--output", default="/workspace/vllm/RiverEdge/results/p4_dual_tail_benchmark.json")
    parser.add_argument("--csv-output", default="")
    parser.add_argument("--exit-layer", type=int, default=3, help="1-indexed prefix depth k.")
    parser.add_argument("--exit-layers", default="", help="Comma/space separated 1-indexed prefix depths.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seq-lens", default="", help="Comma/space separated sequence lengths.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", default="", help="Comma/space separated batch sizes.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--quant-backend", choices=("n", "d", "g", "t"), default="t")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def parse_int_list(value: str, fallback: int) -> List[int]:
    if not value:
        return [fallback]
    items = value.replace(",", " ").split()
    return [int(item) for item in items]


def load_model(args: argparse.Namespace):
    repo_root = Path(args.repo_root).resolve()
    add_lm_eval_to_path(repo_root)

    from transformers import AutoConfig, AutoModelForCausalLM

    importlib.import_module("lm_eval.transformers_extra.models")

    compute_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    config.quantize_exit_layers = True
    config.quantize_full_model = False
    config.quant_backend = args.quant_backend
    config.output_full_model = True
    config.early_exit_threshold = None
    config.count_distribution = False
    config.collect_kv = False
    config.compile = False
    config.output_hidden_states = False
    config.output_attentions = False

    if isinstance(config.exit_layer_indices, str):
        config.exit_layer_indices = [int(x) for x in config.exit_layer_indices.split()]
    if isinstance(config.output_exit_layers, str):
        config.output_exit_layers = [int(x) for x in config.output_exit_layers.split()]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=compute_dtype,
        device_map=None,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    model = model.to("cuda:0").to(dtype=compute_dtype).eval()

    if args.quant_backend != "n":
        from hqq.core.quantize import HQQBackend, HQQLinear
        from hqq.utils.patching import prepare_for_inference

        HQQLinear.set_backend(HQQBackend.PYTORCH)
        backend_map = {"d": None, "g": "gemlite", "t": "torchao_int4"}
        backend = backend_map[args.quant_backend]
        if backend:
            prepare_for_inference(model, backend=backend, verbose=False)
        else:
            prepare_for_inference(model, verbose=False)

    return model


def get_exit_flow(model, exit_layer: int) -> Tuple[Iterable[nn.Module], nn.Module]:
    cfg = model.config
    if cfg.exit_arch != "river":
        raise ValueError(f"Only river exit_arch is supported, got {cfg.exit_arch}")
    if not cfg.exit_decoder_layer:
        raise ValueError("This checkpoint does not enable decoder-layer exits.")
    if exit_layer not in cfg.exit_layer_indices:
        raise ValueError(f"exit_layer={exit_layer} is not in {cfg.exit_layer_indices}")
    if exit_layer >= cfg.num_hidden_layers:
        raise ValueError("exit_layer must be smaller than num_hidden_layers for a non-empty tail.")

    exit_index = cfg.exit_layer_indices.index(exit_layer)
    flow = model.model.exit_modules[0][exit_index + 1 :]
    norm = model.model.exit_modules[1][exit_index]
    real_layers = [module for module in flow if not isinstance(module, nn.Identity)]
    expected = cfg.num_hidden_layers - exit_layer
    if len(real_layers) != expected:
        raise RuntimeError(
            f"PTQ tail length mismatch: got {len(real_layers)} real layers, expected {expected}"
        )
    return flow, norm


def make_inputs(model, batch_size: int, seq_len: int, seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(seed)
    vocab = int(model.config.vocab_size)
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), device="cuda:0", generator=generator)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device="cuda:0")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def run_layers(
    layers: Iterable[nn.Module],
    hidden_states: torch.Tensor,
    causal_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings,
) -> torch.Tensor:
    for layer in layers:
        if isinstance(layer, nn.Identity):
            continue
        hidden_states = layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            collect_kv=False,
            update_cache=False,
        )[0]
    return hidden_states


@torch.inference_mode()
def forward_path(model, inputs: Dict[str, torch.Tensor], exit_layer: int, mode: str) -> torch.Tensor:
    body = model.model
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    hidden_states = body.embed_tokens(input_ids)
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = body._update_causal_mask(
        attention_mask,
        hidden_states,
        cache_position,
        None,
        False,
    )
    position_embeddings = body.rotary_emb(hidden_states, position_ids)

    if mode == "full_fp":
        hidden_states = run_layers(
            body.layers,
            hidden_states,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        hidden_states = body.norm(hidden_states)
    else:
        hidden_states = run_layers(
            body.layers[:exit_layer],
            hidden_states,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        if mode == "fp_tail":
            hidden_states = run_layers(
                body.layers[exit_layer:],
                hidden_states,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
            )
            hidden_states = body.norm(hidden_states)
        elif mode == "ptq_tail":
            flow, norm = get_exit_flow(model, exit_layer)
            hidden_states = run_layers(
                flow,
                hidden_states,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
            )
            hidden_states = norm(hidden_states)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return model.lm_head(hidden_states[:, -1:, :])


def time_mode(model, inputs, exit_layer: int, mode: str, warmup: int, iters: int) -> Dict[str, object]:
    for _ in range(warmup):
        _ = forward_path(model, inputs, exit_layer, mode)
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = forward_path(model, inputs, exit_layer, mode)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    tokens = inputs["input_ids"].numel()
    return {
        "times_s": times,
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "min_s": min(times),
        "prefill_tokens": tokens,
        "mean_prefill_tps": tokens / statistics.mean(times),
        "median_prefill_tps": tokens / statistics.median(times),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model = load_model(args)
    get_exit_flow(model, args.exit_layer)
    exit_layers = parse_int_list(args.exit_layers, args.exit_layer)
    seq_lens = parse_int_list(args.seq_lens, args.seq_len)
    modes = ["full_fp", "fp_tail", "ptq_tail"]
    batch_sizes = parse_int_list(args.batch_sizes, args.batch_size)
    results = {
        "model_path": args.model_path,
        "exit_layers": exit_layers,
        "seq_lens": seq_lens,
        "batch_sizes": batch_sizes,
        "dtype": args.dtype,
        "quant_backend": args.quant_backend,
        "attn_implementation": args.attn_implementation,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cases": [],
    }

    rows = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            inputs = make_inputs(model, batch_size, seq_len, args.seed)
            for exit_layer in exit_layers:
                get_exit_flow(model, exit_layer)
                case = {"batch_size": batch_size, "seq_len": seq_len, "exit_layer": exit_layer, "modes": {}}
                print(f"\ncase: batch_size={batch_size}, seq_len={seq_len}, exit_layer={exit_layer}")
                for mode in modes:
                    result = time_mode(model, inputs, exit_layer, mode, args.warmup, args.iters)
                    case["modes"][mode] = result
                    rows.append(
                        {
                            "seq_len": seq_len,
                            "batch_size": batch_size,
                            "exit_layer": exit_layer,
                            "mode": mode,
                            "median_ms": result["median_s"] * 1000,
                            "mean_ms": result["mean_s"] * 1000,
                            "median_prefill_tps": result["median_prefill_tps"],
                            "mean_prefill_tps": result["mean_prefill_tps"],
                        }
                    )
                    print(
                        f"  {mode}: median={result['median_s'] * 1000:.2f} ms, "
                        f"median_prefill_tps={result['median_prefill_tps']:.2f}"
                    )

                full = case["modes"]["full_fp"]["median_s"]
                fp_tail = case["modes"]["fp_tail"]["median_s"]
                ptq_tail = case["modes"]["ptq_tail"]["median_s"]
                case["speedup_vs_full_fp_median"] = {"fp_tail": full / fp_tail, "ptq_tail": full / ptq_tail}
                case["speedup_vs_fp_tail_median"] = {"ptq_tail": fp_tail / ptq_tail}
                rows.append(
                    {
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "exit_layer": exit_layer,
                        "mode": "ptq_tail_speedup_vs_fp_tail",
                        "median_ms": "",
                        "mean_ms": "",
                        "median_prefill_tps": case["speedup_vs_fp_tail_median"]["ptq_tail"],
                        "mean_prefill_tps": "",
                    }
                )
                print(
                    f"  speedup ptq_tail vs fp_tail: "
                    f"{case['speedup_vs_fp_tail_median']['ptq_tail']:.3f}x"
                )
                results["cases"].append(case)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")

    if args.csv_output:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "seq_len",
                    "batch_size",
                    "exit_layer",
                    "mode",
                    "median_ms",
                    "mean_ms",
                    "median_prefill_tps",
                    "mean_prefill_tps",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
