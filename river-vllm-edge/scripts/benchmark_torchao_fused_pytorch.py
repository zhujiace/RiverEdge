#!/usr/bin/env python3
"""PyTorch-only fused-layer speed probe for serialized torchao Llama weights.

This script does not use vLLM. It loads the unified RiverEdge checkpoint,
restores the HQQ sidecar tensor subclasses with torchao metadata, and
benchmarks a Llama-like per-token fused linear path. Attention score/KV work is
not modeled here; the goal is to isolate whether serialized INT4 fused weights
can accelerate the decode-dominant linear stack under PyTorch/torchao.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torchao.prototype.safetensors.safetensors_support import (
    unflatten_tensor_state_dict,
)


LINEAR_SUFFIXES = (
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq",
    )
    parser.add_argument("--exit-layer", type=int, default=3, help="1-indexed shared FP depth k")
    parser.add_argument("--modes", default="full_fp,shared_fp_plus_fp_tail,shared_fp_plus_ptq_tail,all_ptq_tail")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--output-json",
        default="/workspace/vllm/RiverEdge/experiments/09_torchao_serialized_riveredge/p4_pytorch_fused_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        default="/workspace/vllm/RiverEdge/experiments/09_torchao_serialized_riveredge/p4_pytorch_fused_summary.csv",
    )
    return parser.parse_args()


class IndexedSafetensorsReader:
    def __init__(self, model_dir: Path):
        with (model_dir / "model.safetensors.index.json").open("r", encoding="utf-8") as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self.model_dir = model_dir

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).contiguous()


def load_quant_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    with (checkpoint_dir / "riveredge_export_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        ptq_filename = json.load(handle)["ptq_file"]
    with safe_open(checkpoint_dir / ptq_filename, framework="pt", device="cpu") as handle:
        flat = {name: handle.get_tensor(name) for name in handle.keys()}
        metadata = handle.metadata()
    state, leftover = unflatten_tensor_state_dict(flat, metadata)
    if leftover:
        raise RuntimeError(f"Unflatten left incomplete tensor data: {sorted(leftover)[:5]}")
    return state


def load_config(model_dir: Path) -> dict:
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)
    return out * weight


def load_fp_layer(
    reader: IndexedSafetensorsReader,
    layer_idx: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}"
    qkv = torch.cat(
        [
            reader.get(f"{prefix}.self_attn.q_proj.weight"),
            reader.get(f"{prefix}.self_attn.k_proj.weight"),
            reader.get(f"{prefix}.self_attn.v_proj.weight"),
        ],
        dim=0,
    ).contiguous()
    gate_up = torch.cat(
        [
            reader.get(f"{prefix}.mlp.gate_proj.weight"),
            reader.get(f"{prefix}.mlp.up_proj.weight"),
        ],
        dim=0,
    ).contiguous()
    return {
        "input_layernorm": reader.get(f"{prefix}.input_layernorm.weight").to(device=device, dtype=dtype),
        "post_attention_layernorm": reader.get(f"{prefix}.post_attention_layernorm.weight").to(device=device, dtype=dtype),
        "qkv": qkv.to(device=device, dtype=dtype),
        "o": reader.get(f"{prefix}.self_attn.o_proj.weight").to(device=device, dtype=dtype),
        "gate_up": gate_up.to(device=device, dtype=dtype),
        "down": reader.get(f"{prefix}.mlp.down_proj.weight").to(device=device, dtype=dtype),
    }


def load_ptq_layer(
    quant_state: dict[str, torch.Tensor],
    reader: IndexedSafetensorsReader,
    layer_idx: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    fp_prefix = f"model.layers.{layer_idx}"
    ptq_prefix = f"model.ptq_layers.{layer_idx}"
    return {
        "input_layernorm": reader.get(f"{fp_prefix}.input_layernorm.weight").to(device=device, dtype=dtype),
        "post_attention_layernorm": reader.get(f"{fp_prefix}.post_attention_layernorm.weight").to(device=device, dtype=dtype),
        "qkv": quant_state[f"{ptq_prefix}.self_attn.qkv_proj.weight"].to(device),
        "o": quant_state[f"{ptq_prefix}.self_attn.o_proj.weight"].to(device),
        "gate_up": quant_state[f"{ptq_prefix}.mlp.gate_up_proj.weight"].to(device),
        "down": quant_state[f"{ptq_prefix}.mlp.down_proj.weight"].to(device),
    }


def uses_ptq(mode: str, layer_idx: int, exit_layer: int) -> bool:
    if mode in {"full_fp", "shared_fp_plus_fp_tail"}:
        return False
    if mode == "shared_fp_plus_ptq_tail":
        return layer_idx >= exit_layer
    if mode == "all_ptq_tail":
        return layer_idx >= 1
    raise ValueError(f"Unsupported mode: {mode}")


def build_layers(
    mode: str,
    reader: IndexedSafetensorsReader,
    quant_state: dict[str, torch.Tensor] | None,
    num_layers: int,
    exit_layer: int,
    device: str,
    dtype: torch.dtype,
) -> list[dict[str, torch.Tensor]]:
    layers = []
    for layer_idx in range(num_layers):
        if uses_ptq(mode, layer_idx, exit_layer):
            if quant_state is None:
                raise RuntimeError(f"Mode {mode} requires quant_state.")
            layers.append(load_ptq_layer(quant_state, reader, layer_idx, device, dtype))
        else:
            layers.append(load_fp_layer(reader, layer_idx, device, dtype))
    return layers


def forward_proxy(
    x: torch.Tensor,
    layers: list[dict[str, torch.Tensor]],
    hidden_size: int,
    intermediate_size: int,
    rms_eps: float,
) -> torch.Tensor:
    for layer in layers:
        residual = x
        h = rms_norm(x, layer["input_layernorm"], rms_eps)
        qkv = F.linear(h, layer["qkv"])
        attn_proxy = F.linear(qkv[..., :hidden_size], layer["o"])
        x = residual + attn_proxy

        residual = x
        h = rms_norm(x, layer["post_attention_layernorm"], rms_eps)
        gate_up = F.linear(h, layer["gate_up"])
        gate, up = gate_up.split(intermediate_size, dim=-1)
        mlp = F.linear(F.silu(gate) * up, layer["down"])
        x = residual + mlp
    return x


def benchmark_mode(
    mode: str,
    args: argparse.Namespace,
    cfg: dict,
    reader: IndexedSafetensorsReader,
    quant_state: dict[str, torch.Tensor] | None,
    batch_sizes: list[int],
    dtype: torch.dtype,
) -> list[dict]:
    layers = build_layers(
        mode,
        reader,
        quant_state,
        int(cfg["num_hidden_layers"]),
        args.exit_layer,
        args.device,
        dtype,
    )
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    rms_eps = float(cfg.get("rms_norm_eps", 1e-5))
    rows = []

    for batch_size in batch_sizes:
        x = torch.randn((batch_size, 1, hidden_size), device=args.device, dtype=dtype)
        for _ in range(args.warmup):
            _ = forward_proxy(x, layers, hidden_size, intermediate_size, rms_eps)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = forward_proxy(x, layers, hidden_size, intermediate_size, rms_eps)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = batch_size * args.iters
        rows.append(
            {
                "mode": mode,
                "exit_layer": args.exit_layer,
                "batch_size": batch_size,
                "iters": args.iters,
                "tokens": tokens,
                "elapsed_s": elapsed,
                "tokens_per_s": tokens / elapsed if elapsed > 0 else 0.0,
                "ms_per_token": elapsed * 1000.0 / tokens if tokens else 0.0,
            }
        )

    del layers
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    cfg = load_config(checkpoint)
    reader = IndexedSafetensorsReader(checkpoint)
    modes = [item for item in args.modes.replace(",", " ").split() if item]
    batch_sizes = [int(item) for item in args.batch_sizes.replace(",", " ").split() if item]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    quant_state = None
    if any(mode in {"shared_fp_plus_ptq_tail", "all_ptq_tail"} for mode in modes):
        quant_state = load_quant_state(checkpoint)

    rows: list[dict] = []
    for mode in modes:
        mode_rows = benchmark_mode(mode, args, cfg, reader, quant_state, batch_sizes, dtype)
        rows.extend(mode_rows)
        print(json.dumps({"mode": mode, "rows": mode_rows}, indent=2), flush=True)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "note": "PyTorch fused-linear proxy; attention scores and KV cache are not modeled.",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "exit_layer",
                "batch_size",
                "iters",
                "tokens",
                "elapsed_s",
                "tokens_per_s",
                "ms_per_token",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
