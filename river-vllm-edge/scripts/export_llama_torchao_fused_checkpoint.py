#!/usr/bin/env python3
"""Export a vLLM-loadable Llama checkpoint with serialized torchao INT4 weights.

The output checkpoint uses vLLM's fused parameter names for Llama:
q/k/v are concatenated into qkv_proj and gate/up are concatenated into
gate_up_proj. Selected layers are quantized with torchao Int4WeightOnlyConfig
using the HQQ qparam algorithm, flattened with torchao safetensors metadata,
and embedded in config.json as a serialized torchao checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torchao.core.config import config_to_dict
from torchao.prototype.safetensors.safetensors_support import (
    flatten_tensor_state_dict,
)
from torchao.quantization import Int4WeightOnlyConfig, ModuleFqnToConfig, quantize_
from torchao.quantization.quantize_.workflows.int4.int4_choose_qparams_algorithm import (
    Int4ChooseQParamsAlgorithm,
)
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
    Int4PackingFormat,
)


LINEAR_SUFFIXES = (
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/models/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--output",
        default="/models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-fused",
    )
    parser.add_argument("--first-quant-layer", type=int, default=2, help="1-indexed")
    parser.add_argument("--last-quant-layer", type=int, default=32, help="inclusive")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--ntile", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class IndexedSafetensorsReader:
    def __init__(self, model_dir: Path):
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing safetensors index: {index_path}")
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        self.model_dir = model_dir
        self.weight_map: dict[str, str] = index["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        try:
            shard_name = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"Weight not found in source index: {name}") from exc
        with safe_open(self.model_dir / shard_name, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).contiguous()

    def has(self, name: str) -> bool:
        return name in self.weight_map


def make_int4_config(group_size: int, ntile: int) -> Int4WeightOnlyConfig:
    return Int4WeightOnlyConfig(
        group_size=group_size,
        int4_choose_qparams_algorithm=Int4ChooseQParamsAlgorithm.HQQ,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        int4_tile_packed_ntile=ntile,
    )


def build_module_config(
    first_layer: int,
    last_layer: int,
    quant_cfg: Int4WeightOnlyConfig,
) -> tuple[ModuleFqnToConfig, dict[str, Int4WeightOnlyConfig]]:
    layer_indices = range(first_layer - 1, last_layer)
    mapping = {
        f"model.layers.{layer_idx}.{suffix}": quant_cfg
        for layer_idx in layer_indices
        for suffix in LINEAR_SUFFIXES
    }
    return ModuleFqnToConfig(mapping), mapping


def copy_non_weight_files(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == ".cache":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.suffix in {".safetensors", ".bin"}:
            continue

        dst = output / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst, symlinks=True)
        else:
            shutil.copy2(item, dst)


def quantize_weight(
    weight: torch.Tensor,
    quant_cfg: Int4WeightOnlyConfig,
    device: str,
) -> torch.Tensor:
    weight_gpu = weight.to(device=device, non_blocking=False)
    with torch.device("meta"):
        module = nn.Sequential(
            nn.Linear(weight_gpu.shape[1], weight_gpu.shape[0], bias=False)
        )
    module[0].weight = nn.Parameter(weight_gpu, requires_grad=False)
    quantize_(module, quant_cfg)
    quant_weight = module[0].weight.detach().cpu()
    del module, weight_gpu
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return quant_weight


def add_layer_weights(
    state: dict[str, torch.Tensor],
    reader: IndexedSafetensorsReader,
    layer_idx: int,
    quantized: bool,
    quant_cfg: Int4WeightOnlyConfig,
    device: str,
) -> None:
    hf_prefix = f"model.layers.{layer_idx}"
    vllm_prefix = f"model.layers.{layer_idx}"

    state[f"{vllm_prefix}.input_layernorm.weight"] = reader.get(
        f"{hf_prefix}.input_layernorm.weight"
    )
    state[f"{vllm_prefix}.post_attention_layernorm.weight"] = reader.get(
        f"{hf_prefix}.post_attention_layernorm.weight"
    )

    qkv = torch.cat(
        [
            reader.get(f"{hf_prefix}.self_attn.q_proj.weight"),
            reader.get(f"{hf_prefix}.self_attn.k_proj.weight"),
            reader.get(f"{hf_prefix}.self_attn.v_proj.weight"),
        ],
        dim=0,
    ).contiguous()
    gate_up = torch.cat(
        [
            reader.get(f"{hf_prefix}.mlp.gate_proj.weight"),
            reader.get(f"{hf_prefix}.mlp.up_proj.weight"),
        ],
        dim=0,
    ).contiguous()

    linear_weights = {
        f"{vllm_prefix}.self_attn.qkv_proj.weight": qkv,
        f"{vllm_prefix}.self_attn.o_proj.weight": reader.get(
            f"{hf_prefix}.self_attn.o_proj.weight"
        ),
        f"{vllm_prefix}.mlp.gate_up_proj.weight": gate_up,
        f"{vllm_prefix}.mlp.down_proj.weight": reader.get(
            f"{hf_prefix}.mlp.down_proj.weight"
        ),
    }

    for name, weight in linear_weights.items():
        state[name] = quantize_weight(weight, quant_cfg, device) if quantized else weight


def write_config(
    output: Path,
    args: argparse.Namespace,
    module_config: ModuleFqnToConfig,
    quantized_modules: int,
) -> None:
    config_path = output / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["quantization_config"] = {
        "quant_method": "torchao",
        "quant_type": {"default": config_to_dict(module_config)},
    }
    config["river_edge_quantization"] = {
        "mode": "vllm_torchao_serialized_fused",
        "algorithm": "torchao_int4_weight_only_hqq",
        "weight_bits": 4,
        "group_size": args.group_size,
        "ntile": args.ntile,
        "first_quant_layer_1_indexed": args.first_quant_layer,
        "last_quant_layer_1_indexed": args.last_quant_layer,
        "quantized_modules": quantized_modules,
        "layout": "vllm_fused_qkv_and_gate_up",
    }

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not (source / "config.json").exists():
        raise FileNotFoundError(f"Missing source config.json: {source / 'config.json'}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists, pass --force to replace: {output}")
        shutil.rmtree(output)

    start = time.time()
    copy_non_weight_files(source, output)
    reader = IndexedSafetensorsReader(source)
    quant_cfg = make_int4_config(args.group_size, args.ntile)
    module_config, module_mapping = build_module_config(
        args.first_quant_layer, args.last_quant_layer, quant_cfg
    )
    write_config(output, args, module_config, len(module_mapping))

    with (source / "config.json").open("r", encoding="utf-8") as handle:
        hf_config = json.load(handle)
    num_layers = int(hf_config["num_hidden_layers"])

    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": reader.get("model.embed_tokens.weight"),
        "model.norm.weight": reader.get("model.norm.weight"),
    }
    if reader.has("lm_head.weight"):
        state["lm_head.weight"] = reader.get("lm_head.weight")

    quantized_layer_ids: list[int] = []
    for layer_idx in range(num_layers):
        quantized = args.first_quant_layer - 1 <= layer_idx <= args.last_quant_layer - 1
        add_layer_weights(state, reader, layer_idx, quantized, quant_cfg, args.device)
        if quantized:
            quantized_layer_ids.append(layer_idx)
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "layer_done",
                    "layer_idx": layer_idx,
                    "quantized": quantized,
                    "state_tensors": len(state),
                }
            ),
            flush=True,
        )

    flat_state, metadata = flatten_tensor_state_dict(state)
    cpu_flat_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in flat_state.items()
    }
    save_path = output / "model.safetensors"
    save_file(cpu_flat_state, str(save_path), metadata=metadata)

    summary = {
        "source": str(source),
        "output": str(output),
        "weights": str(save_path),
        "layers": num_layers,
        "quantized_layers_0_indexed": quantized_layer_ids,
        "quantized_modules": len(module_mapping),
        "flat_tensors": len(cpu_flat_state),
        "elapsed_sec": round(time.time() - start, 3),
    }
    with (output / "river_edge_export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
