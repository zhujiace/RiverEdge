#!/usr/bin/env python3
"""Create a vLLM adapter that quantizes selected Llama layers with torchao.

The adapter links to an existing FP Hugging Face checkpoint and writes a
torchao ModuleFqnToConfig file. vLLM should be launched with
quantization="torchao" and hf_overrides={"quantization_config_file": ...}.
Do not embed this config in config.json: vLLM treats that path as a serialized
torchao checkpoint instead of online quantization from FP weights.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


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
        default="/models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-online",
    )
    parser.add_argument("--layer-start", type=int, default=2, help="1-indexed")
    parser.add_argument("--layer-end", type=int, default=32, help="1-indexed, inclusive")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--ntile", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_torchao_config(layer_start: int, layer_end: int, group_size: int, ntile: int) -> dict:
    from torchao.core.config import config_to_dict
    from torchao.quantization import Int4WeightOnlyConfig, ModuleFqnToConfig
    from torchao.quantization.quantize_.workflows.int4.int4_choose_qparams_algorithm import (
        Int4ChooseQParamsAlgorithm,
    )
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
        Int4PackingFormat,
    )

    layer_indices = range(layer_start - 1, layer_end)
    quant_cfg = Int4WeightOnlyConfig(
        group_size=group_size,
        int4_choose_qparams_algorithm=Int4ChooseQParamsAlgorithm.HQQ,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        int4_tile_packed_ntile=ntile,
    )
    module_fqn_to_config = {
        f"model.layers.{layer_idx}.{suffix}": quant_cfg
        for layer_idx in layer_indices
        for suffix in LINEAR_SUFFIXES
    }
    return config_to_dict(ModuleFqnToConfig(module_fqn_to_config))


def link_checkpoint_files(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == ".cache":
            continue
        dst = output / item.name
        if dst.exists() or dst.is_symlink():
            continue
        if item.name == "config.json":
            shutil.copy2(item, dst)
            continue
        rel = os.path.relpath(item, output)
        dst.symlink_to(rel, target_is_directory=item.is_dir())


def add_adapter_metadata(config_path: Path, args: argparse.Namespace, quant_config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["river_edge_quantization"] = {
        "mode": "vllm_torchao_online",
        "algorithm": "hqq",
        "weight_bits": 4,
        "group_size": args.group_size,
        "layer_start_1_indexed": args.layer_start,
        "layer_end_1_indexed": args.layer_end,
        "vllm_quantization": "torchao",
        "quantization_config_file": str(quant_config_path),
        "note": "Passed through hf_overrides; not embedded as HF quantization_config.",
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not (source / "config.json").exists():
        raise FileNotFoundError(f"Missing source config.json: {source}")
    if output.exists() and args.force:
        shutil.rmtree(output)

    link_checkpoint_files(source, output)
    quant_config = build_torchao_config(args.layer_start, args.layer_end, args.group_size, args.ntile)
    quant_config_path = output / "torchao_layer_quant_config.json"
    with quant_config_path.open("w", encoding="utf-8") as handle:
        json.dump(quant_config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    add_adapter_metadata(output / "config.json", args, quant_config_path)

    module_map = quant_config["_data"]["module_fqn_to_config"]
    print(json.dumps({
        "adapter": str(output),
        "source": str(source),
        "quant_config_file": str(quant_config_path),
        "quantized_modules": len(module_map),
    }, indent=2))


if __name__ == "__main__":
    main()
