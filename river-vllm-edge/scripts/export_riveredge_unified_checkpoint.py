#!/usr/bin/env python3
"""Build one RiverEdge checkpoint with full FP and layer 2-32 HQQ weights.

The original Hugging Face FP tensors are rewritten with metadata required by
TorchAO's serialized loader. Quantized TorchAO tensor components are copied
from an existing checkpoint, renamed from ``model.layers`` to
``model.ptq_layers``, and stored in one sidecar shard. No layer is quantized
again during this export.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from torchao.core.config import config_to_dict
from torchao.quantization import Int4WeightOnlyConfig, ModuleFqnToConfig
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
COMPONENT_RE = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj))\."
    r"(_weight_(?:qdata|scale_and_zero))$"
)
METADATA_RE = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj))\.weight$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-source", default="/models/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--hqq-source",
        default="/models/Llama-3.1-8B-Instruct-layer2-32-torchao-hqq-fused",
    )
    parser.add_argument(
        "--output",
        default="/models/Llama-3.1-8B-Instruct-riveredge-fp-hqq",
    )
    parser.add_argument("--first-ptq-layer", type=int, default=2, help="1-indexed")
    parser.add_argument("--last-ptq-layer", type=int, default=32, help="inclusive")
    parser.add_argument("--default-exit-layer", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--ntile", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_support_files(source: Path, output: Path) -> None:
    for item in source.iterdir():
        if not item.is_file():
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if item.suffix in {".safetensors", ".bin"}:
            continue
        shutil.copy2(item, output / item.name)


def rewrite_fp_shard(source: Path, destination: Path) -> int:
    """Copy FP tensors while adding metadata required by TorchAO loading."""
    with safe_open(source, framework="pt", device="cpu") as handle:
        tensors = {
            name: handle.get_tensor(name).contiguous() for name in handle.keys()
        }
    metadata = {name: json.dumps({"_type": "Tensor"}) for name in tensors}
    metadata["tensor_names"] = json.dumps(sorted(tensors))
    save_file(tensors, destination, metadata=metadata)
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def make_module_config(
    first_layer: int,
    last_layer: int,
    group_size: int,
    ntile: int,
) -> ModuleFqnToConfig:
    quant_config = Int4WeightOnlyConfig(
        group_size=group_size,
        int4_choose_qparams_algorithm=Int4ChooseQParamsAlgorithm.HQQ,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        int4_tile_packed_ntile=ntile,
    )
    mapping = {
        f"model.ptq_layers.{layer_idx}.{suffix}": quant_config
        for layer_idx in range(first_layer - 1, last_layer)
        for suffix in LINEAR_SUFFIXES
    }
    return ModuleFqnToConfig(mapping)


def extract_ptq_sidecar(
    source_file: Path,
    destination_file: Path,
    first_layer: int,
    last_layer: int,
) -> tuple[list[str], int, int]:
    first_idx = first_layer - 1
    last_idx = last_layer - 1
    tensors = {}
    output_metadata: dict[str, str] = {}

    with safe_open(source_file, framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata() or {}
        for name in handle.keys():
            match = COMPONENT_RE.match(name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            if not first_idx <= layer_idx <= last_idx:
                continue
            output_name = name.replace("model.layers.", "model.ptq_layers.", 1)
            tensors[output_name] = handle.get_tensor(name).contiguous()

        for name, value in source_metadata.items():
            match = METADATA_RE.match(name)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            if not first_idx <= layer_idx <= last_idx:
                continue
            output_name = name.replace("model.layers.", "model.ptq_layers.", 1)
            output_metadata[output_name] = value

    expected_modules = (last_layer - first_layer + 1) * len(LINEAR_SUFFIXES)
    if len(output_metadata) != expected_modules:
        raise RuntimeError(
            f"Expected {expected_modules} quantized modules, found "
            f"{len(output_metadata)} in {source_file}."
        )
    expected_components = expected_modules * 2
    if len(tensors) != expected_components:
        raise RuntimeError(
            f"Expected {expected_components} flattened tensors, found "
            f"{len(tensors)} in {source_file}."
        )

    output_metadata["tensor_names"] = json.dumps(sorted(output_metadata))
    save_file(tensors, destination_file, metadata=output_metadata)
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    return sorted(tensors), expected_modules, tensor_bytes


def write_config(
    output: Path,
    args: argparse.Namespace,
    module_config: ModuleFqnToConfig,
    num_layers: int,
    quantized_modules: int,
) -> None:
    config_path = output / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["architectures"] = ["RiverEdgeUnifiedForCausalLM"]
    config["quantization_config"] = {
        "quant_method": "torchao",
        "quant_type": {"default": config_to_dict(module_config)},
    }
    config["river_edge_mode"] = "static_ptq_tail"
    config["river_edge_exit_layer"] = args.default_exit_layer
    config["river_edge_first_ptq_layer_1_indexed"] = args.first_ptq_layer
    config["river_edge_last_ptq_layer_1_indexed"] = args.last_ptq_layer
    config["river_edge_checkpoint"] = {
        "format_version": 1,
        "layout": "full_fp_plus_torchao_hqq_sidecar",
        "fp_layers_1_indexed": [1, num_layers],
        "ptq_layers_1_indexed": [args.first_ptq_layer, args.last_ptq_layer],
        "ptq_algorithm": "torchao_int4_weight_only_hqq",
        "group_size": args.group_size,
        "ntile": args.ntile,
        "quantized_modules": quantized_modules,
        "prefill_policy": "full_fp",
        "decode_modes": ["full_fp", "static_fp_tail", "static_ptq_tail"],
    }

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_checkpoint_readme(output: Path, args: argparse.Namespace) -> None:
    content = f"""# RiverEdge Unified Checkpoint

This checkpoint contains the complete BF16 Llama model and a TorchAO HQQ
INT4 weight-only copy of layers {args.first_ptq_layer}-{args.last_ptq_layer}.

- FP weights: `model.layers.0..31`
- PTQ weights: `model.ptq_layers.{args.first_ptq_layer - 1}..{args.last_ptq_layer - 1}`
- Default shared FP depth: `k={args.default_exit_layer}`
- Prefill policy: full FP
- Decode modes: `full_fp`, `static_fp_tail`, `static_ptq_tail`

Select a mode and k through vLLM `hf_overrides`; do not generate a separate
checkpoint for each k. The custom architecture is provided by the
`river-vllm-edge` plugin.
"""
    (output / "RIVEREDGE_CHECKPOINT.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    fp_source = Path(args.fp_source).resolve()
    hqq_source = Path(args.hqq_source).resolve()
    output = Path(args.output).resolve()
    start = time.time()

    fp_index_path = fp_source / "model.safetensors.index.json"
    hqq_file = hqq_source / "model.safetensors"
    if not fp_index_path.exists():
        raise FileNotFoundError(fp_index_path)
    if not hqq_file.exists():
        raise FileNotFoundError(hqq_file)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists, pass --force to replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copy_support_files(fp_source, output)
    with fp_index_path.open("r", encoding="utf-8") as handle:
        fp_index = json.load(handle)
    with (fp_source / "config.json").open("r", encoding="utf-8") as handle:
        fp_config = json.load(handle)
    num_layers = int(fp_config["num_hidden_layers"])
    if args.last_ptq_layer > num_layers:
        raise ValueError(
            f"last PTQ layer {args.last_ptq_layer} exceeds model depth {num_layers}."
        )

    fp_shard_bytes: dict[str, int] = {}
    fp_files = sorted(set(fp_index["weight_map"].values()))
    for filename in fp_files:
        fp_shard_bytes[filename] = rewrite_fp_shard(
            fp_source / filename,
            output / filename,
        )

    ptq_filename = "riveredge-hqq-layer2-32.safetensors"
    component_names, quantized_modules, ptq_tensor_bytes = extract_ptq_sidecar(
        hqq_file,
        output / ptq_filename,
        args.first_ptq_layer,
        args.last_ptq_layer,
    )

    module_config = make_module_config(
        args.first_ptq_layer,
        args.last_ptq_layer,
        args.group_size,
        args.ntile,
    )
    write_config(output, args, module_config, num_layers, quantized_modules)
    write_checkpoint_readme(output, args)

    weight_map = dict(fp_index["weight_map"])
    weight_map.update({name: ptq_filename for name in component_names})
    fp_total_size = int(fp_index.get("metadata", {}).get("total_size", 0))
    unified_index = {
        "metadata": {
            "total_size": fp_total_size + ptq_tensor_bytes,
            "river_edge_fp_bytes": fp_total_size,
            "river_edge_ptq_bytes": ptq_tensor_bytes,
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    with (output / "model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(unified_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary = {
        "fp_source": str(fp_source),
        "hqq_source": str(hqq_source),
        "output": str(output),
        "fp_files": fp_files,
        "fp_shard_bytes": fp_shard_bytes,
        "fp_transfer_mode": "torchao_metadata_rewrite",
        "ptq_file": ptq_filename,
        "fp_layers_1_indexed": [1, num_layers],
        "ptq_layers_1_indexed": [args.first_ptq_layer, args.last_ptq_layer],
        "quantized_modules": quantized_modules,
        "ptq_flat_tensors": len(component_names),
        "fp_tensor_bytes": fp_total_size,
        "ptq_tensor_bytes": ptq_tensor_bytes,
        "total_tensor_bytes": fp_total_size + ptq_tensor_bytes,
        "elapsed_sec": round(time.time() - start, 3),
    }
    with (output / "riveredge_export_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
