#!/usr/bin/env python3
"""Create vLLM-readable adapter directories for River checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/media/orin/Data/models/Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16",
    )
    parser.add_argument(
        "--output-root",
        default="/home/orin/zjc/vllm/RiverEdge/experiments/07_vllm_custom_model/model_adapters",
    )
    parser.add_argument("--exit-layer", type=int, default=3)
    parser.add_argument("--repo-root-in-container", default="/workspace/vllm")
    parser.add_argument("--model-path-in-container", default="/models/Smollm_8B_river_quant_copyinit_kvdistill_weight500/q_4bit_exit_bf16")
    return parser.parse_args()


def link_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        dst = target / item.name
        if item.name == "config.json":
            continue
        if dst.exists() or dst.is_symlink():
            continue
        rel = os.path.relpath(item, target)
        dst.symlink_to(rel)


def write_config(source: Path, target: Path, arch: str, mode: str, args) -> None:
    with (source / "config.json").open() as f:
        cfg = json.load(f)
    cfg["river_edge_original_model_type"] = cfg.get("model_type")
    cfg["river_edge_original_architectures"] = cfg.get("architectures")
    cfg["model_type"] = "llama"
    cfg["architectures"] = [arch]
    cfg["river_edge_mode"] = mode
    cfg["river_edge_exit_layer"] = args.exit_layer
    cfg["river_edge_original_model_path"] = args.model_path_in_container
    cfg["river_edge_repo_root"] = args.repo_root_in_container
    cfg["river_edge_quant_backend"] = "d"
    with (target / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output_root = Path(args.output_root).resolve()

    adapters = {
        "full_fp": ("RiverEdgeLlamaForCausalLM", "full_fp"),
        "static_fp_tail": ("RiverEdgeLlamaForCausalLM", "static_fp_tail"),
        "static_ptq_tail": ("RiverEdgeHFStaticPTQForCausalLM", "static_ptq_tail"),
    }
    for name, (arch, mode) in adapters.items():
        target = output_root / f"{name}_k{args.exit_layer}"
        link_tree(source, target)
        write_config(source, target, arch, mode, args)
        print(target)


if __name__ == "__main__":
    main()
