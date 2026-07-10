"""vLLM plugin entry point for RiverEdge prototype models."""

from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())
    if "RiverEdgeLlamaForCausalLM" not in supported:
        ModelRegistry.register_model(
            "RiverEdgeLlamaForCausalLM",
            "river_vllm_ext.models.routed_llama:RiverEdgeLlamaForCausalLM",
        )
    if "RiverEdgeHFStaticPTQForCausalLM" not in supported:
        ModelRegistry.register_model(
            "RiverEdgeHFStaticPTQForCausalLM",
            "river_vllm_ext.models.routed_llama:RiverEdgeHFStaticPTQForCausalLM",
        )
