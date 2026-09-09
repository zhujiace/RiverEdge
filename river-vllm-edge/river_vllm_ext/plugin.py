"""vLLM plugin entry point for RiverEdge prototype models."""

from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())
    if "RiverEdgeOnlineForCausalLM" not in supported:
        ModelRegistry.register_model(
            "RiverEdgeOnlineForCausalLM",
            "river_vllm_ext.models.online_llama:RiverEdgeOnlineForCausalLM",
        )
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
    if "RiverEdgeUnifiedForCausalLM" not in supported:
        ModelRegistry.register_model(
            "RiverEdgeUnifiedForCausalLM",
            "river_vllm_ext.models.unified_llama:RiverEdgeUnifiedForCausalLM",
        )
