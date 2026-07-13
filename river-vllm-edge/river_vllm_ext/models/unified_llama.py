"""Unified FP + TorchAO HQQ Llama model for RiverEdge.

The checkpoint stores a complete FP Llama under ``model.layers`` and an
additional INT4 weight-only copy of linear weights under ``model.ptq_layers``.
Prefill always executes the FP model.  Static PTQ mode switches only pure
decode batches after the shared FP prefix, while reusing the FP layer's
normalization, RoPE, attention operator, and paged KV cache.
"""

from __future__ import annotations

from itertools import islice

import torch
from torch import nn

from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.sequence import IntermediateTensors


logger = init_logger(__name__)


class RiverEdgePTQAttentionWeights(nn.Module):
    """Quantized projections that reuse an FP Llama attention operator."""

    def __init__(self, *, config, quant_config, prefix: str) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = getattr(
            config, "num_key_value_heads", total_num_heads
        )
        if total_num_kv_heads >= tp_size:
            assert total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % total_num_kv_heads == 0

        num_heads = total_num_heads // tp_size
        num_kv_heads = max(1, total_num_kv_heads // tp_size)
        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or config.hidden_size // total_num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=total_num_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        *,
        fp_attention: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = fp_attention.rotary_emb(positions, q, k)
        attn_output = fp_attention.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class RiverEdgePTQMLPWeights(nn.Module):
    """Quantized MLP projections for one Llama decoder layer."""

    def __init__(self, *, config, quant_config, prefix: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=getattr(config, "mlp_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=getattr(config, "mlp_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class RiverEdgePTQTailLayer(nn.Module):
    """PTQ linear weights paired with the non-linear state of an FP layer."""

    def __init__(self, *, config, quant_config, prefix: str) -> None:
        super().__init__()
        self.self_attn = RiverEdgePTQAttentionWeights(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = RiverEdgePTQMLPWeights(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        *,
        fp_layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = fp_layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fp_layer.input_layernorm(
                hidden_states, residual
            )

        hidden_states = self.self_attn(
            fp_attention=fp_layer.self_attn,
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = fp_layer.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class RiverEdgeUnifiedForCausalLM(LlamaForCausalLM):
    """Llama with full-FP prefill and selectable FP/HQQ decode tails.

    Supported static modes:

    * ``full_fp``: native FP Llama for prefill and decode.
    * ``static_fp_tail``: explicit RiverEdge FP baseline; same graph as full FP.
    * ``static_ptq_tail``: FP prefill, then FP layers ``1..k`` and HQQ layers
      ``k+1..32`` during pure decode batches.

    Mixed prefill/decode batches conservatively execute the FP graph.  This
    preserves the full-precision prefill contract until route-aware batching
    is implemented.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.river_edge_mode = getattr(
            config, "river_edge_mode", "static_ptq_tail"
        )
        self.river_edge_exit_layer = int(
            getattr(config, "river_edge_exit_layer", 3)
        )
        self.river_edge_first_ptq_layer = int(
            getattr(config, "river_edge_first_ptq_layer_1_indexed", 2)
        )
        supported_modes = {"full_fp", "static_fp_tail", "static_ptq_tail"}
        if self.river_edge_mode not in supported_modes:
            raise ValueError(
                f"Unsupported river_edge_mode={self.river_edge_mode!r}; "
                f"expected one of {sorted(supported_modes)}."
            )
        if not (
            self.river_edge_first_ptq_layer - 1
            <= self.river_edge_exit_layer
            < config.num_hidden_layers
        ):
            raise ValueError(
                "river_edge_exit_layer must leave a PTQ tail covered by the "
                f"checkpoint; got k={self.river_edge_exit_layer}, first PTQ layer="
                f"{self.river_edge_first_ptq_layer}."
            )

        ptq_layers: list[nn.Module] = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx < self.river_edge_first_ptq_layer - 1:
                ptq_layers.append(nn.Identity())
            else:
                ptq_layers.append(
                    RiverEdgePTQTailLayer(
                        config=config,
                        quant_config=quant_config,
                        prefix=f"model.ptq_layers.{layer_idx}",
                    )
                )
        self.model.ptq_layers = nn.ModuleList(ptq_layers)
        self._traced_phases: set[str] = set()

    def _is_pure_decode_batch(self) -> bool:
        if not is_forward_context_available():
            return False
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return False
        metadata_groups = metadata if isinstance(metadata, list) else [metadata]
        found = False
        for group in metadata_groups:
            for item in group.values():
                max_query_len = getattr(item, "max_query_len", None)
                max_seq_len = getattr(item, "max_seq_len", None)
                if max_query_len is None or max_seq_len is None:
                    return False
                found = True
                # A one-token initial prompt also has max_query_len == 1.
                # It has no prior context, so keep it on the FP path.
                if max_query_len != 1 or max_seq_len <= 1:
                    return False
        return found

    def _trace_phase_once(self, phase: str) -> None:
        if phase in self._traced_phases:
            return
        self._traced_phases.add(phase)
        logger.info(
            "RiverEdge execution phase=%s mode=%s k=%d",
            phase,
            self.river_edge_mode,
            self.river_edge_exit_layer,
        )

    def _forward_ptq_decode(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor | IntermediateTensors:
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise NotImplementedError(
                "RiverEdge unified PTQ decode currently supports pipeline_parallel_size=1."
            )
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.model.embed_input_ids(input_ids)
        residual = None

        for layer_idx, fp_layer in enumerate(
            islice(
                self.model.layers,
                self.model.start_layer,
                self.model.end_layer,
            ),
            start=self.model.start_layer,
        ):
            if layer_idx < self.river_edge_exit_layer:
                hidden_states, residual = fp_layer(
                    positions, hidden_states, residual
                )
            else:
                ptq_layer = self.model.ptq_layers[layer_idx]
                if not isinstance(ptq_layer, RiverEdgePTQTailLayer):
                    raise RuntimeError(f"Missing PTQ weights for layer {layer_idx + 1}.")
                hidden_states, residual = ptq_layer(
                    fp_layer=fp_layer,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )

        hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        use_ptq = (
            self.river_edge_mode == "static_ptq_tail"
            and self._is_pure_decode_batch()
        )
        if use_ptq:
            self._trace_phase_once("decode_ptq")
            return self._forward_ptq_decode(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )

        phase = (
            "prefill_fp"
            if self.river_edge_mode == "static_ptq_tail"
            else "all_fp"
        )
        self._trace_phase_once(phase)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
