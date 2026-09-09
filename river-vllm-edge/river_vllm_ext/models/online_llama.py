"""Eager mixed-route reference with one attention/KV operation per layer.

The runner supplies explicit per-token routes after arranging its input batch.
Only linear projections are compacted. Attention retains the original token
order and metadata, so each logical layer writes each KV position exactly once.
This reference intentionally rejects graph capture; Python route changes must
never silently replay an old route.
"""

from itertools import islice

import torch

from .unified_llama import RiverEdgeUnifiedForCausalLM


class RiverEdgeOnlineForCausalLM(RiverEdgeUnifiedForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("Online mixed-route reference requires enforce_eager=True")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.row_routes = None

    @staticmethod
    def _linear(module, values):
        result = module(values)
        return result[0] if isinstance(result, tuple) else result

    def _project(self, fp, ptq, values, fp_indices, ptq_indices):
        pieces = []
        for module, indices in ((fp, fp_indices), (ptq, ptq_indices)):
            if indices.numel():
                pieces.append((indices, self._linear(module, values[indices])))
        result = values.new_empty((values.shape[0], pieces[0][1].shape[-1]))
        for indices, output in pieces:
            result.index_copy_(0, indices, output)
        return result

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        routes = self.row_routes
        if routes is None or not any(routes):
            return super().forward(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Dynamic Python routes cannot be CUDA Graph captured")
        if intermediate_tensors is not None:
            raise NotImplementedError("Online reference supports PP=1 only")
        if len(routes) != positions.shape[0]:
            raise ValueError("Route rows do not match the runner's token ordering")
        if all(routes):
            return self._forward_ptq_decode(input_ids, positions, None, inputs_embeds)

        hidden = inputs_embeds if inputs_embeds is not None else self.model.embed_input_ids(input_ids)
        fp_indices = torch.tensor([i for i, route in enumerate(routes) if not route],
                                  device=hidden.device, dtype=torch.long)
        ptq_indices = torch.tensor([i for i, route in enumerate(routes) if route],
                                   device=hidden.device, dtype=torch.long)
        residual = None
        for index, layer in enumerate(islice(self.model.layers,
                                             self.model.start_layer,
                                             self.model.end_layer),
                                      start=self.model.start_layer):
            if index < self.river_edge_exit_layer:
                hidden, residual = layer(positions, hidden, residual)
                continue
            tail = self.model.ptq_layers[index]
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            attention = layer.self_attn
            qkv = self._project(attention.qkv_proj, tail.self_attn.qkv_proj,
                                hidden, fp_indices, ptq_indices)
            q, k, v = qkv.split([tail.self_attn.q_size, tail.self_attn.kv_size,
                                 tail.self_attn.kv_size], dim=-1)
            q, k = attention.rotary_emb(positions, q, k)
            hidden = attention.attn(q, k, v)
            hidden = self._project(attention.o_proj, tail.self_attn.o_proj,
                                   hidden, fp_indices, ptq_indices)
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = self._project(layer.mlp, tail.mlp, hidden,
                                   fp_indices, ptq_indices)
        hidden, _ = self.model.norm(hidden, residual)
        return hidden
