"""Opt-in HQQ LM-head experiment; original checkpoint/FP head are retained.

Quantize vocabulary blocks separately to bound temporary HQQ memory. Only
PTQ decode rows use this candidate; prefill and FP decode use the original
head. This is a quality-changing ablation, never enabled by default.
"""

import torch
from torch import nn


class Int4HeadCandidate:
    def __init__(self, model, block_rows=8192):
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        from torchao.quantization.quantize_.workflows.int4.int4_choose_qparams_algorithm import Int4ChooseQParamsAlgorithm
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat

        self.model = model
        self.original = model.lm_head.quant_method
        self.blocks = []
        config = Int4WeightOnlyConfig(
            group_size=128,
            int4_choose_qparams_algorithm=Int4ChooseQParamsAlgorithm.HQQ,
            int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
            int4_tile_packed_ntile=16,
        )
        weight = model.lm_head.weight
        with torch.no_grad():
            for start in range(0, weight.shape[0], block_rows):
                rows = weight[start:start + block_rows]
                with torch.device("meta"):
                    block = nn.Sequential(nn.Linear(rows.shape[1], rows.shape[0], bias=False))
                block[0].weight = nn.Parameter(rows, requires_grad=False)
                quantize_(block, config)
                self.blocks.append(block)

    def apply(self, layer, hidden_states, bias=None):
        routes = getattr(self.model, "logit_routes", None)
        if routes is None or not any(routes):
            return self.original.apply(layer, hidden_states, bias=bias)
        if bias is not None or len(routes) != hidden_states.shape[0]:
            raise ValueError("Unsupported LM-head bias or row mapping")
        ptq = [i for i, route in enumerate(routes) if route]
        fp = [i for i, route in enumerate(routes) if not route]
        result = hidden_states.new_empty((hidden_states.shape[0], layer.weight.shape[0]))
        if fp:
            indices = torch.tensor(fp, device=hidden_states.device)
            result.index_copy_(0, indices, self.original.apply(layer, hidden_states[indices], bias=None))
        indices = torch.tensor(ptq, device=hidden_states.device)
        selected = hidden_states[indices]
        logits = torch.cat([block(selected) for block in self.blocks], dim=-1)
        result.index_copy_(0, indices, logits)
        return result
