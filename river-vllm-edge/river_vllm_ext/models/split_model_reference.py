#!/usr/bin/env python3
"""PyTorch reference for RiverEdge routed dual-tail decoding.

Design implemented here:

* Prefill uses the normal full-FP backbone and writes the initial KV cache.
* Decode runs shared FP layers 1..k.
* Decode then routes to either FP layers k+1..32 or PTQ exit layers k+1..32.

Layer numbers in the public API are 1-indexed. Internally, Python module
indices are 0-indexed, so k=3 means FP layers [0, 1, 2] are shared and tail
layers start from module index 3.
"""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DecodeStepOutput:
    logits: torch.Tensor
    route: str
    route_score: float
    threshold: float


@dataclass
class GenerationStats:
    prefill_s: float = 0.0
    decode_s: float = 0.0
    generated_tokens: int = 0
    decode_steps: int = 0
    route_counts: Dict[str, int] = field(default_factory=lambda: {"ptq": 0, "fp": 0})
    route_scores: List[float] = field(default_factory=list)

    @property
    def decode_tps(self) -> float:
        return self.decode_steps / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s


def add_lm_eval_to_path(repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    lm_eval_dir = repo_root / "rivier" / "lm-evaluation"
    hqq_dir = lm_eval_dir / "HQQ-test-from-git"
    for path in (str(lm_eval_dir), str(hqq_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _parse_layer_list(value):
    if isinstance(value, str):
        return [int(x) for x in value.replace(",", " ").split()]
    return value


def load_river_model(
    model_path: str,
    repo_root: str | Path = "/workspace/vllm",
    dtype: torch.dtype = torch.bfloat16,
    quant_backend: str = "t",
    attn_implementation: str = "sdpa",
):
    """Load the existing River checkpoint without modifying River source code."""

    add_lm_eval_to_path(repo_root)

    from transformers import AutoConfig, AutoModelForCausalLM

    importlib.import_module("lm_eval.transformers_extra.models")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.quantize_exit_layers = True
    config.quantize_full_model = False
    config.quant_backend = quant_backend
    config.output_full_model = True
    config.early_exit_threshold = None
    config.count_distribution = False
    config.collect_kv = False
    config.compile = False
    config.output_hidden_states = False
    config.output_attentions = False
    config.exit_layer_indices = _parse_layer_list(config.exit_layer_indices)
    config.output_exit_layers = _parse_layer_list(config.output_exit_layers)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    )
    model = model.to("cuda:0").to(dtype=dtype).eval()

    if quant_backend != "n":
        from hqq.core.quantize import HQQBackend, HQQLinear
        from hqq.utils.patching import prepare_for_inference

        HQQLinear.set_backend(HQQBackend.PYTORCH)
        backend_map = {"d": None, "g": "gemlite", "t": "torchao_int4"}
        backend = backend_map.get(quant_backend)
        if backend:
            prepare_for_inference(model, backend=backend, verbose=False)
        else:
            prepare_for_inference(model, verbose=False)

    return model


class RiverRoutedDualTailModel:
    """Single-request PyTorch routed dual-tail reference.

    This class keeps one mixed KV cache. Prefill writes FP KV for all layers.
    Decode updates shared FP-prefix KV, then updates the selected FP/PTQ tail
    KV in the same cache. This mirrors the intended fast path for a prototype,
    but it intentionally does not solve future vLLM paged-KV separation yet.
    """

    def __init__(self, model, exit_layer: int = 3, route_threshold: float = 0.5):
        self.model = model.eval()
        self.body = model.model
        self.exit_layer = int(exit_layer)
        self.route_threshold = float(route_threshold)
        self.cache = None
        self._validate()

    def _validate(self) -> None:
        cfg = self.model.config
        if getattr(cfg, "exit_arch", None) != "river":
            raise ValueError(f"Expected exit_arch=river, got {getattr(cfg, 'exit_arch', None)}")
        if not getattr(cfg, "exit_decoder_layer", False):
            raise ValueError("This reference requires decoder-layer exits.")
        if self.exit_layer not in cfg.exit_layer_indices:
            raise ValueError(f"exit_layer={self.exit_layer} is not in {cfg.exit_layer_indices}")
        if self.exit_layer >= cfg.num_hidden_layers:
            raise ValueError("exit_layer must be smaller than num_hidden_layers.")
        self._get_ptq_tail()

    def _get_ptq_tail(self) -> Tuple[Iterable[nn.Module], nn.Module]:
        exit_index = self.model.config.exit_layer_indices.index(self.exit_layer)
        flow = self.body.exit_modules[0][exit_index + 1 :]
        norm = self.body.exit_modules[1][exit_index]
        real_layers = [module for module in flow if not isinstance(module, nn.Identity)]
        expected = self.model.config.num_hidden_layers - self.exit_layer
        if len(real_layers) != expected:
            raise RuntimeError(f"PTQ tail length mismatch: got {len(real_layers)}, expected {expected}")
        return flow, norm

    def reset_cache(self) -> None:
        self.cache = None

    def _make_positions(self, input_ids: torch.Tensor):
        if self.cache is None:
            past_seen_tokens = 0
        else:
            past_seen_tokens = self.cache.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + input_ids.shape[1],
            device=input_ids.device,
        )
        position_ids = cache_position.unsqueeze(0)
        return cache_position, position_ids

    def _causal_mask(self, attention_mask, hidden_states, cache_position):
        return self.body._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            self.cache,
            False,
        )

    def _run_layers(
        self,
        layers: Iterable[nn.Module],
        hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_embeddings,
        update_cache: bool,
    ) -> torch.Tensor:
        for layer in layers:
            if isinstance(layer, nn.Identity):
                continue
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=self.cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                collect_kv=False,
                update_cache=update_cache,
            )[0]
        return hidden_states

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run normal full-FP prefill and initialize KV cache."""

        self.reset_cache()
        outputs = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            is_first_token=True,
        )
        self.cache = outputs.past_key_values
        hidden_states = outputs.last_hidden_states
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[-1]
        return self.model.lm_head(hidden_states[:, -1:, :])

    def _route(self, hidden_states: torch.Tensor, previous_hidden_states: torch.Tensor) -> Tuple[str, float]:
        cos_sim = F.cosine_similarity(hidden_states, previous_hidden_states, dim=-1)
        score = float(cos_sim.mean().item())
        route = "ptq" if bool((cos_sim > self.route_threshold).all().item()) else "fp"
        return route, score

    @torch.inference_mode()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        force_route: Optional[str] = None,
    ) -> DecodeStepOutput:
        """Decode one token through shared FP prefix and selected tail."""

        if self.cache is None:
            raise RuntimeError("decode_step requires prefill() to initialize cache first.")
        if input_ids.shape[1] != 1:
            raise ValueError("decode_step expects input_ids with sequence length 1.")

        hidden_states = self.body.embed_tokens(input_ids)
        cache_position, position_ids = self._make_positions(input_ids)
        causal_mask = self._causal_mask(attention_mask, hidden_states, cache_position)
        position_embeddings = self.body.rotary_emb(hidden_states, position_ids)

        previous_hidden_states = hidden_states
        for layer in self.body.layers[: self.exit_layer]:
            previous_hidden_states = hidden_states
            hidden_states = self._run_layers(
                [layer],
                hidden_states,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
                update_cache=True,
            )

        route, score = self._route(hidden_states, previous_hidden_states)
        if force_route is not None:
            if force_route not in {"ptq", "fp"}:
                raise ValueError("force_route must be one of: ptq, fp")
            route = force_route

        if route == "ptq":
            tail, norm = self._get_ptq_tail()
            hidden_states = self._run_layers(
                tail,
                hidden_states,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
                update_cache=True,
            )
            hidden_states = norm(hidden_states)
        else:
            hidden_states = self._run_layers(
                self.body.layers[self.exit_layer :],
                hidden_states,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
                update_cache=True,
            )
            hidden_states = self.body.norm(hidden_states)

        logits = self.model.lm_head(hidden_states[:, -1:, :])
        return DecodeStepOutput(
            logits=logits,
            route=route,
            route_score=score,
            threshold=self.route_threshold,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: Optional[torch.Tensor] = None,
        force_route: Optional[str] = None,
    ) -> Tuple[torch.Tensor, GenerationStats]:
        """Greedy generation. The first new token comes from full-FP prefill."""

        stats = GenerationStats()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = self.prefill(input_ids, attention_mask=attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats.prefill_s = time.perf_counter() - start

        generated: List[torch.Tensor] = []
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

        for _ in range(max(max_new_tokens - 1, 0)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            step = self.decode_step(next_token, force_route=force_route)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stats.decode_s += time.perf_counter() - start
            stats.decode_steps += 1
            stats.route_counts[step.route] = stats.route_counts.get(step.route, 0) + 1
            stats.route_scores.append(step.route_score)
            next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

        stats.generated_tokens = len(generated)
        return torch.cat(generated, dim=1), stats
