"""RiverEdge custom model prototypes for vLLM.

The vLLM-native class is used for the full-FP and static FP-tail baselines.
The PTQ class is an experimental single-request wrapper around the existing
River/HQQ PyTorch model; it intentionally does not use vLLM paged KV yet.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from vllm.model_executor.models.llama import LlamaForCausalLM


_IGNORED_RIVER_PREFIXES = (
    "model.exit_modules.",
    "model.exit_counts",
)

_HQQ_STATE_KEYS = {
    "W_q",
    "axis",
    "bias",
    "channel_wise",
    "compute_dtype",
    "encoded_state_dict",
    "group_size",
    "nbits",
    "offload_meta",
    "optimize",
    "packing",
    "quant_scale",
    "quant_zero",
    "round_zero",
    "scale",
    "shape",
    "stores_quant_config",
    "unpack_view_dtype",
    "view_as_float",
    "zero",
}


class RiverEdgeLlamaForCausalLM(LlamaForCausalLM):
    """vLLM-native RiverEdge FP model.

    `full_fp` and `static_fp_tail` are equivalent compute graphs in this first
    vLLM stage: both run shared layers 1..k followed by the FP tail k+1..32.
    The explicit mode is kept in config/logs so benchmark output can track the
    intended RiverEdge path.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hf_config = vllm_config.model_config.hf_config
        self.river_edge_mode = getattr(hf_config, "river_edge_mode", "full_fp")
        self.river_edge_exit_layer = int(getattr(hf_config, "river_edge_exit_layer", 3))
        if self.river_edge_mode not in {"full_fp", "static_fp_tail"}:
            raise ValueError(
                "RiverEdgeLlamaForCausalLM only supports full_fp/static_fp_tail; "
                f"got {self.river_edge_mode!r}."
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def filtered_weights():
            for name, tensor in weights:
                if name.startswith(_IGNORED_RIVER_PREFIXES):
                    continue
                yield name, tensor

        return super().load_weights(filtered_weights())


def _add_river_paths(repo_root: str | Path) -> None:
    repo_root = Path(repo_root)
    lm_eval_dir = repo_root / "rivier" / "lm-evaluation"
    hqq_dir = lm_eval_dir / "HQQ-test-from-git"
    for path in (str(hqq_dir),):
        if path not in sys.path:
            sys.path.insert(0, path)


def _ensure_transformers_compat() -> None:
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "LossKwargs"):
        try:
            from typing import TypedDict
        except ImportError:
            from typing_extensions import TypedDict

        class LossKwargs(TypedDict, total=False):
            pass

        transformers_utils.LossKwargs = LossKwargs


def _patch_hqq_meta_init() -> None:
    from hqq.core.quantize import HQQLinear

    if getattr(HQQLinear, "_riveredge_meta_safe_init", False):
        return

    original_init = HQQLinear.__init__

    def meta_safe_init(
        self,
        linear_layer,
        quant_config,
        del_orig: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        initialize: bool = True,
    ):
        weight = getattr(linear_layer, "weight", None) if linear_layer is not None else None
        if initialize and weight is not None and getattr(weight, "is_meta", False):
            original_init(
                self,
                linear_layer,
                quant_config,
                del_orig=del_orig,
                compute_dtype=compute_dtype,
                device=device,
                initialize=False,
            )
            if del_orig and getattr(self, "linear_layer", None) is not None:
                for name, _ in list(self.linear_layer.named_parameters(recurse=False)):
                    setattr(self.linear_layer, name, None)
                self.linear_layer = None
            return

        original_init(
            self,
            linear_layer,
            quant_config,
            del_orig=del_orig,
            compute_dtype=compute_dtype,
            device=device,
            initialize=initialize,
        )

    HQQLinear.__init__ = meta_safe_init
    HQQLinear._riveredge_meta_safe_init = True


def _register_sideway_llama(repo_root: str | Path) -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    _ensure_transformers_compat()
    _patch_hqq_meta_init()
    llama_dir = (
        Path(repo_root)
        / "rivier"
        / "lm-evaluation"
        / "lm_eval"
        / "transformers_extra"
        / "models"
        / "llama"
    )
    package_name = "_riveredge_sideway_llama"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(llama_dir)]
        sys.modules[package_name] = package

    config_module = importlib.import_module(f"{package_name}.configuration_sidewayllama")
    model_module = importlib.import_module(f"{package_name}.modeling_sidewayllama")
    AutoConfig.register(
        "sideway_llama",
        config_module.SidewayLlamaConfig,
        exist_ok=True,
    )
    AutoModelForCausalLM.register(
        config_module.SidewayLlamaConfig,
        model_module.SidewayLlamaForCausalLM,
        exist_ok=True,
    )


def _parse_layer_list(value):
    if isinstance(value, str):
        return [int(x) for x in value.replace(",", " ").split()]
    return value


def _load_river_hf_model(
    model_path: str,
    repo_root: str | Path,
    dtype: torch.dtype,
    quant_backend: str,
):
    _add_river_paths(repo_root)

    from transformers import AutoConfig, AutoModelForCausalLM

    _register_sideway_llama(repo_root)

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

    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            device_map=None,
            attn_implementation="sdpa",
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )
    if quant_backend != "n":
        _manual_load_hqq_weights(model, Path(model_path))
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


def _manual_load_hqq_weights(model: nn.Module, model_path: Path) -> None:
    from hqq.core.quantize import HQQLinear

    index_path = model_path / "pytorch_model.bin.index.json"
    if not index_path.exists():
        return

    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    keys_by_file: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        if "." not in key:
            continue
        prefix, suffix = key.rsplit(".", 1)
        if suffix not in _HQQ_STATE_KEYS or not prefix.startswith("model.exit_modules."):
            continue
        keys_by_file.setdefault(filename, set()).add(key)

    loaded = 0
    for filename, wanted_keys in keys_by_file.items():
        shard = torch.load(model_path / filename, map_location="cpu")
        states: dict[str, dict[str, torch.Tensor]] = {}
        for key in wanted_keys:
            if key not in shard:
                continue
            prefix, suffix = key.rsplit(".", 1)
            states.setdefault(prefix, {})[suffix] = shard[key]
        del shard

        for prefix, state in states.items():
            module = model.get_submodule(prefix)
            if not isinstance(module, HQQLinear):
                continue
            module.load_state_dict(dict(state), strict=False)
            loaded += 1

    unready = [
        name
        for name, module in model.named_modules()
        if isinstance(module, HQQLinear) and not module.is_initialized()
    ]
    if unready:
        preview = ", ".join(unready[:5])
        raise RuntimeError(
            f"Loaded {loaded} HQQLinear modules, but {len(unready)} remain uninitialized: {preview}"
        )


class _DecodeState:
    def __init__(self) -> None:
        self.cache = None


class RiverEdgeHFStaticPTQForCausalLM(nn.Module):
    """Experimental PTQ-tail model under the vLLM engine shell.

    This class keeps a single internal HF/River KV cache and therefore is only
    valid for controlled single-request speed probes (`max_num_seqs=1`). It is
    not a production vLLM paged-KV implementation.
    """

    is_attention_free = True

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        dtype = vllm_config.model_config.dtype
        self.repo_root = getattr(hf_config, "river_edge_repo_root", "/workspace/vllm")
        self.model_path = getattr(
            hf_config,
            "river_edge_original_model_path",
            vllm_config.model_config.model,
        )
        self.exit_layer = int(getattr(hf_config, "river_edge_exit_layer", 3))
        self.quant_backend = getattr(hf_config, "river_edge_quant_backend", "d")
        self.river_model = _load_river_hf_model(
            self.model_path,
            repo_root=self.repo_root,
            dtype=dtype,
            quant_backend=self.quant_backend,
        )
        self.body = self.river_model.model
        self.state = _DecodeState()
        self.vocab_size = int(self.river_model.config.vocab_size)
        self._validate()

    def _validate(self) -> None:
        cfg = self.river_model.config
        if getattr(cfg, "exit_arch", None) != "river":
            raise ValueError(f"Expected exit_arch=river, got {getattr(cfg, 'exit_arch', None)}")
        if self.exit_layer not in cfg.exit_layer_indices:
            raise ValueError(f"exit_layer={self.exit_layer} is not in {cfg.exit_layer_indices}")

    def _get_ptq_tail(self):
        cfg = self.river_model.config
        exit_index = cfg.exit_layer_indices.index(self.exit_layer)
        flow = self.body.exit_modules[0][exit_index + 1 :]
        norm = self.body.exit_modules[1][exit_index]
        return flow, norm

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.body.embed_tokens(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return None

    @torch.inference_mode()
    def _prefill_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.state.cache = None
        outputs = self.body(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            is_first_token=True,
        )
        self.state.cache = outputs.past_key_values
        hidden_states = outputs.last_hidden_states
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[-1]
        return self.river_model.lm_head(hidden_states[:, -1:, :])

    def _make_positions(self, input_ids: torch.Tensor):
        past_seen_tokens = 0 if self.state.cache is None else self.state.cache.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + input_ids.shape[1],
            device=input_ids.device,
        )
        return cache_position, cache_position.unsqueeze(0)

    def _run_layers(
        self,
        layers,
        hidden_states: torch.Tensor,
        causal_mask,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        for layer in layers:
            if isinstance(layer, nn.Identity):
                continue
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=self.state.cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                collect_kv=False,
                update_cache=True,
            )[0]
        return hidden_states

    @torch.inference_mode()
    def _decode_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.body.embed_tokens(input_ids)
        cache_position, position_ids = self._make_positions(input_ids)
        causal_mask = self.body._update_causal_mask(
            None,
            hidden_states,
            cache_position,
            self.state.cache,
            False,
        )
        position_embeddings = self.body.rotary_emb(hidden_states, position_ids)
        hidden_states = self._run_layers(
            self.body.layers[: self.exit_layer],
            hidden_states,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        tail, norm = self._get_ptq_tail()
        hidden_states = self._run_layers(
            tail,
            hidden_states,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        hidden_states = norm(hidden_states)
        return self.river_model.lm_head(hidden_states[:, -1:, :])

    def _expand_last_logits(self, logits: torch.Tensor, num_tokens: int) -> torch.Tensor:
        flat = torch.empty(
            (num_tokens, logits.shape[-1]),
            device=logits.device,
            dtype=logits.dtype,
        )
        flat.zero_()
        flat[-1].copy_(logits[0, -1])
        return flat

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("RiverEdgeHFStaticPTQForCausalLM requires input_ids.")

        flat_ids = input_ids.reshape(-1).to(device="cuda:0")
        flat_positions = positions.reshape(-1)
        if flat_ids.numel() == 0:
            return torch.empty((0, self.vocab_size), device="cuda:0")

        is_prefill = (
            self.state.cache is None
            or flat_ids.numel() > 1
            or int(flat_positions[0].item()) == 0
        )
        if is_prefill:
            logits = self._prefill_logits(flat_ids.unsqueeze(0))
            return self._expand_last_logits(logits, flat_ids.numel())

        outputs = []
        for token in flat_ids:
            logits = self._decode_logits(token.reshape(1, 1))
            outputs.append(logits[0, -1])
        return torch.stack(outputs, dim=0)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states
