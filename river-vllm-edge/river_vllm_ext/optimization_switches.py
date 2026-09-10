"""Explicit, process-lifetime RiverEdge switches; all off uses stock vLLM.

No torch/vLLM imports at module import time. Native vLLM optimizations retain
their defaults unless the caller explicitly supplies ordinary vLLM options.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class OptimizationSwitches:
    ptq_tail: bool = False
    online_routing: bool = False
    grouped_scheduling: bool = False
    batch_aware_routing: bool = False
    conservative_fallback: bool = False
    int4_head: bool = False
    decode_cudagraph: bool = False

    def validate(self):
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError('Optimization switches must be booleans')
        if self.online_routing and not self.ptq_tail:
            raise ValueError('online_routing requires ptq_tail')
        if any((self.grouped_scheduling, self.batch_aware_routing,
                self.conservative_fallback, self.int4_head)) and not self.online_routing:
            raise ValueError('Scheduling, adaptive routing and INT4 head require online_routing')
        if self.online_routing and self.decode_cudagraph:
            raise ValueError('Online routing currently requires eager; dynamic Graph is not implemented')

    @property
    def all_off(self):
        return not any(asdict(self).values())

    @property
    def architecture(self):
        if self.online_routing:
            return 'RiverEdgeOnlineForCausalLM'
        return 'RiverEdgeUnifiedForCausalLM' if self.ptq_tail else 'LlamaForCausalLM'


class SwitchRoutePolicy:
    """Independent hysteresis and mixed-eligibility fallback switches."""

    def __init__(self, switches, low=8, high=16):
        switches.validate()
        if not 0 < low < high:
            raise ValueError('Require 0 < low < high')
        self.switches, self.low, self.high = switches, low, high
        self.use_ptq, self.mixed_eligibility = True, False

    def update(self, active_decode, eligible_decode=None):
        self.mixed_eligibility = (eligible_decode is not None
                                  and 0 < eligible_decode < active_decode)
        if self.switches.batch_aware_routing:
            if active_decode >= self.high:
                self.use_ptq = False
            elif active_decode <= self.low:
                self.use_ptq = True

    def select(self, extra_args):
        if not (extra_args or {}).get('riveredge_allow_ptq', False):
            return 'fp'
        if self.switches.conservative_fallback and self.mixed_eligibility:
            return 'fp'
        return 'ptq' if self.use_ptq else 'fp'


def validate_checkpoint(path, switches):
    """Fail closed: never reinterpret a dual-tail checkpoint as a native model."""
    config = json.loads((Path(path) / 'config.json').read_text())
    if switches.ptq_tail:
        if config.get('architectures') != ['RiverEdgeUnifiedForCausalLM']:
            raise ValueError('PTQ requires a RiverEdge unified FP+HQQ checkpoint')
    else:
        if (config.get('architectures') != ['LlamaForCausalLM']
                or config.get('quantization_config')
                or any(key.startswith('river_edge_') for key in config)):
            raise ValueError('Native mode requires an unquantized native Llama checkpoint, not a RiverEdge checkpoint')
        index = Path(path) / 'model.safetensors.index.json'
        if index.exists():
            names = json.loads(index.read_text()).get('weight_map', {})
            if any('ptq_layers' in name or 'exit_modules' in name for name in names):
                raise ValueError('Native checkpoint index contains RiverEdge sidecar weights')
    return config


def model_options(switches, native_model, riveredge_model, exit_layer=3):
    switches.validate()
    options = {'model': riveredge_model if switches.ptq_tail else native_model}
    if switches.ptq_tail:
        options['hf_overrides'] = {
            'architectures': [switches.architecture],
            'river_edge_mode': 'full_fp' if switches.online_routing else 'static_ptq_tail',
            'river_edge_exit_layer': exit_layer,
        }
        # Static phase detection cannot safely quantize mixed prefill/decode.
        options.update(enable_prefix_caching=False, enable_chunked_prefill=False)
    if switches.online_routing:
        options.update(enforce_eager=True, async_scheduling=False)
    return options


class ConfiguredEngine:
    def __init__(self, llm, switches, runtime=None, head=None):
        self.llm, self.switches, self.runtime, self.head = llm, switches, runtime, head

    def close(self):
        # Only remove mutations installed by this wrapper.
        if self.runtime is not None:
            if self.head is not None:
                self.runtime.model.lm_head.quant_method = self.head.original
            self.runtime.close()
        self.llm.llm_engine.engine_core.shutdown()


def create_engine(switches, *, native_model, riveredge_model, exit_layer=3,
                  low_batch=8, high_batch=16, capture_sizes=(1, 2, 4, 8, 16), **kwargs):
    """Create a fresh engine; changing switches requires a new engine/process.

    In native mode this calls LLM(model=native_model, **kwargs) unchanged, after
    checking checkpoint metadata. No runtime hook or head patch is installed.
    """
    options = model_options(switches, native_model, riveredge_model, exit_layer)
    if {'model', 'hf_overrides', 'quantization', 'load_format'} & kwargs.keys():
        raise ValueError('Model/quantization overrides must be controlled by switches')
    config = validate_checkpoint(options['model'], switches)
    if switches.ptq_tail and not (
            config.get('river_edge_first_ptq_layer_1_indexed', 2) - 1
            <= exit_layer < config['num_hidden_layers']):
        raise ValueError('Exit layer is outside the checkpoint PTQ coverage')
    policy = SwitchRoutePolicy(switches, low_batch, high_batch) if switches.online_routing else None
    for key, value in options.items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(f'{key} conflicts with the selected RiverEdge execution path')
    if switches.online_routing and any(kwargs.get(key, 1) != 1 for key in
                                       ('tensor_parallel_size', 'pipeline_parallel_size')):
        raise ValueError('Online reference requires TP=PP=1')
    if switches.online_routing and kwargs.get('speculative_config') is not None:
        raise ValueError('Online reference does not support speculative decoding')
    if switches.decode_cudagraph:
        if kwargs.get('enforce_eager') or 'compilation_config' in kwargs:
            raise ValueError('decode_cudagraph conflicts with eager/custom compilation_config')
        from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
        if not capture_sizes or any(size < 1 for size in capture_sizes):
            raise ValueError('Graph capture sizes must be positive')
        options['compilation_config'] = CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=sorted(set(capture_sizes)), cudagraph_num_of_warmups=1)
    elif switches.ptq_tail and not switches.online_routing:
        # The experimental PTQ Inductor path was reverted. Graph off means eager
        # for PTQ only; the native path still keeps upstream defaults.
        if kwargs.get('enforce_eager') is False or 'compilation_config' in kwargs:
            raise ValueError('Static PTQ without decode_cudagraph requires eager')
        options['enforce_eager'] = True
    from vllm import LLM
    if switches.ptq_tail:
        from .plugin import register
        register()
    llm = LLM(**(kwargs | options))
    configured = ConfiguredEngine(llm, switches)
    try:
        if switches.online_routing:
            from .online_runtime import OnlineRuntime
            configured.runtime = OnlineRuntime(llm, policy, grouped=switches.grouped_scheduling)
            if switches.int4_head:
                from .head_ablation import Int4HeadCandidate
                configured.head = Int4HeadCandidate(configured.runtime.model)
                configured.runtime.model.lm_head.quant_method = configured.head
        return configured
    except Exception:
        configured.close()
        raise
