"""CPU regressions for native identity, switch dependencies and policy isolation."""

from dataclasses import fields
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from river_vllm_ext.optimization_switches import (
    OptimizationSwitches, SwitchRoutePolicy, create_engine, model_options, validate_checkpoint,
)


class SwitchTests(unittest.TestCase):
    def test_all_off_options_are_stock(self):
        switches = OptimizationSwitches()
        self.assertTrue(switches.all_off)
        self.assertEqual(switches.architecture, 'LlamaForCausalLM')
        self.assertEqual(model_options(switches, 'native', 'dual'), {'model': 'native'})

    def test_each_switch_is_explicitly_off_by_default(self):
        self.assertTrue(all(getattr(OptimizationSwitches(), field.name) is False
                            for field in fields(OptimizationSwitches)))

    def test_invalid_dependencies(self):
        for name in ('online_routing', 'grouped_scheduling', 'batch_aware_routing',
                     'conservative_fallback', 'int4_head'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                OptimizationSwitches(**{name: True}).validate()
        with self.assertRaises(ValueError):
            OptimizationSwitches(ptq_tail=True, online_routing=True, decode_cudagraph=True).validate()

    def test_nonboolean_switch_rejected(self):
        with self.assertRaises(ValueError):
            OptimizationSwitches(ptq_tail='false').validate()

    def test_static_and_online_architecture(self):
        static = model_options(OptimizationSwitches(ptq_tail=True), 'native', 'dual')
        online = model_options(OptimizationSwitches(ptq_tail=True, online_routing=True), 'native', 'dual')
        self.assertEqual(static['hf_overrides']['river_edge_mode'], 'static_ptq_tail')
        self.assertEqual(online['hf_overrides']['river_edge_mode'], 'full_fp')
        self.assertEqual(online['hf_overrides']['architectures'], ['RiverEdgeOnlineForCausalLM'])
        self.assertTrue(online['enforce_eager'])
        self.assertFalse(online['async_scheduling'])

    def test_batch_switch_does_not_leak_into_plain_routing(self):
        policy = SwitchRoutePolicy(OptimizationSwitches(ptq_tail=True, online_routing=True))
        policy.update(100, 50)
        self.assertEqual(policy.select({'riveredge_allow_ptq': True}), 'ptq')
        self.assertEqual(policy.select({}), 'fp')

    def test_hysteresis(self):
        policy = SwitchRoutePolicy(OptimizationSwitches(ptq_tail=True, online_routing=True,
                                                        batch_aware_routing=True), low=2, high=4)
        for size, expected in [(1, 'ptq'), (4, 'fp'), (3, 'fp'), (2, 'ptq')]:
            policy.update(size, size)
            self.assertEqual(policy.select({'riveredge_allow_ptq': True}), expected)

    def test_fallback_without_hysteresis(self):
        policy = SwitchRoutePolicy(OptimizationSwitches(ptq_tail=True, online_routing=True,
                                                        conservative_fallback=True))
        policy.update(100, 50)
        self.assertEqual(policy.select({'riveredge_allow_ptq': True}), 'fp')
        policy.update(100, 100)
        self.assertEqual(policy.select({'riveredge_allow_ptq': True}), 'ptq')

    def test_native_rejects_custom_or_quantized_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            for config in ({'architectures': ['RiverEdgeUnifiedForCausalLM']},
                           {'architectures': ['LlamaForCausalLM'], 'quantization_config': {'quant_method': 'torchao'}},
                           {'architectures': ['LlamaForCausalLM'], 'river_edge_exit_layer': 3}):
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    validate_checkpoint(directory, OptimizationSwitches())

    def test_native_rejects_sidecar_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps({'architectures': ['LlamaForCausalLM']}))
            (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'model.ptq_layers.1.weight': 'tail'}}))
            with self.assertRaises(ValueError):
                validate_checkpoint(directory, OptimizationSwitches())

    def test_all_off_calls_unmodified_llm_and_no_hooks(self):
        llm = Mock()
        constructor = Mock(return_value=llm)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps({'architectures': ['LlamaForCausalLM']}))
            with patch.dict(sys.modules, {'vllm': SimpleNamespace(LLM=constructor)}):
                engine = create_engine(OptimizationSwitches(), native_model=directory,
                                       riveredge_model='unused', max_num_seqs=2)
        constructor.assert_called_once_with(model=directory, max_num_seqs=2)
        self.assertIsNone(engine.runtime)
        self.assertIsNone(engine.head)
        engine.close()
        llm.llm_engine.engine_core.shutdown.assert_called_once()

    def test_native_rejects_hidden_model_override(self):
        with self.assertRaises(ValueError):
            create_engine(OptimizationSwitches(), native_model='unused', riveredge_model='unused',
                          hf_overrides={'architectures': ['RiverEdgeOnlineForCausalLM']})

    def test_online_optional_mutations_and_cleanup(self):
        for enabled in (False, True):
            with self.subTest(group_and_head=enabled), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / 'config.json').write_text(json.dumps({
                    'architectures': ['RiverEdgeUnifiedForCausalLM'], 'num_hidden_layers': 32}))
                llm, runtime = Mock(), Mock()
                original = object()
                runtime.model.lm_head.quant_method = original
                candidate = SimpleNamespace(original=original)
                constructor = Mock(return_value=llm)
                hook = Mock(return_value=runtime)
                head = Mock(return_value=candidate)
                modules = {
                    'vllm': SimpleNamespace(LLM=constructor),
                    'river_vllm_ext.online_runtime': SimpleNamespace(OnlineRuntime=hook),
                    'river_vllm_ext.head_ablation': SimpleNamespace(Int4HeadCandidate=head),
                }
                switches = OptimizationSwitches(ptq_tail=True, online_routing=True,
                                                grouped_scheduling=enabled, int4_head=enabled)
                with patch.dict(sys.modules, modules), patch('river_vllm_ext.plugin.register'):
                    engine = create_engine(switches, native_model='unused', riveredge_model=directory)
                self.assertEqual(hook.call_args.kwargs['grouped'], enabled)
                self.assertEqual(head.call_count, int(enabled))
                self.assertIs(runtime.model.lm_head.quant_method, candidate if enabled else original)
                engine.close()
                runtime.close.assert_called_once()
                self.assertIs(runtime.model.lm_head.quant_method, original)
                llm.llm_engine.engine_core.shutdown.assert_called_once()

    def test_static_graph_off_is_eager_without_online_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps({
                'architectures': ['RiverEdgeUnifiedForCausalLM'], 'num_hidden_layers': 32}))
            constructor = Mock()
            with patch.dict(sys.modules, {'vllm': SimpleNamespace(LLM=constructor)}), patch('river_vllm_ext.plugin.register'):
                engine = create_engine(OptimizationSwitches(ptq_tail=True),
                                       native_model='unused', riveredge_model=directory)
            self.assertTrue(constructor.call_args.kwargs['enforce_eager'])
            self.assertNotIn('compilation_config', constructor.call_args.kwargs)
            self.assertIsNone(engine.runtime)
            engine.close()


if __name__ == '__main__':
    unittest.main()
