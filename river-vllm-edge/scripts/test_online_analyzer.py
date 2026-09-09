"""CPU-only regression tests for experiment completeness and paired outputs."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analyze_riveredge_online import analyze


def fixture():
    row = dict(request_id='request-0', prompt='test', arrival_s=0, max_tokens=2,
               allow_ptq=True, token_ids=[1, 2], token_times_s=[0.1, 0.2], finished_s=0.2)
    cases = []
    for policy in ('fp', 'ptq', 'mixed'):
        cases.append(dict(exit_layer=3, policy=policy, scheduler='native', repeat=0,
                          head='fp', workload='bursty', force_length=True,
                          expected_request_ids=['request-0'], requests=[copy.deepcopy(row)],
                          summary=dict(requests=1, output_tokens=2, output_tps=10,
                                       ttft_s={}, tpot_s={}, slo_requests_per_s=5, quality={}),
                          energy=dict(sampled_joules_per_output_token=None),
                          steps=[dict(scheduled_tokens={'request-0-1234abcd': 2},
                                      routes={'request-0-1234abcd': dict(route='fp', phase='prefill', tokens=2)})]))
    return dict(cases=cases, config=dict(exit_layers='3', policies='fp,ptq,mixed',
                                        schedulers='native', repeats=1, head_variants='fp',
                                        arrivals='bursty', requests=1))


class AnalyzerTests(unittest.TestCase):
    def test_valid(self):
        result = analyze(fixture())
        self.assertTrue(result['analysis_pass'])
        self.assertEqual(result['output_comparison']['status'], 'exact_match')

    def test_empty_cases(self):
        self.assertFalse(analyze({'cases': []})['analysis_pass'])

    def test_empty_and_duplicate_requests(self):
        for duplicate in (False, True):
            data = fixture()
            rows = data['cases'][0]['requests']
            rows[:] = rows * 2 if duplicate else []
            self.assertFalse(analyze(data)['analysis_pass'])

    def test_manifest_catches_consistently_removed_request(self):
        data = fixture()
        data['cases'][0]['expected_request_ids'].append('request-1')
        self.assertFalse(analyze(data)['analysis_pass'])

    def test_missing_and_duplicate_cases(self):
        for duplicate in (False, True):
            data = fixture()
            if duplicate:
                data['cases'].append(copy.deepcopy(data['cases'][0]))
            else:
                data['cases'].pop(0)
            self.assertFalse(analyze(data)['analysis_pass'])

    def test_missing_baseline_is_reported(self):
        data = fixture()
        data['cases'].pop(1)
        result = analyze(data)
        self.assertFalse(result['analysis_pass'])
        self.assertEqual(result['output_comparison']['errors'][0]['reason'],
                         'missing_or_ambiguous_baseline')

    def test_protocol_mismatch(self):
        data = fixture()
        data['cases'][1]['requests'][0]['prompt'] = 'different'
        self.assertFalse(analyze(data)['analysis_pass'])

    def test_output_mismatch_is_separate(self):
        data = fixture()
        data['cases'][2]['requests'][0]['token_ids'][1] = 3
        result = analyze(data)
        self.assertTrue(result['analysis_pass'])
        self.assertEqual(result['output_comparison']['status'], 'mismatch')
        self.assertEqual(result['output_comparison']['unequal_pairs'], 1)

    def test_delivery_and_summary_corruption(self):
        for change in ('finish', 'timestamps', 'count'):
            data = fixture()
            case = data['cases'][0]
            if change == 'finish':
                case['requests'][0]['finished_s'] = None
            elif change == 'timestamps':
                case['requests'][0]['token_times_s'] = [0.2, 0.1]
            else:
                case['summary']['output_tokens'] = 99
            self.assertFalse(analyze(data)['analysis_pass'])

    def test_trace_mapping_and_missing_step_routes(self):
        for change in ('unknown', 'empty'):
            data = fixture()
            step = data['cases'][0]['steps'][0]
            if change == 'empty':
                step['routes'] = {}
            else:
                step['routes']['unknown'] = step['routes'].pop('request-0-1234abcd')
            self.assertFalse(analyze(data)['analysis_pass'])

    def test_legacy_trace_reports_weaker_evidence(self):
        data = fixture()
        data['config'] = {'trace': 'unavailable.jsonl'}
        for case in data['cases']:
            del case['expected_request_ids']
        result = analyze(data)
        self.assertTrue(result['analysis_pass'])
        self.assertTrue(any('summary only' in warning for warning in result['warnings']))

    def test_cli_strict_mode_and_input_protection(self):
        script = Path(__file__).with_name('analyze_riveredge_online.py')
        data = fixture()
        data['cases'][2]['requests'][0]['token_ids'][1] = 3
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.json'
            output = Path(directory) / 'output.json'
            source.write_text(json.dumps(data))
            command = [sys.executable, '-B', str(script), str(source), '--output', str(output)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(command + ['--require-token-equality'],
                                            capture_output=True).returncode, 1)
            self.assertEqual(subprocess.run(command[:-1] + [str(source)],
                                            capture_output=True).returncode, 2)
            self.assertEqual(json.loads(source.read_text()), data)


if __name__ == '__main__':
    unittest.main()
