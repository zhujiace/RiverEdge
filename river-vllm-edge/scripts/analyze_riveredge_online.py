#!/usr/bin/env python3
"""Check phase contracts and paired outputs from real online experiments."""

import argparse
import json
import math
import re
from itertools import product
from pathlib import Path


def case_key(case):
    return (case['exit_layer'], case['policy'], case['scheduler'], case['repeat'],
            case.get('head', 'fp'), case.get('workload', 'legacy'))


def expected_case_keys(data):
    config = data.get('config', {})
    fields = ('exit_layers', 'policies', 'schedulers', 'repeats', 'head_variants')
    if not all(field in config for field in fields):
        return None
    patterns = (['trace'] if config.get('trace') else
                (config.get('arrival_patterns') or config.get('arrivals', 'bursty')).split(','))
    keys = set(product(map(int, config['exit_layers'].split(',')),
                       config['policies'].split(','), config['schedulers'].split(','),
                       range(config['repeats']), config['head_variants'].split(','), patterns))
    if config.get('latency_probe'):
        keys.update((3, 'ptq', 'native', repeat, head, 'latency_probe')
                    for repeat in range(5) for head in config['head_variants'].split(','))
    return keys


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_delivery(row):
    stamps = row.get('token_times_s', [])
    arrival, finished = row.get('arrival_s'), row.get('finished_s')
    return (finite(arrival) and finite(finished) and bool(stamps)
            and len(stamps) == len(row.get('token_ids', []))
            and all(finite(stamp) for stamp in stamps)
            and arrival <= stamps[0] <= stamps[-1] <= finished
            and all(a <= b for a, b in zip(stamps, stamps[1:])))


def analyze(data):
    checks, table, pairs, pairing_errors, warnings = [], [], [], [], []
    cases = data["cases"]
    keys = [case_key(case) for case in cases]
    expected_keys = expected_case_keys(data)
    checks.append({'case': 'run', 'check': 'nonempty_unique_cases',
                   'pass': bool(cases) and len(keys) == len(set(keys))})
    if expected_keys is not None:
        checks.append({'case': 'run', 'check': 'expected_case_matrix',
                       'pass': set(keys) == expected_keys})
    else:
        warnings.append('No complete run configuration: planned case coverage cannot be verified.')
    expected_pairs = 0
    for case in cases:
        name = f'k{case["exit_layer"]}/{case["policy"]}/{case["scheduler"]}/{case["repeat"]}/{case.get("head", "fp")}/{case.get("workload", "legacy")}'
        requests = case['requests']
        ids = [row.get('request_id') for row in requests]
        unique_ids = all(isinstance(rid, str) and rid for rid in ids) and len(set(ids)) == len(ids)
        expected_ids = case.get('expected_request_ids')
        config = data.get('config', {})
        source = 'case_manifest'
        if expected_ids is None and case.get('workload') == 'latency_probe':
            expected_ids, source = ['latency-probe'], 'latency_probe_protocol'
        elif expected_ids is None and not config.get('trace') and 'requests' in config:
            expected_ids = [f'request-{i}' for i in range(config['requests'])]
            source = 'synthetic_config'
        if expected_ids is None:
            source = 'legacy_summary_only'
            warnings.append(f'{name}: no request manifest; trace completeness checked against summary only.')
        complete = (bool(requests) and unique_ids
                    and len(requests) == case['summary'].get('requests'))
        if expected_ids is not None:
            complete = (complete and len(expected_ids) == len(set(expected_ids))
                        and set(ids) == set(expected_ids))
        checks.append({'case': name, 'check': 'request_set_complete',
                       'pass': complete, 'expected_source': source})
        checks.append({'case': name, 'check': 'delivery_records_valid',
                       'pass': bool(requests) and all(valid_delivery(r) for r in requests)})
        checks.append({'case': name, 'check': 'output_token_count_matches',
                       'pass': sum(len(r.get('token_ids', [])) for r in requests)
                       == case['summary'].get('output_tokens')})
        # vLLM appends an eight-hex suffix to external request IDs. Require a
        # unique exact/suffixed match; never strip arbitrary request suffixes.
        observed, mappings, trace_valid = set(), {}, bool(case['steps'])
        for step in case['steps']:
            scheduled = step.get('scheduled_tokens', {})
            trace_valid &= set(scheduled) == set(step['routes'])
            for internal_id, route in step['routes'].items():
                matches = [rid for rid in ids if isinstance(rid, str) and
                           (internal_id == rid or re.fullmatch(re.escape(rid) + r'-[0-9a-f]{8}', internal_id))]
                trace_valid &= (len(matches) == 1 and route.get('phase') in {'prefill', 'decode'}
                                and route.get('route') in {'fp', 'ptq'}
                                and isinstance(route.get('tokens'), int) and route['tokens'] > 0
                                and route['tokens'] == scheduled.get(internal_id))
                if len(matches) == 1:
                    observed.add(matches[0])
                    mappings.setdefault(matches[0], set()).add(internal_id)
        checks.append({'case': name, 'check': 'scheduler_trace_complete',
                       'pass': bool(requests) and trace_valid and observed == set(ids)
                       and all(len(values) == 1 for values in mappings.values())})
        routes = [r for s in case["steps"] for r in s["routes"].values()]
        checks.append({"case": name, "check": "prefill_always_fp",
                       "pass": bool(routes) and all(r["route"] == "fp" for r in routes if r["phase"] == "prefill")})
        checks.append({"case": name, "check": "all_requests_finished",
                       "pass": bool(requests) and all(finite(r.get("finished_s")) for r in requests)})
        if case["scheduler"] == "grouped":
            checks.append({"case": name, "check": "homogeneous_phase_route_steps",
                           "pass": all(len({(r["phase"], r["route"]) for r in s["routes"].values()}) <= 1
                                       for s in case["steps"])})
        summary = case["summary"]
        table.append({"case": name, "output_tps": summary["output_tps"],
                      "ttft_s": summary["ttft_s"], "tpot_s": summary["tpot_s"],
                      "goodput_requests_s": summary["slo_requests_per_s"],
                      "quality": summary["quality"],
                      "rail_joules_token": case["energy"]["sampled_joules_per_output_token"]})
        if case["policy"] != "mixed":
            continue
        for request in case["requests"]:
            expected_pairs += 1
            route = "ptq" if request["allow_ptq"] else "fp"
            baselines = [c for c in cases if c["policy"] == route
                             and c["scheduler"] == "native"
                             and c["exit_layer"] == case["exit_layer"]
                             and c.get("head", "fp") == case.get("head", "fp")
                             and c.get("workload") == case.get("workload")
                             and c["repeat"] == case["repeat"]]
            references = [r for c in baselines for r in c['requests']
                          if r['request_id'] == request['request_id']]
            if len(baselines) != 1 or len(references) != 1:
                pairing_errors.append({'case': name, 'request_id': request['request_id'],
                                       'reason': 'missing_or_ambiguous_baseline'})
                continue
            reference = references[0]
            protocol = ('prompt', 'arrival_s', 'max_tokens', 'allow_ptq', 'answer', 'answer_regex', 'task')
            if (any(reference.get(field) != request.get(field) for field in protocol)
                    or baselines[0].get('force_length') != case.get('force_length')):
                pairing_errors.append({'case': name, 'request_id': request['request_id'],
                                       'reason': 'baseline_protocol_mismatch'})
                continue
            pairs.append({"case": name, "baseline_case": '/'.join(map(str, case_key(baselines[0]))),
                              "request_id": request["request_id"], "scheduler": case["scheduler"],
                              "exit_layer": case["exit_layer"], "route": route,
                              "token_equal": reference["token_ids"] == request["token_ids"]})
    structural_pass = bool(checks) and all(c['pass'] for c in checks)
    coverage_pass = not pairing_errors and len(pairs) == expected_pairs
    unequal = sum(not pair['token_equal'] for pair in pairs)
    status = ('incomplete' if not structural_pass or not coverage_pass else
              'not_applicable' if not expected_pairs else 'mismatch' if unequal else 'exact_match')
    return {"schema_version": 3, "checks": checks,
            "all_contracts_pass": structural_pass,
            "structural_checks_pass": structural_pass,
            "analysis_pass": structural_pass and coverage_pass,
            "warnings": warnings,
            "output_comparison": {'status': status, 'expected_pairs': expected_pairs,
                                  'compared_pairs': len(pairs), 'unequal_pairs': unequal,
                                  'pairing_complete': coverage_pass, 'errors': pairing_errors},
            "mixed_static_pairs": pairs, "table": table,
            "limits": ["Shared device; eager experimental runtime, not production latency.",
                       "Externally assigned PTQ eligibility is not a confidence gate.",
                       "Quality applies only to the supplied trace and scoring protocol, not a full benchmark.",
                       "Power is the named rail including other workloads, not isolated board energy."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument('--require-token-equality', action='store_true',
                        help='Fail unless a complete mixed/static comparison has exact token equality')
    args = parser.parse_args()
    result = analyze(json.loads(Path(args.input).read_text()))
    output = Path(args.output)
    if output.resolve() == Path(args.input).resolve():
        parser.error('Output must not overwrite the input experiment')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if (not result["analysis_pass"] or (args.require_token_equality
            and result['output_comparison']['status'] != 'exact_match')):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
