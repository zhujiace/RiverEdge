#!/usr/bin/env python3
"""CPU tests for route hysteresis and delivery/SLO metric definitions."""

import unittest

from benchmark_riveredge_online import PowerSampler, score_answer, summarize
from river_vllm_ext.online_runtime import RoutePolicy


class OnlineContractTests(unittest.TestCase):
    def test_final_answer_is_not_a_letter_inside_reasoning(self):
        row = {"answer": "B", "answer_regex": r"(?i)Final answer:\s*([ABCD])\b"}
        self.assertEqual(score_answer("Choice B looks plausible", row), (None, False))
        self.assertEqual(score_answer("Final answer: A\nCorrection. Final answer: B", row), ("B", True))
        self.assertEqual(score_answer("Final answer: A", row), ("A", False))

    def test_energy_clips_to_request_window(self):
        sampler = PowerSampler("NONEXISTENT_TEST_RAIL")
        sampler.samples = [(0, 10), (2, 20)]
        result = sampler.finish(0.5, 1.5, 3)
        self.assertAlmostEqual(result["sampled_joules"], 15)
        self.assertAlmostEqual(result["sampled_joules_per_output_token"], 5)
        self.assertEqual(result["sample_coverage_fraction"], 1)

    def test_conservative_mixed_fallback(self):
        policy = RoutePolicy("conservative")
        eligible = {"riveredge_allow_ptq": True}
        policy.update(4, 2)
        self.assertEqual(policy.select(eligible), "fp")
        policy.update(4, 4)
        self.assertEqual(policy.select(eligible), "ptq")

    def test_hysteresis_and_ineligible_fallback(self):
        policy = RoutePolicy("batch", low=2, high=4)
        eligible = {"riveredge_allow_ptq": True}
        for batch, expected in [(1, "ptq"), (3, "ptq"), (4, "fp"),
                                (3, "fp"), (2, "ptq")]:
            policy.update(batch)
            self.assertEqual(policy.select(eligible), expected)
            self.assertEqual(policy.select({}), "fp")

    def test_strict_tpot_excludes_ttft_and_includes_wait(self):
        rows = [{"arrival_s": 1, "token_times_s": [2, 2.1, 2.4],
                 "token_ids": [1, 2, 3], "finished_s": 2.5,
                 "admission_lag_s": 0.2}]
        result = summarize(rows, 4, 0.5, 0.3)
        self.assertAlmostEqual(rows[0]["tpot_s"], 0.2)
        self.assertEqual(rows[0]["ttft_s"], 1)
        self.assertEqual(result["slo_requests_per_s"], 0)
        self.assertEqual(result["output_tps"], 0.75)

    def test_single_token_and_quality_goodput(self):
        rows = [{"arrival_s": 0, "token_times_s": [0.2], "token_ids": [1],
                 "finished_s": 0.2, "admission_lag_s": 0,
                 "correct": True, "task": "sanity"}]
        result = summarize(rows, 1, 0.5, 0.1)
        self.assertIsNone(rows[0]["tpot_s"])
        self.assertEqual(result["quality_slo_requests_per_s"], 1)


if __name__ == "__main__":
    unittest.main()
