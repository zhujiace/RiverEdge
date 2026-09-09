#!/usr/bin/env python3
"""Compare paired GSM8K outputs from full-FP and static-PTQ runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-samples", type=Path, required=True)
    parser.add_argument("--ptq-samples", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["dataset_index"])
            if index in rows:
                raise ValueError(f"Duplicate dataset_index={index} in {path}")
            rows[index] = row
    return rows


def percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_mcnemar_p(fp_only: int, ptq_only: int) -> float:
    discordant = fp_only + ptq_only
    if discordant == 0:
        return 1.0
    lower = min(fp_only, ptq_only)
    one_tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * one_tail)


def correctness_pair(
    fp_rows: dict[int, dict[str, Any]],
    ptq_rows: dict[int, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    both_correct = fp_only = ptq_only = both_wrong = 0
    fp_only_indices = []
    ptq_only_indices = []
    for index in sorted(fp_rows):
        fp_correct = bool(fp_rows[index][field])
        ptq_correct = bool(ptq_rows[index][field])
        if fp_correct and ptq_correct:
            both_correct += 1
        elif fp_correct:
            fp_only += 1
            fp_only_indices.append(index)
        elif ptq_correct:
            ptq_only += 1
            ptq_only_indices.append(index)
        else:
            both_wrong += 1
    return {
        "both_correct": both_correct,
        "fp_only_correct": fp_only,
        "ptq_only_correct": ptq_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": exact_mcnemar_p(fp_only, ptq_only),
        "fp_only_indices": fp_only_indices,
        "ptq_only_indices": ptq_only_indices,
    }


def main() -> None:
    args = parse_args()
    fp_rows = load_jsonl(args.fp_samples)
    ptq_rows = load_jsonl(args.ptq_samples)
    if set(fp_rows) != set(ptq_rows):
        raise ValueError("FP and PTQ files do not contain the same dataset indices")

    indices = sorted(fp_rows)
    token_deltas = [
        int(ptq_rows[index]["output_tokens"]) - int(fp_rows[index]["output_tokens"])
        for index in indices
    ]
    fp_cap = {
        index for index in indices if fp_rows[index]["finish_reason"] == "length"
    }
    ptq_cap = {
        index for index in indices if ptq_rows[index]["finish_reason"] == "length"
    }
    strict_prediction_agreement = sum(
        fp_rows[index]["strict_prediction"] == ptq_rows[index]["strict_prediction"]
        for index in indices
    )
    exact_text_agreement = sum(
        fp_rows[index]["generated_text"] == ptq_rows[index]["generated_text"]
        for index in indices
    )

    result = {
        "num_samples": len(indices),
        "strict_correctness": correctness_pair(
            fp_rows, ptq_rows, "strict_correct"
        ),
        "flexible_correctness": correctness_pair(
            fp_rows, ptq_rows, "flexible_correct"
        ),
        "strict_prediction_agreement": strict_prediction_agreement,
        "strict_prediction_agreement_rate": strict_prediction_agreement / len(indices),
        "exact_generated_text_agreement": exact_text_agreement,
        "exact_generated_text_agreement_rate": exact_text_agreement / len(indices),
        "output_token_delta_ptq_minus_fp": {
            "total": sum(token_deltas),
            "mean": sum(token_deltas) / len(token_deltas),
            "min": min(token_deltas),
            "p50": percentile(token_deltas, 0.50),
            "p90": percentile(token_deltas, 0.90),
            "max": max(token_deltas),
            "ptq_shorter_count": sum(delta < 0 for delta in token_deltas),
            "equal_count": sum(delta == 0 for delta in token_deltas),
            "ptq_longer_count": sum(delta > 0 for delta in token_deltas),
        },
        "length_cap": {
            "fp_indices": sorted(fp_cap),
            "ptq_indices": sorted(ptq_cap),
            "both": sorted(fp_cap.intersection(ptq_cap)),
            "fp_only": sorted(fp_cap.difference(ptq_cap)),
            "ptq_only": sorted(ptq_cap.difference(fp_cap)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
