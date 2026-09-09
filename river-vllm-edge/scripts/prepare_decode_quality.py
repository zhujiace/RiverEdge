#!/usr/bin/env python3
"""Make decode-sensitive questions from the saved MMLU snapshot."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-task", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or args.per_task < 1:
        parser.error("Choose a new output path and a positive per-task limit")
    raw = Path(args.source).read_bytes()
    counts, selected = Counter(), []
    for line in raw.decode().splitlines():
        row = json.loads(line)
        task = row["task"]
        if counts[task] >= args.per_task:
            continue
        suffix = "\nRespond with only the letter A, B, C, or D."
        if not row["prompt"].endswith(suffix):
            raise ValueError("Unexpected source prompt format")
        row["prompt"] = row["prompt"][:-len(suffix)] + (
            "\nFirst explain your reasoning in one or two short sentences "
            "(at most 45 words). Then write exactly 'Final answer: X', "
            "replacing X with A, B, C, or D. Do not give the answer before the reasoning.")
        row["max_tokens"] = 96
        row["answer_regex"] = r"(?i)Final answer:\s*([ABCD])\b"
        counts[task] += 1
        selected.append(row)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "source": args.source, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "counts": counts, "protocol": "zero-shot short reasoning; final-answer extraction; subset only"
    }, indent=2))
    print(json.dumps({"requests": len(selected), "counts": counts}))


if __name__ == "__main__":
    main()
