#!/usr/bin/env python3
"""Snapshot small real MMLU test subsets without a datasets dependency.

Uses the Hugging Face dataset viewer rows API; saves response hashes and the
exact prompts/labels. This is a zero-shot generation sanity subset, not the
official full five-shot MMLU evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-task", type=int, default=16)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() or not 1 <= args.per_task <= 100:
        parser.error("Choose a new output path and 1-100 rows per task")
    rows, sources = [], []
    for task in ("college_mathematics", "high_school_computer_science", "world_religions"):
        query = urlencode({"dataset": "cais/mmlu", "config": task, "split": "test",
                           "offset": 0, "length": args.per_task})
        url = "https://datasets-server.huggingface.co/rows?" + query
        with urlopen(url, timeout=30) as response:
            raw = response.read()
        payload = json.loads(raw)
        if len(payload["rows"]) != args.per_task:
            raise RuntimeError(f"Incomplete dataset response for {task}")
        sources.append({"task": task, "url": url,
                        "response_sha256": hashlib.sha256(raw).hexdigest()})
        for item in payload["rows"]:
            row = item["row"]
            choices = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(row["choices"]))
            rows.append({"prompt": f'{row["question"]}\n{choices}\nRespond with only the letter A, B, C, or D.',
                         "answer": chr(65 + row["answer"]), "task": task,
                         "allow_ptq": len(rows) % 2 == 0, "max_tokens": 8,
                         "arrival_s": len(rows) * 0.05,
                         "source_row": item["row_idx"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    path.with_suffix(".sources.json").write_text(json.dumps({"downloaded_unix_s": time.time(),
                                                           "sources": sources}, indent=2))
    print(f"Saved {len(rows)} labeled requests to {path}")


if __name__ == "__main__":
    main()
