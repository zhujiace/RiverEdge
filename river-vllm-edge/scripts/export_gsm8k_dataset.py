#!/usr/bin/env python3
"""Export a cached Hugging Face GSM8K Arrow split to portable JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from datasets import Dataset

    args = parse_args()
    dataset = Dataset.from_file(str(args.arrow_path))
    required_columns = {"question", "answer"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"GSM8K Arrow file is missing columns: {sorted(missing)}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            record = {
                "dataset_index": index,
                "question": row["question"],
                "answer": row["answer"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "arrow_path": str(args.arrow_path),
                "output_jsonl": str(args.output_jsonl),
                "rows": len(dataset),
                "columns": dataset.column_names,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
