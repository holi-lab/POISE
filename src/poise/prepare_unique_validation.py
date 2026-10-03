"""Build POISE-only validation parquet files with one row per prompt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _prompt_key(prompt: object) -> str:
    return json.dumps(
        prompt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_unique_prompts(
    *,
    input_path: Path,
    output_path: Path,
    expected_rows: int,
) -> None:
    table = pq.read_table(input_path)
    if "prompt" not in table.column_names:
        raise ValueError(f"Validation parquet has no prompt column: {input_path}")

    selected_indices: list[int] = []
    seen: set[str] = set()
    for index, prompt in enumerate(table.column("prompt").to_pylist()):
        key = _prompt_key(prompt)
        if key in seen:
            continue
        seen.add(key)
        selected_indices.append(index)

    if len(selected_indices) != expected_rows:
        raise ValueError(
            f"Unexpected unique prompt count for {input_path}: "
            f"{len(selected_indices)} != {expected_rows}"
        )

    unique_table = table.take(pa.array(selected_indices, type=pa.int64()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        pq.write_table(unique_table, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(
        f"Wrote {len(unique_table)} unique prompts from {len(table)} rows: "
        f"{output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    write_unique_prompts(
        input_path=args.input,
        output_path=args.output,
        expected_rows=args.expected_rows,
    )


if __name__ == "__main__":
    main()
