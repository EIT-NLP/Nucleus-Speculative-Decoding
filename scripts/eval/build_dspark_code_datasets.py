
"""Tag native MBPP/HumanEval rows without changing their DSpark prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.eval.dspark_code_contract import (
    HUMANEVAL_POSTPROCESSOR,
    MBPP_POSTPROCESSOR,
    PROFILE_NAME,
)


PROFILE = {
    "name": PROFILE_NAME,
    "prompt_contract": "original_dspark",
    "prompt_preserved": True,
    "mbpp_postprocessor": MBPP_POSTPROCESSOR,
    "humaneval_postprocessor": HUMANEVAL_POSTPROCESSOR,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("eval_datasets_scored"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eval_datasets_dspark"),
    )
    parser.add_argument("--expected-mbpp", type=int, default=257)
    parser.add_argument("--expected-humaneval", type=int, default=164)
    parser.add_argument(
        "--tasks",
        default="mbpp,humaneval",
        help="Comma-separated code datasets to tag.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object.")
            rows.append(row)
    return rows


def tag_row(row: dict[str, Any], task: str) -> dict[str, Any]:
    output = dict(row)
    output["evaluation_profile"] = dict(PROFILE, task=task)
    for prompt_field in ("turns", "messages", "assistant_prefix"):
        if output.get(prompt_field) != row.get(prompt_field):
            raise AssertionError(f"DSpark builder changed prompt field {prompt_field}.")
    return output


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode("utf-8"))
    os.replace(temporary, path)
    return digest.hexdigest()


def validate_rows(name: str, rows: list[dict[str, Any]]) -> None:
    source_ids = [row.get("source_id") for row in rows]
    if any(source_id is None for source_id in source_ids):
        raise ValueError(f"{name} contains a row without source_id.")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{name} contains duplicate source_id values.")
    for row in rows:
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns or not all(
            isinstance(turn, str) and turn for turn in turns
        ):
            raise ValueError(f"{name} contains an invalid DSpark turns field.")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    sources = {
        "mbpp": source_root / "mbpp.jsonl",
        "humaneval": source_root / "humaneval.jsonl",
    }
    expected = {
        "mbpp": args.expected_mbpp,
        "humaneval": args.expected_humaneval,
    }
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(tasks) - set(sources))
    if unknown:
        raise ValueError(f"Unknown code datasets: {unknown}")

    datasets: dict[str, dict[str, Any]] = {}
    for task in tasks:
        source_path = sources[task]
        source_rows = read_jsonl(source_path)
        rows = [tag_row(row, task) for row in source_rows]
        if len(rows) != expected[task]:
            raise ValueError(
                f"Expected {expected[task]} {task} rows, found {len(rows)}."
            )
        validate_rows(task, rows)
        for source_row, output_row in zip(source_rows, rows):
            for field in ("turns", "messages", "assistant_prefix"):
                if source_row.get(field) != output_row.get(field):
                    raise AssertionError(
                        f"{task} prompt field {field} was not preserved."
                    )
        output_path = output_root / f"{task}.jsonl"
        digest = write_jsonl_atomic(output_path, rows)
        datasets[task] = {
            "source": str(source_path),
            "output": str(output_path),
            "rows": len(rows),
            "sha256": digest,
            "prompt_preserved": True,
        }

    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "source_root": str(source_root),
        "datasets": datasets,
    }
    manifest_path = output_root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
