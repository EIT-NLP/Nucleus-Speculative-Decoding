from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from scripts.eval.dspark_code_contract import (
    PROFILE_NAME as DSPARK_CODE_PROFILE,
    CodePreparationError,
    build_humaneval_program,
    build_mbpp_program,
    postprocessor_name,
    prepare_humaneval_program,
    prepare_mbpp_program,
)


MATH_TASKS = ("math500", "aime24", "aime25", "aime26")
MULTIPLE_CHOICE_TASKS = (
    "gpqa_diamond",
    "arc_challenge",
    "mmlu",
    "mmlu_pro",
    "commonsense_qa",
    "openbookqa",
    "piqa",
)
CODE_TASKS = ("humaneval", "mbpp")
OBJECTIVE_TASKS = ("gsm8k", *MATH_TASKS, *MULTIPLE_CHOICE_TASKS, *CODE_TASKS)
SUBJECTIVE_TASKS = ("mt-bench", "alpaca", "arena-hard-v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score rank-local prediction JSONL files written by eval.py. "
            "MATH-500/AIME use OpenCompass math verification. Native "
            "HumanEval/MBPP preserve the original DSpark prompts, apply "
            "DSpark-aware output extraction, and execute public tests only "
            "inside a bubblewrap sandbox."
        )
    )
    parser.add_argument(
        "prediction_root",
        type=Path,
        help="Run directory containing artifacts/step_*/<dataset>/predictions.*.jsonl.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(OBJECTIVE_TASKS + SUBJECTIVE_TASKS),
        help="Comma-separated dataset names to report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to <prediction_root>/quality_metrics.json.",
    )
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help=(
            "Run HumanEval/MBPP tests. Execution is refused unless bubblewrap "
            "is available; there is no unsafe fallback."
        ),
    )
    parser.add_argument("--code-workers", type=int, default=4)
    parser.add_argument(
        "--code-timeout",
        type=float,
        default=None,
        help=(
            "Backward-compatible override applied to both code tasks. Prefer "
            "--humaneval-timeout and --mbpp-timeout."
        ),
    )
    parser.add_argument("--humaneval-timeout", type=float, default=3.0)
    parser.add_argument("--mbpp-timeout", type=float, default=10.0)
    parser.add_argument("--bwrap", default="bwrap")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def _prediction_paths(root: Path, dataset_name: str) -> list[Path]:
    paths = [
        path
        for path in root.rglob("predictions.rank*.jsonl")
        if path.parent.name == dataset_name
    ]
    return sorted(paths)


def load_prediction_records(root: Path, dataset_name: str) -> list[dict[str, Any]]:
    paths = _prediction_paths(root, dataset_name)
    records: list[dict[str, Any]] = []
    for path in paths:
        manifest_path = path.with_name(
            path.name.replace("predictions.rank", "predictions_manifest.rank").replace(
                ".jsonl", ".json"
            )
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing completion manifest for {path}: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError(f"Prediction manifest is incomplete: {manifest_path}")
        path_records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("dataset") != dataset_name:
                    raise ValueError(
                        f"{path}:{line_number} has dataset={record.get('dataset')!r}, "
                        f"expected {dataset_name!r}."
                    )
                path_records.append(record)
        if len(path_records) != int(manifest["record_count"]):
            raise ValueError(
                f"{path} has {len(path_records)} rows but its manifest reports "
                f"{manifest['record_count']}."
            )
        records.extend(path_records)

    records.sort(key=lambda row: (int(row["sample_index"]), int(row["rank"])))
    sample_indices = [int(row["sample_index"]) for row in records]
    duplicates = [
        index for index, count in Counter(sample_indices).items() if count != 1
    ]
    if duplicates:
        raise ValueError(
            f"Dataset {dataset_name} has duplicate sample_index values: "
            f"{duplicates[:10]}"
        )
    return records


def _reference_field(record: dict[str, Any], field: str) -> Any:
    reference = record.get("reference")
    if isinstance(reference, dict) and field in reference:
        return reference[field]
    source = record.get("source")
    if isinstance(source, dict) and field in source:
        return source[field]
    raise KeyError(
        f"Prediction for dataset={record.get('dataset')!r}, "
        f"sample_index={record.get('sample_index')} has no reference field {field!r}."
    )


def score_math(records: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from opencompass.evaluator import MATHVerifyEvaluator
    except ImportError as exc:
        raise RuntimeError(
            "OpenCompass math evaluation is unavailable in this environment."
        ) from exc

    predictions = [str(record["output"]) for record in records]
    references = [str(_reference_field(record, "answer")) for record in records]
    result = MATHVerifyEvaluator().score(predictions, references)
    details = result.get("details", [])
    num_correct = sum(bool(detail.get("correct")) for detail in details)
    detail_rows = []
    for record, detail in zip(records, details):
        detail_rows.append(
            {
                "sample_index": int(record["sample_index"]),
                "source_id": record.get("source_id"),
                "correct": bool(detail.get("correct")),
                "parsed_prediction": detail.get("pred"),
                "parsed_reference": detail.get("answer"),
            }
        )
    accuracy_percent = float(result["accuracy"])
    return {
        "metric": "accuracy",
        "num_samples": len(records),
        "num_correct": num_correct,
        "accuracy": accuracy_percent / 100.0,
        "accuracy_percent": accuracy_percent,
        "evaluator": "opencompass.evaluator.MATHVerifyEvaluator",
        "details": detail_rows,
    }


_BOXED_ANSWER_RE = re.compile(
    r"\\boxed\s*\{\s*((?:[^{}]|\{[^{}]*\})+?)\s*\}", re.IGNORECASE
)
_ANSWER_LABEL_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?::|is)?\s*\(?([A-J0-9])\)?",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_choice_answer(text: str, valid_answers: set[str]) -> str | None:
    boxed = _BOXED_ANSWER_RE.findall(text)
    candidates = boxed or _ANSWER_LABEL_RE.findall(text)
    for candidate in reversed(candidates):
        normalized = re.sub(
            r"\\(?:text|mathrm|mathbf)\s*\{\s*([A-J0-9])\s*\}",
            r"\1",
            str(candidate),
            flags=re.IGNORECASE,
        )
        normalized = normalized.strip().strip("().: ").upper()
        if normalized in valid_answers:
            return normalized
    return None


def score_multiple_choice(records: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    num_correct = 0
    for record in records:
        reference = str(_reference_field(record, "answer")).strip().upper()
        choices = _reference_field(record, "choices")
        valid_answers = {str(label).strip().upper() for label in choices}
        prediction = extract_choice_answer(str(record["output"]), valid_answers)
        correct = prediction == reference
        num_correct += int(correct)
        details.append(
            {
                "sample_index": int(record["sample_index"]),
                "source_id": record.get("source_id"),
                "correct": correct,
                "parsed_prediction": prediction,
                "parsed_reference": reference,
            }
        )
    accuracy = num_correct / len(records)
    return {
        "metric": "accuracy",
        "num_samples": len(records),
        "num_correct": num_correct,
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "evaluator": "dspark_boxed_multiple_choice_v1",
        "details": details,
    }


def _normalize_number(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "").strip())
    except InvalidOperation:
        return None


def _extract_gsm8k_number(text: str, *, reference: bool) -> Decimal | None:
    if reference and "####" in text:
        return _normalize_number(text.rsplit("####", 1)[1])
    boxed = _BOXED_ANSWER_RE.findall(text)
    if boxed:
        boxed_numbers = _NUMBER_RE.findall(boxed[-1])
        if boxed_numbers:
            return _normalize_number(boxed_numbers[-1])
    numbers = _NUMBER_RE.findall(text)
    return _normalize_number(numbers[-1]) if numbers else None


def score_gsm8k(records: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    num_correct = 0
    for record in records:
        prediction = _extract_gsm8k_number(str(record["output"]), reference=False)
        reference = _extract_gsm8k_number(
            str(_reference_field(record, "answer")), reference=True
        )
        correct = prediction is not None and reference is not None and prediction == reference
        num_correct += int(correct)
        details.append(
            {
                "sample_index": int(record["sample_index"]),
                "source_id": record.get("source_id"),
                "correct": correct,
                "parsed_prediction": str(prediction) if prediction is not None else None,
                "parsed_reference": str(reference) if reference is not None else None,
            }
        )
    accuracy = num_correct / len(records)
    return {
        "metric": "accuracy",
        "num_samples": len(records),
        "num_correct": num_correct,
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "evaluator": "dspark_gsm8k_numeric_v1",
        "details": details,
    }


def _expected_dspark_prompt(
    record: dict[str, Any],
    dataset_name: str,
) -> str:
    source = record.get("source")
    reference = record.get("reference")
    if dataset_name == "mbpp":
        if not isinstance(source, dict) or not isinstance(source.get("prompt"), str):
            raise ValueError(
                "MBPP prediction has no source.prompt field for DSpark prompt "
                "validation."
            )
        return str(source["prompt"])
    if dataset_name == "humaneval":
        canonical_prompt = None
        if isinstance(reference, dict) and isinstance(reference.get("prompt"), str):
            canonical_prompt = reference["prompt"]
        elif isinstance(source, dict) and isinstance(source.get("prompt"), str):
            canonical_prompt = source["prompt"]
        if canonical_prompt is None:
            raise ValueError(
                "HumanEval prediction has no canonical prompt reference for "
                "DSpark prompt validation."
            )
        return (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{canonical_prompt}\n```"
        )
    raise ValueError(f"Unsupported DSpark code task: {dataset_name}")


def validate_dspark_code_profile(
    record: dict[str, Any],
    dataset_name: str,
) -> str:
    actual_prompt = record.get("prompt")
    expected_prompt = _expected_dspark_prompt(record, dataset_name)
    if actual_prompt != expected_prompt:
        raise ValueError(
            f"{dataset_name} prediction sample_index="
            f"{record.get('sample_index')} does not use the exact original "
            "DSpark prompt and cannot be scored under the DSpark functional "
            "evaluation contract."
        )

    profile = record.get("evaluation_profile")
    if profile is None:
        if "messages" in record or record.get("assistant_prefix"):
            raise ValueError(
                f"{dataset_name} legacy prediction sample_index="
                f"{record.get('sample_index')} contains a non-DSpark chat "
                "prompt contract."
            )
        return "legacy_prompt_inferred"
    if not isinstance(profile, dict):
        raise ValueError("evaluation_profile must be a dictionary.")
    if (
        profile.get("name") != DSPARK_CODE_PROFILE
        or profile.get("task") != dataset_name
        or profile.get("prompt_preserved") is not True
    ):
        raise ValueError(
            f"{dataset_name} prediction sample_index="
            f"{record.get('sample_index')} declares an incompatible code "
            "evaluation profile. Expected the original-DSpark-prompt profile "
            f"{DSPARK_CODE_PROFILE!r}."
        )
    if "messages" in record or record.get("assistant_prefix"):
        raise ValueError(
            f"{dataset_name} prediction sample_index="
            f"{record.get('sample_index')} altered the original DSpark prompt "
            "with role messages or an assistant prefix."
        )
    return "declared"


def _sandbox_command(
    *,
    bwrap_bin: str,
    python_bin: str,
    program_path: Path,
    timeout_seconds: float,
) -> list[str]:
    python_path = Path(python_bin).resolve()
    python_prefix = python_path.parent.parent
    command = [
        bwrap_bin,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--setenv",
        "PATH",
        f"{python_prefix}/bin:/usr/bin",
        "--setenv",
        "PYTHONHASHSEED",
        "0",
        "--setenv",
        "LC_ALL",
        "C.UTF-8",
    ]
    for system_path in (Path("/usr"), Path("/lib"), Path("/lib64")):
        if system_path.exists():
            command.extend(["--ro-bind", str(system_path), str(system_path)])

    prefix_parts = python_prefix.parts
    current = Path(prefix_parts[0])
    for part in prefix_parts[1:-1]:
        current /= part
        if current not in (Path("/usr"), Path("/lib"), Path("/lib64")):
            command.extend(["--dir", str(current)])
    if not str(python_prefix).startswith(("/usr", "/lib", "/lib64")):
        command.extend(["--ro-bind", str(python_prefix), str(python_prefix)])

    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/sandbox",
            "--ro-bind",
            str(program_path),
            "/sandbox/program.py",
            "--chdir",
            "/sandbox",
            "/usr/bin/prlimit",
            f"--cpu={max(1, int(timeout_seconds))}",
            "--as=2147483648",
            "--nproc=64",
            "--fsize=1048576",
            "--nofile=64",
            "--",
            str(python_path),
            "-I",
            "/sandbox/program.py",
        ]
    )
    return command


def run_sandboxed_program(
    *,
    program: str,
    bwrap_bin: str,
    python_bin: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="deepspec-code-eval-") as tmp_dir:
        program_path = Path(tmp_dir) / "program.py"
        program_path.write_text(program, encoding="utf-8")
        command = _sandbox_command(
            bwrap_bin=bwrap_bin,
            python_bin=python_bin,
            program_path=program_path,
            timeout_seconds=timeout_seconds,
        )
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds + 3.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stderr": "wall-clock timeout"}
    if completed.returncode == 0:
        return {"status": "pass", "stderr": ""}
    stderr = completed.stderr[-4000:]
    return {"status": "failed", "stderr": stderr}


def score_code(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
    bwrap_bin: str,
    python_bin: str,
    timeout_seconds: float,
    workers: int,
) -> dict[str, Any]:
    prepare_program = (
        prepare_humaneval_program
        if dataset_name == "humaneval"
        else prepare_mbpp_program
    )
    jobs: list[tuple[dict[str, Any], str]] = []
    results: dict[int, dict[str, Any]] = {}
    preparation_metadata: dict[int, dict[str, Any]] = {}
    for record in records:
        sample_index = int(record["sample_index"])
        try:
            prepared = prepare_program(record)
        except CodePreparationError as exc:
            results[sample_index] = {
                "status": exc.status,
                "stderr": str(exc),
            }
            preparation_metadata[sample_index] = exc.metadata
        else:
            jobs.append((record, prepared.program))
            preparation_metadata[sample_index] = prepared.metadata

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_record = {
            executor.submit(
                run_sandboxed_program,
                program=program,
                bwrap_bin=bwrap_bin,
                python_bin=python_bin,
                timeout_seconds=timeout_seconds,
            ): record
            for record, program in jobs
        }
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            results[int(record["sample_index"])] = future.result()

    details = []
    status_counts: Counter[str] = Counter()
    for record in records:
        sample_index = int(record["sample_index"])
        result = results[sample_index]
        status_counts[result["status"]] += 1
        details.append(
            {
                "sample_index": sample_index,
                "source_id": record.get("source_id"),
                "correct": result["status"] == "pass",
                "status": result["status"],
                "stderr": result["stderr"],
                "extraction": preparation_metadata.get(sample_index, {}),
            }
        )

    num_correct = status_counts["pass"]
    pass_at_1 = num_correct / len(records) if records else 0.0
    return {
        "metric": "pass@1",
        "num_samples": len(records),
        "num_correct": num_correct,
        "pass_at_1": pass_at_1,
        "pass_at_1_percent": 100.0 * pass_at_1,
        "status_counts": dict(status_counts),
        "sandbox": "bubblewrap+prlimit",
        "postprocessor": postprocessor_name(dataset_name),
        "prompt_contract": "original_dspark",
        "evaluation_profile": DSPARK_CODE_PROFILE,
        "timeout_seconds": timeout_seconds,
        "details": details,
    }


def subjective_status(dataset_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    reason = "A configured judge model is required; this task has no accuracy metric."
    if dataset_name == "mt-bench" and any(
        int(record.get("evaluated_turn_count", 1)) != len(record.get("turns", []))
        for record in records
    ):
        reason = (
            "A configured judge model is required, and these predictions contain "
            "only the first MT-Bench turn, so they are not standard MT-Bench scores."
        )
    return {
        "status": "predictions_only",
        "num_samples": len(records),
        "reason": reason,
    }


def main() -> None:
    args = parse_args()
    root = args.prediction_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {root}")
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    unknown = sorted(set(tasks) - set(OBJECTIVE_TASKS + SUBJECTIVE_TASKS))
    if unknown:
        raise ValueError(f"Unknown scoring tasks: {unknown}")
    if args.code_workers < 1:
        raise ValueError("--code-workers must be >= 1.")
    code_timeouts = {
        "humaneval": float(
            args.code_timeout
            if args.code_timeout is not None
            else args.humaneval_timeout
        ),
        "mbpp": float(
            args.code_timeout if args.code_timeout is not None else args.mbpp_timeout
        ),
    }
    if any(timeout <= 0 for timeout in code_timeouts.values()):
        raise ValueError("Code timeouts must be > 0.")

    bwrap_bin = shutil.which(args.bwrap)
    if args.allow_code_execution and bwrap_bin is None:
        raise RuntimeError(
            "bubblewrap is required for code scoring; refusing unsafe execution."
        )
    if args.allow_code_execution and shutil.which("prlimit") is None:
        raise RuntimeError("prlimit is required for sandbox resource limits.")

    report: dict[str, Any] = {
        "schema_version": 1,
        "prediction_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_evaluation_profile": DSPARK_CODE_PROFILE,
        "datasets": {},
    }
    for dataset_name in tasks:
        records = load_prediction_records(root, dataset_name)
        if not records:
            report["datasets"][dataset_name] = {
                "status": "missing_predictions",
                "num_samples": 0,
            }
            continue
        if dataset_name in {"gsm8k", "gsm_hard"}:
            report["datasets"][dataset_name] = score_gsm8k(records)
        elif dataset_name in MATH_TASKS:
            report["datasets"][dataset_name] = score_math(records)
        elif dataset_name in MULTIPLE_CHOICE_TASKS:
            report["datasets"][dataset_name] = score_multiple_choice(records)
        elif dataset_name in CODE_TASKS:
            profile_sources: Counter[str] = Counter()
            for record in records:
                profile_sources[
                    validate_dspark_code_profile(record, dataset_name)
                ] += 1
            if not args.allow_code_execution:
                report["datasets"][dataset_name] = {
                    "status": "code_execution_disabled",
                    "num_samples": len(records),
                    "prompt_contract": "original_dspark",
                    "profile_sources": dict(profile_sources),
                }
            else:
                assert bwrap_bin is not None
                dataset_report = score_code(
                    records,
                    dataset_name=dataset_name,
                    bwrap_bin=bwrap_bin,
                    python_bin=args.python_bin,
                    timeout_seconds=code_timeouts[dataset_name],
                    workers=int(args.code_workers),
                )
                dataset_report["profile_sources"] = dict(profile_sources)
                report["datasets"][dataset_name] = dataset_report
        else:
            report["datasets"][dataset_name] = subjective_status(
                dataset_name,
                records,
            )

    output_path = args.output or (root / "quality_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote quality metrics to {output_path}")


if __name__ == "__main__":
    main()
