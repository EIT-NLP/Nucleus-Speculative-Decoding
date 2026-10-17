
"""DSpark-prompt extraction and functional-test assembly for code benchmarks.

The inference prompts remain the original DSpark prompts.  HumanEval exposes
its required entry point in the prompt, so generated code must define that
entry point.  The original DSpark MBPP prompt exposes only the natural-language
problem and hides the reference function name; for that task only, a single
signature-compatible generated function may be aliased to the hidden entry
point before the public tests run.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from typing import Any


PROFILE_NAME = "dspark_prompt_functional_v1"
MBPP_POSTPROCESSOR = "dspark.mbpp.entrypoint_normalized_v1"
HUMANEVAL_POSTPROCESSOR = "dspark.humaneval.entrypoint_strict_v1"

_FENCED_CODE = re.compile(
    r"```[ \t]*(?:python|py)?[ \t]*\r?\n?(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_FUNCTION_LINE = re.compile(
    r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(",
    flags=re.MULTILINE,
)
_CODE_START = re.compile(
    r"^[ \t]*(?:from[ \t]+|import[ \t]+|(?:async[ \t]+)?def[ \t]+|"
    r"class[ \t]+|@)"
)


class CodePreparationError(ValueError):
    """A generated answer cannot be converted into one unambiguous program."""

    def __init__(
        self,
        status: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.metadata = metadata or {}


@dataclass(frozen=True)
class PreparedCode:
    program: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _Candidate:
    source: str
    label: str
    tree: ast.Module


@dataclass(frozen=True)
class _Signature:
    positional_required: int
    positional_total: int
    has_vararg: bool
    keyword_only_required: tuple[str, ...]
    has_kwarg: bool


def reference_field(record: dict[str, Any], field: str) -> Any:
    reference = record.get("reference")
    if isinstance(reference, dict) and field in reference:
        return reference[field]
    source = record.get("source")
    if isinstance(source, dict) and field in source:
        return source[field]
    raise KeyError(
        f"Prediction for dataset={record.get('dataset')!r}, "
        f"sample_index={record.get('sample_index')} has no reference field "
        f"{field!r}."
    )


def _clean_source(text: str) -> str:
    source = text.strip()
    source = re.sub(r"^\s*\[BEGIN\]\s*", "", source, flags=re.IGNORECASE)
    source = re.sub(r"\s*\[DONE\]\s*$", "", source, flags=re.IGNORECASE)
    source = source.strip()
    if len(source) >= 2 and source[0] == source[-1] and source[0] in {"'", '"'}:
        try:
            value = ast.literal_eval(source)
        except (SyntaxError, ValueError):
            pass
        else:
            if isinstance(value, str):
                source = value.strip()
    return source


def _longest_parseable_prefix(source: str) -> tuple[str, ast.Module] | None:
    lines = source.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            return candidate, ast.parse(candidate)
        except SyntaxError:
            continue
    return None


def _parse_candidate_variants(source: str, label: str) -> list[_Candidate]:
    cleaned = _clean_source(source)
    if not cleaned:
        return []
    lines = cleaned.splitlines()
    start_indices = [0]
    start_indices.extend(
        index
        for index, line in enumerate(lines)
        if index != 0 and _CODE_START.match(line)
    )
    candidates: list[_Candidate] = []
    seen: set[str] = set()
    for start_index in start_indices:
        parsed = _longest_parseable_prefix("\n".join(lines[start_index:]))
        if parsed is None:
            continue
        candidate_source, tree = parsed
        if candidate_source in seen:
            continue
        seen.add(candidate_source)
        candidates.append(
            _Candidate(
                source=candidate_source.rstrip() + "\n",
                label=(
                    label
                    if start_index == 0
                    else f"{label}:code_line_{start_index + 1}"
                ),
                tree=tree,
            )
        )
    return candidates


def _raw_answer_sources(text: str) -> list[tuple[str, str]]:
    sources = [
        (match.group(1), f"fenced_block_{index}")
        for index, match in enumerate(_FENCED_CODE.finditer(text), start=1)
    ]
    sources.append((text, "raw_response"))
    return sources


def _parsed_candidates(text: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    seen: set[str] = set()
    for source, label in _raw_answer_sources(text):
        for candidate in _parse_candidate_variants(source, label):
            if candidate.source in seen:
                continue
            seen.add(candidate.source)
            candidates.append(candidate)
    return candidates


def _top_level_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> _Signature:
    positional = [*node.args.posonlyargs, *node.args.args]
    return _Signature(
        positional_required=len(positional) - len(node.args.defaults),
        positional_total=len(positional),
        has_vararg=node.args.vararg is not None,
        keyword_only_required=tuple(
            argument.arg
            for argument, default in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
            )
            if default is None
        ),
        has_kwarg=node.args.kwarg is not None,
    )


def _signature_compatible(
    expected: ast.FunctionDef | ast.AsyncFunctionDef,
    generated: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Use a deliberately conservative compatibility check for MBPP aliasing."""
    return _signature(expected) == _signature(generated)


def _expected_mbpp_function(
    record: dict[str, Any],
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    reference_code = str(reference_field(record, "code"))
    try:
        tree = ast.parse(reference_code)
    except SyntaxError as exc:
        raise CodePreparationError(
            "invalid_reference",
            f"MBPP reference code is not parseable: {exc}",
        ) from exc
    functions = _top_level_functions(tree)
    if not functions:
        raise CodePreparationError(
            "invalid_reference",
            "MBPP reference code defines no top-level function.",
        )

    tests = reference_field(record, "test_list")
    if not isinstance(tests, list) or not all(isinstance(test, str) for test in tests):
        raise CodePreparationError(
            "invalid_reference",
            "MBPP test_list must be a list of strings.",
        )
    called_names: set[str] = set()
    try:
        test_tree = ast.parse("\n".join(tests))
    except SyntaxError as exc:
        raise CodePreparationError(
            "invalid_reference",
            f"MBPP public tests are not parseable: {exc}",
        ) from exc
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
    tested_functions = [node for node in functions if node.name in called_names]
    if len(tested_functions) == 1:
        return tested_functions[0]
    if not tested_functions and len(functions) == 1:
        return functions[0]
    raise CodePreparationError(
        "invalid_reference",
        "Could not identify exactly one tested MBPP reference entry point.",
        metadata={
            "reference_functions": [node.name for node in functions],
            "test_called_names": sorted(called_names),
        },
    )


def _select_mbpp_candidate(
    output: str,
    expected: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[_Candidate, str | None, list[str]]:
    candidates = _parsed_candidates(output)
    exact_matches: list[tuple[_Candidate, list[ast.FunctionDef | ast.AsyncFunctionDef]]] = []
    for candidate in candidates:
        functions = _top_level_functions(candidate.tree)
        if any(node.name == expected.name for node in functions):
            exact_matches.append((candidate, functions))
    if exact_matches:
        candidate, functions = exact_matches[0]
        return candidate, None, [node.name for node in functions]

    function_candidates = [
        (candidate, _top_level_functions(candidate.tree))
        for candidate in candidates
        if _top_level_functions(candidate.tree)
    ]
    if not function_candidates:
        raise CodePreparationError(
            "extraction_failed",
            "No parseable top-level function was found in the MBPP response.",
            metadata={"expected_entry_point": expected.name},
        )

    candidate, functions = function_candidates[0]
    compatible = [
        node for node in functions if _signature_compatible(expected, node)
    ]
    metadata = {
        "expected_entry_point": expected.name,
        "candidate_source": candidate.label,
        "generated_functions": [node.name for node in functions],
        "signature_compatible_functions": [node.name for node in compatible],
    }
    if len(compatible) != 1:
        status = "ambiguous_entrypoint" if len(compatible) > 1 else "incompatible_entrypoint"
        raise CodePreparationError(
            status,
            (
                "MBPP entry-point normalization requires exactly one "
                "signature-compatible generated function; found "
                f"{len(compatible)}."
            ),
            metadata=metadata,
        )
    return candidate, compatible[0].name, [node.name for node in functions]


def prepare_mbpp_program(record: dict[str, Any]) -> PreparedCode:
    expected = _expected_mbpp_function(record)
    candidate, alias_source, generated_functions = _select_mbpp_candidate(
        str(record["output"]),
        expected,
    )
    tests = reference_field(record, "test_list")
    imports = reference_field(record, "test_imports")
    if not isinstance(imports, list) or not all(
        isinstance(item, str) for item in imports
    ):
        raise CodePreparationError(
            "invalid_reference",
            "MBPP test_imports must be a list of strings.",
        )
    alias_line = (
        f"\n{expected.name} = {alias_source}\n" if alias_source is not None else "\n"
    )
    program = (
        "\n".join(imports)
        + ("\n" if imports else "")
        + candidate.source
        + alias_line
        + "\n".join(tests)
        + "\n"
    )
    try:
        compile(program, "<dspark-mbpp>", "exec")
    except SyntaxError as exc:
        raise CodePreparationError(
            "extraction_failed",
            f"Prepared MBPP program is not parseable: {exc}",
        ) from exc
    return PreparedCode(
        program=program,
        metadata={
            "extraction_mode": (
                "exact_entrypoint" if alias_source is None else "signature_alias"
            ),
            "candidate_source": candidate.label,
            "expected_entry_point": expected.name,
            "generated_functions": generated_functions,
            "entrypoint_alias": (
                None
                if alias_source is None
                else {"from": alias_source, "to": expected.name}
            ),
        },
    )


def _humaneval_reference_prefix(prompt: str, entry_point: str) -> str:
    try:
        tree = ast.parse(prompt)
    except SyntaxError as exc:
        raise CodePreparationError(
            "invalid_reference",
            f"HumanEval reference prompt is not parseable: {exc}",
        ) from exc
    nodes = [
        node
        for node in _top_level_functions(tree)
        if node.name == entry_point
    ]
    if len(nodes) != 1:
        raise CodePreparationError(
            "invalid_reference",
            f"HumanEval prompt must define entry point {entry_point!r} exactly once.",
        )
    return "".join(prompt.splitlines(keepends=True)[: nodes[0].lineno - 1])


def _assemble_humaneval_program(
    *,
    prefix: str,
    solution: str,
    tests: str,
    entry_point: str,
) -> str:
    return (
        prefix
        + solution.rstrip()
        + "\n\n"
        + tests.rstrip()
        + f"\n\ncheck({entry_point})\n"
    )


def prepare_humaneval_program(record: dict[str, Any]) -> PreparedCode:
    output = str(record["output"])
    prompt = str(reference_field(record, "prompt"))
    tests = str(reference_field(record, "test"))
    entry_point = str(reference_field(record, "entry_point"))
    prefix = _humaneval_reference_prefix(prompt, entry_point)
    candidates = _parsed_candidates(output)

    for candidate in candidates:
        functions = _top_level_functions(candidate.tree)
        if any(node.name == entry_point for node in functions):
            program = _assemble_humaneval_program(
                prefix=prefix,
                solution=candidate.source,
                tests=tests,
                entry_point=entry_point,
            )
            try:
                compile(program, "<dspark-humaneval>", "exec")
            except SyntaxError as exc:
                raise CodePreparationError(
                    "extraction_failed",
                    f"Prepared HumanEval program is not parseable: {exc}",
                ) from exc
            return PreparedCode(
                program=program,
                metadata={
                    "extraction_mode": "full_function",
                    "candidate_source": candidate.label,
                    "expected_entry_point": entry_point,
                    "generated_functions": [node.name for node in functions],
                    "entrypoint_alias": None,
                },
            )

    generated_names = sorted(set(_FUNCTION_LINE.findall(output)))
    if generated_names:
        raise CodePreparationError(
            "missing_entrypoint",
            (
                f"HumanEval response must define the visible entry point "
                f"{entry_point!r}; found {generated_names}."
            ),
            metadata={
                "expected_entry_point": entry_point,
                "generated_functions": generated_names,
            },
        )



    for body_source, label in _raw_answer_sources(output):
        body = _clean_source(body_source)
        if not body:
            continue
        completion = prompt.rstrip() + "\n" + textwrap.indent(
            textwrap.dedent(body).strip(),
            "    ",
        )
        program = _assemble_humaneval_program(
            prefix="",
            solution=completion,
            tests=tests,
            entry_point=entry_point,
        )
        try:
            compile(program, "<dspark-humaneval>", "exec")
        except SyntaxError:
            continue
        return PreparedCode(
            program=program,
            metadata={
                "extraction_mode": "function_body_completion",
                "candidate_source": label,
                "expected_entry_point": entry_point,
                "generated_functions": [],
                "entrypoint_alias": None,
            },
        )

    raise CodePreparationError(
        "extraction_failed",
        (
            "No parseable HumanEval implementation or function-body "
            f"completion for entry point {entry_point!r} was found."
        ),
        metadata={"expected_entry_point": entry_point},
    )


def build_mbpp_program(record: dict[str, Any]) -> str:
    return prepare_mbpp_program(record).program


def build_humaneval_program(record: dict[str, Any]) -> str:
    return prepare_humaneval_program(record).program


def postprocessor_name(task_name: str) -> str:
    if task_name == "mbpp":
        return MBPP_POSTPROCESSOR
    if task_name == "humaneval":
        return HUMANEVAL_POSTPROCESSOR
    raise ValueError(f"Unsupported DSpark code task: {task_name}")
