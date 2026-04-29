import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .utils import AnswerMode, randomize_choices

REFUSAL_ANSWER = "Insufficient information to answer the question"


class CodexStructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    insufficient_information: bool = False
    summary: str = ""
    files_used: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class PreparedQuery:
    prompt: str
    target: str
    unsure: str | None
    choice_lines: list[str]


def prepare_capsule_data(zip_path: str | Path, output_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    marker_path = output_dir / ".capsule_ready"
    if marker_path.is_file():
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = output_dir.parent / f".{output_dir.name}.extracting"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.unpack_archive(str(zip_path), str(extract_dir))
        data_dir = next(
            (
                path
                for path in extract_dir.rglob("*")
                if path.is_dir() and "Data" in path.name
            ),
            None,
        )
        if data_dir is None:
            raise FileNotFoundError(
                f"Could not find a data directory while extracting {zip_path}"
            )

        for item in data_dir.iterdir():
            shutil.move(str(item), str(output_dir / item.name))

        for notebook_file in output_dir.rglob("*.ipynb"):
            notebook_file.unlink()

        marker_path.write_text("ready\n", encoding="utf-8")
        return output_dir
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def _build_mcq_rules(with_refusal: bool) -> str:
    refusal_rule = (
        "If the capsule is insufficient to answer the question, choose the option whose "
        f"text is exactly `{REFUSAL_ANSWER}`."
        if with_refusal
        else (
            "If the capsule is insufficient to answer the question, set `answer` to the "
            f"exact string `{REFUSAL_ANSWER}` and set `insufficient_information` to true."
        )
    )
    return textwrap.dedent(
        f"""\
        Final answer rules:
        - `answer` must be the single option letter only, for example `A`.
        - Do not return the option text in `answer`.
        - {refusal_rule}
        """
    ).strip()


def build_codex_prompt(
    *,
    question: str,
    answer_mode: AnswerMode,
    capsule_dir: str | Path,
    artifacts_dir: str | Path,
    choice_lines: list[str] | None = None,
    with_refusal: bool = False,
) -> str:
    capsule_dir = Path(capsule_dir).resolve()
    artifacts_dir = Path(artifacts_dir).resolve()
    answer_rules = (
        _build_mcq_rules(with_refusal)
        if answer_mode == AnswerMode.mcq
        else textwrap.dedent(
            f"""\
            Final answer rules:
            - `answer` must be the concise final answer only.
            - If the capsule is insufficient to answer the question, set `answer` to the exact
              string `{REFUSAL_ANSWER}` and set `insufficient_information` to true.
            """
        ).strip()
    )
    mcq_block = ""
    if choice_lines:
        mcq_block = "Answer options:\n" + "\n".join(choice_lines)

    return (
        textwrap.dedent(
            f"""\
        You are answering one BixBench question with access to a local data capsule.

        Question:
        {question}

        Local paths:
        - Capsule data directory: {capsule_dir}
        - Writable artifacts directory: {artifacts_dir}

        Requirements:
        - Inspect the capsule data, form a plan, then write and execute any code needed to answer
          the question as rigorously as possible.
        - Use the local capsule data and files you create during analysis. 
        - Use the web to find up to date tools and packages.
        - The reference notebook has been removed. Work directly from the data files.
        - Save any scripts, intermediate outputs, or notes under the writable artifacts directory.
        - Keep the final summary brief and factual.

        {mcq_block}

        {answer_rules}
        """
        ).strip()
        + "\n"
    )


def prepare_query(
    *,
    question: str,
    ideal: str,
    distractors: list[str],
    answer_mode: AnswerMode,
    with_refusal: bool,
    capsule_dir: str | Path,
    artifacts_dir: str | Path,
) -> PreparedQuery:
    choice_lines: list[str] = []
    target = ideal
    unsure = None

    if answer_mode == AnswerMode.mcq:
        choice_lines, target, unsure = randomize_choices(
            ideal, distractors, with_refusal=with_refusal
        )

    prompt = build_codex_prompt(
        question=question,
        answer_mode=answer_mode,
        capsule_dir=capsule_dir,
        artifacts_dir=artifacts_dir,
        choice_lines=choice_lines,
        with_refusal=with_refusal,
    )
    return PreparedQuery(
        prompt=prompt,
        target=target,
        unsure=unsure,
        choice_lines=choice_lines,
    )


def build_output_schema() -> dict[str, Any]:
    schema = CodexStructuredAnswer.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}).keys())
    return schema


def parse_structured_answer(path: str | Path) -> CodexStructuredAnswer:
    return CodexStructuredAnswer.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def _letter_to_choice_text(choice_lines: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in choice_lines:
        match = re.match(r"^\(([A-Z])\)\s*(.+)$", line.strip())
        if match:
            mapping[match.group(1)] = match.group(2).strip()
    return mapping


def normalize_predicted_answer(
    raw_answer: str,
    *,
    answer_mode: AnswerMode,
    choice_lines: list[str] | None = None,
    refusal_letter: str | None = None,
    insufficient_information: bool = False,
) -> str:
    normalized = raw_answer.strip()
    result = normalized
    if answer_mode == AnswerMode.openanswer:
        if insufficient_information:
            result = REFUSAL_ANSWER
        return result

    if normalized in {"", "()"}:
        result = refusal_letter or REFUSAL_ANSWER
    elif normalized == REFUSAL_ANSWER:
        result = refusal_letter or normalized
    else:
        letter_match = re.search(r"\b([A-Z])\b", normalized)
        if letter_match:
            result = letter_match.group(1)
        elif choice_lines:
            for letter, choice_text in _letter_to_choice_text(choice_lines).items():
                if normalized.casefold() == choice_text.casefold():
                    result = letter
                    break

    if insufficient_information and refusal_letter is not None:
        result = refusal_letter
    return result


def build_codex_command(
    *,
    workdir: str | Path,
    capsule_dir: str | Path,
    schema_path: str | Path,
    output_path: str | Path,
    model_name: str | None = None,
) -> list[str]:
    workdir = Path(workdir).resolve()
    capsule_dir = Path(capsule_dir).resolve()
    schema_path = Path(schema_path).resolve()
    output_path = Path(output_path).resolve()
    command = [
        "codex",
        "exec",
        "--full-auto",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
        "-C",
        str(workdir),
        "--add-dir",
        str(capsule_dir),
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
    ]
    if model_name:
        command[2:2] = ["-m", model_name]
    return command


def run_codex_command(
    *,
    command: list[str],
    prompt: str,
    stdout_path: str | Path,
    stderr_path: str | Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    env = environ.copy()
    codex_home = _prepare_writable_codex_home(Path(stdout_path).parent)
    env["CODEX_HOME"] = str(codex_home)
    completed = subprocess.run(  # noqa: S603
        command,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        env=env,
        timeout=timeout_seconds,
    )
    Path(stdout_path).write_text(completed.stdout, encoding="utf-8")
    Path(stderr_path).write_text(completed.stderr, encoding="utf-8")
    return completed


def _prepare_writable_codex_home(run_dir: Path) -> Path:
    source_home = Path(environ.get("CODEX_HOME", Path.home() / ".codex"))
    target_home = run_dir / ".codex-home"
    target_home.mkdir(parents=True, exist_ok=True)

    for filename in ("auth.json", "config.toml", "installation_id", "version.json"):
        source_path = source_home / filename
        target_path = target_home / filename
        if source_path.is_file() and not target_path.exists():
            shutil.copy2(source_path, target_path)

    return target_home
