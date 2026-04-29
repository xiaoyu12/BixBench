import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Reuse shared utilities from the codex agent module
from .codex_agent import (
    REFUSAL_ANSWER,
    PreparedQuery,
    build_output_schema,
    normalize_predicted_answer,
    prepare_capsule_data,
    prepare_query,
)

__all__ = [
    "REFUSAL_ANSWER",
    "ClaudeStructuredAnswer",
    "PreparedQuery",
    "build_claude_command",
    "build_output_schema",
    "normalize_predicted_answer",
    "parse_claude_response",
    "prepare_capsule_data",
    "prepare_query",
    "run_claude_command",
]


class ClaudeStructuredAnswer(BaseModel):
    """Structured answer from Claude Code CLI, matching the Codex schema."""

    model_config = ConfigDict(extra="ignore")
    answer: str
    insufficient_information: bool = False
    summary: str = ""
    files_used: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)


def build_claude_command(
    *,
    model_name: str | None = None,
    timeout_seconds: int = 1800,
) -> list[str]:
    """Build the ``claude`` CLI command for non-interactive execution.

    The prompt is supplied via stdin (piped), and the response is returned as
    JSON on stdout thanks to ``--output-format json``.
    """
    command = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
    ]
    if model_name:
        command.extend(["--model", model_name])
    return command


def _build_json_schema() -> str:
    """Return the JSON schema string for ``--json-schema``."""
    schema = build_output_schema()
    return json.dumps(schema)


def build_claude_command_with_schema(
    *,
    model_name: str | None = None,
) -> list[str]:
    """Build the full ``claude`` CLI command including the JSON schema."""
    command = build_claude_command(model_name=model_name)
    command.extend(["--json-schema", _build_json_schema()])
    return command


def run_claude_command(
    *,
    command: list[str],
    prompt: str,
    stdout_path: str | Path,
    stderr_path: str | Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Execute the ``claude`` CLI and capture output to files."""
    completed = subprocess.run(  # noqa: S603
        command,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    Path(stdout_path).write_text(completed.stdout, encoding="utf-8")
    Path(stderr_path).write_text(completed.stderr, encoding="utf-8")
    return completed


def parse_claude_response(stdout_text: str) -> ClaudeStructuredAnswer:
    """Parse the JSON output from the Claude CLI into a structured answer.

    The Claude CLI with ``--output-format json`` returns a JSON object.
    The structured output (conforming to our schema) may be nested under
    ``result``, ``structured_output``, or at the top level.
    """
    data = json.loads(stdout_text)

    # The Claude CLI wraps the response — try common locations
    if isinstance(data, dict):
        # Try "result" field first (most common)
        if "result" in data:
            result = data["result"]
            # result could be a JSON string or a dict
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, ValueError):
                    # Plain text answer — wrap it
                    return ClaudeStructuredAnswer(answer=result)
            if isinstance(result, dict):
                return ClaudeStructuredAnswer.model_validate(result)

        # Try "structured_output" field
        if "structured_output" in data:
            structured = data["structured_output"]
            if isinstance(structured, str):
                try:
                    structured = json.loads(structured)
                except (json.JSONDecodeError, ValueError):
                    return ClaudeStructuredAnswer(answer=structured)
            if isinstance(structured, dict):
                return ClaudeStructuredAnswer.model_validate(structured)

        # Top-level might be the answer itself
        if "answer" in data:
            return ClaudeStructuredAnswer.model_validate(data)

    raise ValueError(f"Could not parse Claude CLI response: {stdout_text[:500]}")
