#!/usr/bin/env python3
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

from bixbench.codex_agent import (
    REFUSAL_ANSWER,
    build_codex_command,
    build_output_schema,
    normalize_predicted_answer,
    parse_structured_answer,
    prepare_capsule_data,
    prepare_query,
    run_codex_command,
)
from bixbench.utils import AnswerMode
from zeroshot_evals import (
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_FILE,
    load_verified_50_dataset,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Codex agent evaluations on BixBench-Verified-50. Each question gets its "
            "own Codex workspace plus access to the extracted capsule data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_DATASET_FILE,
        help="JSONL dataset file to evaluate",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing the dataset JSONL and capsule zips",
    )
    parser.add_argument(
        "--no-capsule-validation",
        action="store_true",
        help="Skip validation that every row's data_folder zip exists",
    )
    parser.add_argument(
        "--answer-mode",
        choices=["mcq", "openanswer"],
        default="mcq",
        help="Whether Codex should answer as multiple choice or open answer",
    )
    parser.add_argument(
        "--with-refusal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a refusal option in MCQ mode",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Codex model override passed to `codex exec -m`",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=-1,
        help="Number of examples to evaluate. Use -1 for all examples",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Restrict evaluation to one or more 1-based question indices from the "
            "dataset, for example `--question-id 1 --question-id 3`"
        ),
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help=(
            "Restrict evaluation to one or more questions by `question_id` or `short_id`, "
            "for example `bix-12-q2` or `bix-12`"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-question timeout for the Codex run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for MCQ choice shuffling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "codex_agent",
        help="Directory to save results and per-question run artifacts",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output CSV filename",
    )
    return parser.parse_args()


def _default_output_file(args: argparse.Namespace) -> str:
    model_label = args.model or "codex-default"
    return (
        "verified50_codex_"
        f"{args.answer_mode}_{args.with_refusal}_{model_label}_{args.seed}.csv"
    )


def _select_dataset_rows(
    dataset: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    if args.question_id:
        n_rows = len(dataset)
        invalid_indices = sorted(
            {
                question_index
                for question_index in args.question_id
                if question_index < 1 or question_index > n_rows
            }
        )
        if invalid_indices:
            invalid_str = ", ".join(str(index) for index in invalid_indices)
            raise ValueError(
                "Question indices must be between 1 and "
                f"{n_rows}: {invalid_str}"
            )

        selected_positions = [question_index - 1 for question_index in args.question_id]
        dataset = dataset.iloc[selected_positions].copy()

    if args.question:
        selectors = {str(value) for value in args.question}
        short_ids = (
            dataset["short_id"].fillna("").astype(str)
            if "short_id" in dataset.columns
            else pd.Series([""] * len(dataset), index=dataset.index)
        )
        mask = dataset["question_id"].astype(str).isin(selectors) | short_ids.isin(
            selectors
        )
        dataset = dataset[mask].copy()

    if args.num_examples > 0:
        dataset = dataset.head(args.num_examples).copy()
    return dataset.reset_index(drop=True)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _summarize_codex_failure(stderr_path: Path, return_code: int | None) -> str:
    prefix = (
        f"codex exec exited with code {return_code}"
        if return_code is not None
        else "codex exec failed"
    )
    if not stderr_path.is_file():
        return prefix

    stderr_text = stderr_path.read_text(encoding="utf-8").strip()
    if not stderr_text:
        return prefix

    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(("Error:", "ERROR:")):
            return f"{prefix}: {line}"

    tail = " ".join(lines[-3:])
    return f"{prefix}: {tail[:500]}"


def _setup_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    capsules_dir = output_dir / "capsules"
    runs_dir.mkdir(parents=True, exist_ok=True)
    capsules_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir, capsules_dir


def _run_single_question(
    *,
    row: pd.Series,
    args: argparse.Namespace,
    answer_mode: AnswerMode,
    runs_dir: Path,
    capsules_dir: Path,
) -> dict:
    question_id = str(row["question_id"])
    capsule_zip = args.dataset_dir / str(row["data_folder"])
    capsule_dir = prepare_capsule_data(
        capsule_zip,
        capsules_dir / Path(str(row["data_folder"])).stem,
    )

    run_dir = runs_dir / question_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    prepared_query = prepare_query(
        question=str(row["question"]),
        ideal=str(row["ideal"]),
        distractors=[str(item) for item in row.get("distractors", [])],
        answer_mode=answer_mode,
        with_refusal=args.with_refusal,
        capsule_dir=capsule_dir,
        artifacts_dir=artifacts_dir,
    )

    metadata_path = run_dir / "question_metadata.json"
    prompt_path = run_dir / "prompt.txt"
    schema_path = run_dir / "output_schema.json"
    response_path = run_dir / "response.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    _write_json(
        metadata_path,
        {
            "question_id": question_id,
            "question": row["question"],
            "ideal": row["ideal"],
            "distractors": row["distractors"],
            "answer_mode": args.answer_mode,
            "with_refusal": args.with_refusal,
            "capsule_zip": str(capsule_zip),
            "capsule_dir": str(capsule_dir),
        },
    )
    prompt_path.write_text(prepared_query.prompt, encoding="utf-8")
    _write_json(schema_path, build_output_schema())

    command = build_codex_command(
        workdir=run_dir,
        capsule_dir=capsule_dir,
        schema_path=schema_path,
        output_path=response_path,
        model_name=args.model,
    )

    error_message = ""
    structured_answer = None
    return_code = None
    try:
        completed = run_codex_command(
            command=command,
            prompt=prepared_query.prompt,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=args.timeout_seconds,
        )
        return_code = completed.returncode
        if completed.returncode == 0 and response_path.is_file():
            structured_answer = parse_structured_answer(response_path)
        else:
            error_message = (
                _summarize_codex_failure(stderr_path, completed.returncode)
                if completed.returncode != 0
                else "codex exec did not produce a structured response"
            )
    except Exception as exc:
        error_message = str(exc)
        stderr_path.write_text(f"{exc}\n", encoding="utf-8")

    raw_answer = structured_answer.answer if structured_answer else "failed"
    normalized_answer = normalize_predicted_answer(
        raw_answer,
        answer_mode=answer_mode,
        choice_lines=prepared_query.choice_lines,
        refusal_letter=prepared_query.unsure,
        insufficient_information=(
            structured_answer.insufficient_information if structured_answer else False
        ),
    )
    return {
        "uuid": question_id,
        "source_id": row.get("id", None),
        "question_id": question_id,
        "question": row["question"],
        "predicted": normalized_answer,
        "target": prepared_query.target,
        "unsure": prepared_query.unsure,
        "evaluation_mode": row.get("eval_mode", None),
        "capsule_uuid": row.get("capsule_uuid", None),
        "data_folder": row.get("data_folder", None),
        "short_id": row.get("short_id", None),
        "raw_answer": raw_answer,
        "insufficient_information": (
            structured_answer.insufficient_information if structured_answer else False
        ),
        "agent_summary": structured_answer.summary if structured_answer else "",
        "files_used": json.dumps(structured_answer.files_used)
        if structured_answer
        else "[]",
        "commands_run": json.dumps(structured_answer.commands_run)
        if structured_answer
        else "[]",
        "run_dir": str(run_dir),
        "capsule_dir": str(capsule_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "return_code": return_code,
        "error": error_message,
        "refusal_answer": REFUSAL_ANSWER,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    dataset = load_verified_50_dataset(
        input_jsonl=args.input_jsonl,
        dataset_dir=args.dataset_dir,
        validate_capsules=not args.no_capsule_validation,
    )
    dataset = _select_dataset_rows(dataset, args)
    if dataset.empty:
        raise ValueError("No examples selected for evaluation")

    runs_dir, capsules_dir = _setup_output_dirs(args.output_dir)
    output_file = args.output_file or _default_output_file(args)
    csv_path = args.output_dir / output_file

    logger.info("Selected %s question(s)", len(dataset))
    logger.info("Results will be written to %s", csv_path)

    results: list[dict] = []
    answer_mode = AnswerMode(args.answer_mode)
    for index, (_, row) in enumerate(dataset.iterrows(), start=1):
        question_id = str(row["question_id"])
        logger.info("[%s/%s] Running Codex for %s", index, len(dataset), question_id)
        result_dict = _run_single_question(
            row=row,
            args=args,
            answer_mode=answer_mode,
            runs_dir=runs_dir,
            capsules_dir=capsules_dir,
        )
        results.append(result_dict)
        pd.DataFrame(results).to_csv(csv_path, index=False)

        if result_dict["error"]:
            logger.error("[%s] %s", question_id, result_dict["error"])
        else:
            logger.info("[%s] predicted=%s", question_id, result_dict["predicted"])

    logger.info("Finished. Results saved to %s", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
