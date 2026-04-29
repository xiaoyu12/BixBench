#!/usr/bin/env python3
import argparse
import ast
import asyncio
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from bixbench import Query, ZeroshotBaseline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = REPO_ROOT / "BixBench-Verified-50"
DEFAULT_DATASET_FILE = DEFAULT_DATASET_DIR / "BixBench-Verified-50.jsonl"
CAPSULE_ERROR_SAMPLE_SIZE = 5
REQUIRED_COLUMNS = {
    "question_id",
    "question",
    "ideal",
    "distractors",
    "capsule_uuid",
    "data_folder",
}

dotenv_path = REPO_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)


def _normalize_distractors(value: Any) -> list[str]:
    if value is None:
        normalized = []
    elif isinstance(value, list | tuple):
        normalized = [str(item) for item in value]
    elif isinstance(value, float) and pd.isna(value):
        normalized = []
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                normalized = [stripped]
            else:
                normalized = (
                    [str(item) for item in parsed]
                    if isinstance(parsed, list | tuple)
                    else [str(parsed)]
                )
        else:
            normalized = []
    else:
        normalized = [str(value)]
    return normalized


def _validate_dataset_columns(dataset: pd.DataFrame, input_path: Path) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataset.columns))
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise ValueError(f"{input_path} is missing required columns: {columns}")


def _validate_capsule_zips(dataset: pd.DataFrame, dataset_dir: Path) -> None:
    missing_capsules = sorted(
        {
            data_folder
            for data_folder in dataset["data_folder"].dropna().astype(str)
            if not (dataset_dir / data_folder).is_file()
        }
    )
    if missing_capsules:
        sample = ", ".join(missing_capsules[:CAPSULE_ERROR_SAMPLE_SIZE])
        more = (
            ""
            if len(missing_capsules) <= CAPSULE_ERROR_SAMPLE_SIZE
            else f", ... ({len(missing_capsules)} total)"
        )
        raise FileNotFoundError(
            "Missing capsule zip files referenced by the dataset under "
            f"{dataset_dir}: {sample}{more}"
        )


def load_verified_50_dataset(
    input_jsonl: str | Path = DEFAULT_DATASET_FILE,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    validate_capsules: bool = True,
) -> pd.DataFrame:
    input_path = Path(input_jsonl)
    resolved_dataset_dir = Path(dataset_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    dataset = pd.read_json(input_path, lines=True)
    _validate_dataset_columns(dataset, input_path)
    dataset["distractors"] = dataset["distractors"].map(_normalize_distractors)

    if validate_capsules:
        _validate_capsule_zips(dataset, resolved_dataset_dir)

    return dataset


def print_dataset_details(dataset: pd.DataFrame) -> None:
    """Print a detailed summary of each question in the dataset."""
    separator = "-" * 80
    print("=" * 80)
    print(f"DATASET OVERVIEW — {len(dataset)} question(s)")
    print(f"  Unique capsules : {dataset['capsule_uuid'].nunique()}")
    if "categories" in dataset.columns:
        all_cats = dataset["categories"].dropna()
        unique_cats = set()
        for cats in all_cats:
            if isinstance(cats, list):
                unique_cats.update(cats)
            elif isinstance(cats, str):
                unique_cats.add(cats)
        print(f"  Categories       : {', '.join(sorted(unique_cats)) or 'N/A'}")
    if "eval_mode" in dataset.columns:
        modes = dataset["eval_mode"].dropna().unique()
        print(f"  Eval modes       : {', '.join(sorted(modes))}")
    print("=" * 80)

    for idx, (_, row) in enumerate(dataset.iterrows(), start=1):
        qid = row.get("question_id") or row.get("id", "N/A")
        print(f"\n{separator}")
        print(f"  [{idx}/{len(dataset)}]  Question ID : {qid}")
        if "short_id" in row and pd.notna(row.get("short_id")):
            print(f"              Short ID : {row['short_id']}")
        print(f"              Capsule  : {row.get('capsule_uuid', 'N/A')}")
        if "categories" in row and row.get("categories"):
            cats = row["categories"]
            if isinstance(cats, list):
                cats = ", ".join(cats)
            print(f"              Categories: {cats}")
        if "eval_mode" in row and pd.notna(row.get("eval_mode")):
            print(f"              Eval mode: {row['eval_mode']}")
        if "paper" in row and pd.notna(row.get("paper")):
            paper = str(row["paper"])
            if len(paper) > 120:
                paper = paper[:117] + "..."
            print(f"              Paper    : {paper}")
        print()
        question_text = row["question"]
        if len(question_text) > 500:
            question_text = question_text[:497] + "..."
        print(f"  Question : {question_text}")
        print(f"  Ideal    : {row['ideal']}")
        distractors = row.get("distractors", [])
        if distractors:
            print(f"  Distractors ({len(distractors)}):")
            for i, d in enumerate(distractors, 1):
                d_str = str(d)
                if len(d_str) > 120:
                    d_str = d_str[:117] + "..."
                print(f"    {i}. {d_str}")
        else:
            print("  Distractors: (none)")

    print(f"\n{'=' * 80}")
    print(f"END OF DATASET — {len(dataset)} question(s) listed")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot BixBench evaluations. By default, this uses the local "
            "BixBench-Verified-50 questions JSONL and validates its capsule zips."
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
        help="Evaluation mode",
    )
    parser.add_argument("--model", default="gpt-5.4", help="Model name to use")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Model temperature",
    )
    parser.add_argument(
        "--with-refusal",
        action="store_true",
        default=False,
        help="Add refusal option for MCQs",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=-1,
        help="Number of examples to evaluate. Use -1 for all examples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output CSV filename",
    )
    return parser.parse_args()


async def evaluate(
    dataset: pd.DataFrame,
    zeroshot_agent: ZeroshotBaseline,
    output_dir: str | Path = "results",
    output_file: str | None = None,
) -> Path:
    results = []
    for _, row in dataset.iterrows():
        query_id = row.get("question_id") or row.get("id")
        query = await zeroshot_agent.generate_zeroshot_answers(
            Query(
                id=query_id,
                question=row["question"],
                target=str(row["ideal"]),
                choices=row["distractors"],
                evaluation_mode=row.get("eval_mode", None),
            )
        )

        result_dict = {
            "uuid": query.id,
            "source_id": row.get("id", None),
            "question_id": row.get("question_id", query.id),
            "question": query.question,
            "predicted": query.predicted,
            "target": query.target,
            "unsure": query.unsure,
            "evaluation_mode": query.evaluation_mode,
            "capsule_uuid": row.get("capsule_uuid", None),
            "data_folder": row.get("data_folder", None),
            "short_id": row.get("short_id", None),
        }
        results.append(result_dict)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_file is None:
        raise ValueError("output_file must be provided")

    csv_path = output_path / output_file
    pd.DataFrame(results).to_csv(csv_path, index=False)
    return csv_path


async def main() -> None:
    args = parse_args()
    logger.info("Starting zero-shot evaluation...")
    output_file = (
        args.output_file
        if args.output_file is not None
        else (
            "verified50_"
            f"{args.answer_mode}_{args.with_refusal}_{args.model}_{args.temperature}.csv"
        )
    )

    dataset = load_verified_50_dataset(
        input_jsonl=args.input_jsonl,
        dataset_dir=args.dataset_dir,
        validate_capsules=not args.no_capsule_validation,
    )
    if args.num_examples > 0:
        dataset = dataset.head(args.num_examples)

    print_dataset_details(dataset)

    zeroshot_agent = ZeroshotBaseline(
        answer_mode=args.answer_mode,
        with_refusal=args.with_refusal,
        model_name=args.model,
        temperature=args.temperature,
    )
    output_path = await evaluate(dataset, zeroshot_agent, args.output_dir, output_file)
    logger.info("Evaluation completed and results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
