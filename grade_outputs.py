import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from lmi import LiteLLMModel

from bixbench import (
    AnswerMode,
    GradeAnswer,
    compute_metrics,
)

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grade answers from a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file", required=True, help="Input CSV file with answers to grade"
    )
    parser.add_argument(
        "--answer-mode",
        choices=["mcq", "openanswer"],
        required=True,
        help="Answer mode",
    )
    parser.add_argument(
        "--model", default="gpt-4o", help="Model name for open-answer grading"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Model temperature"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save results"
    )
    parser.add_argument("--output-file", default=None, help="Output JSON filename")
    return parser.parse_args()


async def grade_answers(
    input_file: str | Path,
    answer_mode: AnswerMode,
    model_name: str = "gpt-4o",
    temperature: float = 1.0,
    **kwargs: dict[str, Any],
):
    """Grade answers based on evaluation mode."""
    query_df = pd.read_csv(input_file)

    if answer_mode == AnswerMode.openanswer:
        llm_client = LiteLLMModel(
            name=f"{model_name}",
            config={"name": model_name, "temperature": temperature, **kwargs},
        )
        grader = GradeAnswer(
            answer_mode=answer_mode,
            llm_client=llm_client,
        )

        results = [
            await grader.grade(
                question=row["question"],
                target=str(row["target"]),
                predicted=str(row["predicted"]),
                unsure=None,
                evaluation_mode=row.get("evaluation_mode", "llm_verifier"),
                partial_match=True,
                llm_match=True,
            )
            for _, row in query_df.iterrows()
        ]

        query_df["grade"], query_df["correct"], query_df["sure"] = zip(
            *results, strict=True
        )
    elif answer_mode == AnswerMode.mcq:
        grader = GradeAnswer(answer_mode=answer_mode)
        results = [
            await grader.grade(
                target=row["target"],
                predicted=row["predicted"],
                unsure=row["unsure"],
                evaluation_mode="str_verifier",
            )
            for _, row in query_df.iterrows()
        ]

        query_df["grade"], query_df["correct"], query_df["sure"] = zip(
            *results, strict=True
        )

    else:
        raise ValueError(f"Unknown answer mode: {answer_mode}")

    # save query_df as pd
    query_df.to_csv(input_file, index=False)
    metrics = compute_metrics(query_df["grade"].to_list(), query_df["sure"].to_list())
    return metrics, query_df


def generate_grading_report(
    query_df: pd.DataFrame,
    metrics: dict,
    model_name: str,
    answer_mode: str,
    input_file: str,
) -> str:
    """Generate a detailed markdown grading report highlighting incorrect answers."""
    total = len(query_df)
    correct_count = int(query_df["correct"].sum())
    incorrect_count = total - correct_count
    accuracy = correct_count / total * 100 if total > 0 else 0

    incorrect_df = query_df[~query_df["correct"].astype(bool)]
    correct_df = query_df[query_df["correct"].astype(bool)]

    lines = []
    lines.append(f"# Grading Report")
    lines.append(f"")
    lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Input file:** `{input_file}`")
    lines.append(f"- **Model:** {model_name}")
    lines.append(f"- **Answer mode:** {answer_mode}")
    lines.append(f"")
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total questions | {total} |")
    lines.append(f"| Correct | {correct_count} |")
    lines.append(f"| Incorrect | {incorrect_count} |")
    lines.append(f"| Accuracy | {accuracy:.1f}% |")
    for key, value in metrics.items():
        display_val = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {display_val} |")
    lines.append(f"")

    # --- Incorrect answers (highlighted) ---
    lines.append(f"## ❌ Incorrect Answers ({incorrect_count})")
    lines.append(f"")
    if len(incorrect_df) == 0:
        lines.append("All answers are correct! 🎉")
        lines.append("")
    else:
        for i, (_, row) in enumerate(incorrect_df.iterrows(), 1):
            qid = row.get("question_id", "N/A")
            lines.append(f"### {i}. {qid}")
            lines.append(f"")
            question_text = str(row.get("question", "N/A"))
            # Truncate very long questions
            if len(question_text) > 500:
                question_text = question_text[:500] + "..."
            lines.append(f"**Question:** {question_text}")
            lines.append(f"")
            lines.append(f"| | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| **Expected (target)** | `{row.get('target', 'N/A')}` |")
            lines.append(f"| **Predicted** | `{row.get('predicted', 'N/A')}` |")
            lines.append(f"| **Unsure** | `{row.get('unsure', 'N/A')}` |")
            lines.append(f"| **Grade** | `{row.get('grade', 'N/A')}` |")
            lines.append(f"")
            summary = str(row.get("agent_summary", ""))
            if summary and summary != "nan":
                if len(summary) > 800:
                    summary = summary[:800] + "..."
                lines.append(f"**Agent summary:** {summary}")
                lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

    # --- Correct answers (collapsed) ---
    lines.append(f"## ✅ Correct Answers ({correct_count})")
    lines.append(f"")
    if len(correct_df) == 0:
        lines.append("No correct answers.")
        lines.append("")
    else:
        lines.append(f"| # | Question ID | Target | Predicted |")
        lines.append(f"|---|-------------|--------|-----------|")
        for i, (_, row) in enumerate(correct_df.iterrows(), 1):
            qid = row.get("question_id", "N/A")
            target = row.get("target", "N/A")
            predicted = row.get("predicted", "N/A")
            lines.append(f"| {i} | {qid} | `{target}` | `{predicted}` |")
        lines.append(f"")

    return "\n".join(lines)


async def main():
    try:
        args = parse_args()
        metrics, query_df = await grade_answers(
            args.input_file,
            args.answer_mode,
            args.model,
            args.temperature,
        )

        # make dir if doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_file = (
            Path(args.input_file).stem + "_graded.json"
            if args.output_file is None
            else args.output_file
        )

        output_path = Path(args.output_dir) / output_file

        print(metrics)
        print(f"Saving results to {output_path}")
        with open(os.path.join(output_path), "w") as f:
            json.dump(metrics, f, indent=4)

        # Generate and save detailed grading report
        report = generate_grading_report(
            query_df=query_df,
            metrics=metrics,
            model_name=args.model,
            answer_mode=args.answer_mode,
            input_file=args.input_file,
        )
        report_filename = Path(args.input_file).stem + "_grading_report.md"
        report_path = Path(args.output_dir) / report_filename
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Detailed grading report saved to {report_path}")

    except Exception as e:
        print(f"Error: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
