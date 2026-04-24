# ruff: noqa: EXE002
import json
from pathlib import Path

import pytest

from zeroshot_evals import DEFAULT_DATASET_FILE, load_verified_50_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_default_dataset_is_verified_50_jsonl():
    assert DEFAULT_DATASET_FILE.name == "BixBench-Verified-50.jsonl"
    assert DEFAULT_DATASET_FILE.parent.name == "BixBench-Verified-50"


def test_load_verified_50_dataset_validates_capsule_zips(tmp_path: Path):
    capsule_name = "CapsuleFolder-test.zip"
    dataset_path = tmp_path / "BixBench-Verified-50.jsonl"
    (tmp_path / capsule_name).write_bytes(b"zip placeholder")
    _write_jsonl(
        dataset_path,
        [
            {
                "question_id": "bix-test-q1",
                "question": "What is the answer?",
                "ideal": "42",
                "distractors": ["1", "2", "3"],
                "capsule_uuid": "test",
                "data_folder": capsule_name,
            }
        ],
    )

    dataset = load_verified_50_dataset(dataset_path, tmp_path)

    assert len(dataset) == 1
    assert dataset.iloc[0]["distractors"] == ["1", "2", "3"]


def test_load_verified_50_dataset_reports_missing_capsules(tmp_path: Path):
    dataset_path = tmp_path / "BixBench-Verified-50.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "question_id": "bix-test-q1",
                "question": "What is the answer?",
                "ideal": "42",
                "distractors": ["1", "2", "3"],
                "capsule_uuid": "test",
                "data_folder": "CapsuleFolder-missing.zip",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="CapsuleFolder-missing.zip"):
        load_verified_50_dataset(dataset_path, tmp_path)
