from pathlib import Path
import json

from ml.data_pipeline.cleaning import CleaningConfig, clean_text
from ml.data_pipeline.pipeline import PipelineConfig, preprocess_dataset
from ml.data_pipeline.validation import validate_dataset


def test_clean_text_normalizes_whitespace_and_control_chars() -> None:
    text = "  Hello\x00\tworld\r\n\r\n\r\nThis   is Bebin AI.  "

    cleaned = clean_text(text, CleaningConfig(min_chars=1))

    assert cleaned == "Hello world\n\nThis is Bebin AI."


def test_preprocess_dataset_writes_canonical_jsonl(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "notes.txt").write_text(
        "Bebin AI needs clean, durable training data for a small language model.",
        encoding="utf-8",
    )
    (raw_dir / "records.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"text": "This row is long enough to keep.", "topic": "keep"}),
                json.dumps({"text": "This row is long enough to keep.", "topic": "duplicate"}),
                json.dumps({"text": "short"}),
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "processed" / "train.jsonl"

    result = preprocess_dataset(
        PipelineConfig(
            input_paths=[raw_dir],
            output_path=output_path,
            min_chars=20,
        )
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert result.input_files == 2
    assert result.raw_records == 4
    assert result.written_records == 2
    assert result.skipped_duplicates == 1
    assert result.skipped_too_short == 1
    assert set(rows[0]) == {"id", "metadata", "source", "text"}


def test_validate_dataset_reports_passed_file(tmp_path: Path) -> None:
    output_path = tmp_path / "train.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "id": "abc",
                "text": "This is a valid processed training record.",
                "source": "memory",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(output_path, min_chars=20)

    assert report.passed is True
    assert report.valid_records == 1
    assert report.avg_text_chars > 20

