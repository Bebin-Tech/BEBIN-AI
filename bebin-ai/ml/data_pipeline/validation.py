from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ValidationReport:
    path: Path
    total_records: int
    valid_records: int
    invalid_json_lines: int
    missing_required_fields: int
    empty_text_records: int
    short_text_records: int
    duplicate_text_records: int
    min_text_chars: int
    max_text_chars: int
    avg_text_chars: float

    @property
    def passed(self) -> bool:
        return (
            self.invalid_json_lines == 0
            and self.missing_required_fields == 0
            and self.empty_text_records == 0
            and self.short_text_records == 0
            and self.duplicate_text_records == 0
            and self.valid_records > 0
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["passed"] = self.passed
        return data


def validate_dataset(path: Path, min_chars: int = 20) -> ValidationReport:
    total_records = 0
    valid_records = 0
    invalid_json_lines = 0
    missing_required_fields = 0
    empty_text_records = 0
    short_text_records = 0
    duplicate_text_records = 0
    text_lengths: list[int] = []
    seen_texts: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue

            if not _has_required_fields(record):
                missing_required_fields += 1
                continue

            text = record["text"]
            if not text.strip():
                empty_text_records += 1
                continue

            if len(text) < min_chars:
                short_text_records += 1
                continue

            if text in seen_texts:
                duplicate_text_records += 1
                continue

            seen_texts.add(text)
            valid_records += 1
            text_lengths.append(len(text))

    return ValidationReport(
        path=path,
        total_records=total_records,
        valid_records=valid_records,
        invalid_json_lines=invalid_json_lines,
        missing_required_fields=missing_required_fields,
        empty_text_records=empty_text_records,
        short_text_records=short_text_records,
        duplicate_text_records=duplicate_text_records,
        min_text_chars=min(text_lengths, default=0),
        max_text_chars=max(text_lengths, default=0),
        avg_text_chars=(sum(text_lengths) / len(text_lengths)) if text_lengths else 0.0,
    )


def _has_required_fields(record: object) -> bool:
    if not isinstance(record, dict):
        return False

    return (
        isinstance(record.get("id"), str)
        and isinstance(record.get("text"), str)
        and isinstance(record.get("source"), str)
        and isinstance(record.get("metadata"), dict)
    )

