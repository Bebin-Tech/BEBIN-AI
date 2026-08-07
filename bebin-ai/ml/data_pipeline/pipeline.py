from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import hashlib

from ml.data_pipeline.cleaning import CleaningConfig, clean_text
from ml.data_pipeline.io import discover_input_files, read_json, read_jsonl, write_jsonl


@dataclass(frozen=True)
class PipelineConfig:
    input_paths: list[Path]
    output_path: Path
    text_field: str = "text"
    min_chars: int = 20
    max_chars: int = 12_000
    deduplicate: bool = True


@dataclass(frozen=True)
class PipelineResult:
    input_files: int
    raw_records: int
    written_records: int
    skipped_too_short: int
    skipped_duplicates: int
    output_path: Path


def preprocess_dataset(config: PipelineConfig) -> PipelineResult:
    files = discover_input_files(config.input_paths)
    cleaner_config = CleaningConfig(min_chars=config.min_chars, max_chars=config.max_chars)
    seen_hashes: set[str] = set()
    counters = {
        "raw_records": 0,
        "written_records": 0,
        "skipped_too_short": 0,
        "skipped_duplicates": 0,
    }

    def records() -> Iterator[dict[str, object]]:
        for path in files:
            for raw_index, raw_text, metadata in _iter_raw_text(path, config.text_field):
                counters["raw_records"] += 1
                text = clean_text(raw_text, cleaner_config)
                if len(text) < config.min_chars:
                    counters["skipped_too_short"] += 1
                    continue

                text_hash = _hash_text(text)
                if config.deduplicate and text_hash in seen_hashes:
                    counters["skipped_duplicates"] += 1
                    continue

                seen_hashes.add(text_hash)
                counters["written_records"] += 1
                yield {
                    "id": _record_id(path, raw_index, text_hash),
                    "text": text,
                    "source": str(path),
                    "metadata": metadata,
                }

    write_jsonl(config.output_path, records())

    return PipelineResult(
        input_files=len(files),
        raw_records=counters["raw_records"],
        written_records=counters["written_records"],
        skipped_too_short=counters["skipped_too_short"],
        skipped_duplicates=counters["skipped_duplicates"],
        output_path=config.output_path,
    )


def _iter_raw_text(path: Path, text_field: str) -> Iterator[tuple[int, str, dict[str, object]]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        yield 0, text, {"format": suffix.lstrip(".")}
        return

    rows = read_jsonl(path) if suffix == ".jsonl" else read_json(path)
    for index, row in enumerate(rows):
        value = row.get(text_field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{path}: field '{text_field}' must be a string")

        metadata = {key: item for key, item in row.items() if key != text_field}
        metadata["format"] = suffix.lstrip(".")
        yield index, value, metadata


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_id(path: Path, raw_index: int, text_hash: str) -> str:
    source = f"{path.as_posix()}:{raw_index}:{text_hash}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]

