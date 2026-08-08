from collections.abc import Iterator
from pathlib import Path
import json


SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".md", ".txt"}


def discover_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            files.extend(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            raise FileNotFoundError(f"Input path does not exist or is unsupported: {path}")

    return sorted(files)


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield value


def read_json(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)

    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: item {index} must be an object")
            yield item
        return

    if isinstance(value, dict):
        yield value
        return

    raise ValueError(f"{path}: JSON input must be an object or list of objects")


def write_jsonl(path: Path, records: Iterator[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")
            count += 1
    return count
