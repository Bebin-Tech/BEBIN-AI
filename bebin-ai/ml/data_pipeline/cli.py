from pathlib import Path
import argparse
import json

from ml.data_pipeline.pipeline import PipelineConfig, preprocess_dataset
from ml.data_pipeline.validation import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Bebin AI dataset pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Clean and normalize raw data")
    preprocess.add_argument("--input", nargs="+", required=True, help="Input files or directories")
    preprocess.add_argument("--output", required=True, help="Output JSONL path")
    preprocess.add_argument("--text-field", default="text", help="Text field for JSON/JSONL inputs")
    preprocess.add_argument("--min-chars", type=int, default=20)
    preprocess.add_argument("--max-chars", type=int, default=12_000)
    preprocess.add_argument("--keep-duplicates", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate processed JSONL data")
    validate.add_argument("--input", required=True, help="Processed JSONL path")
    validate.add_argument("--min-chars", type=int, default=20)

    args = parser.parse_args()

    if args.command == "preprocess":
        result = preprocess_dataset(
            PipelineConfig(
                input_paths=[Path(item) for item in args.input],
                output_path=Path(args.output),
                text_field=args.text_field,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                deduplicate=not args.keep_duplicates,
            )
        )
        print(json.dumps(result.__dict__ | {"output_path": str(result.output_path)}, indent=2))
        return 0

    report = validate_dataset(Path(args.input), min_chars=args.min_chars)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

