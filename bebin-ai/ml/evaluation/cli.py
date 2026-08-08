from pathlib import Path
import argparse
import json

from ml.evaluation.metrics import EvaluationConfig, evaluate_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Bebin AI model evaluation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--dataset", default=None, help="Processed JSONL dataset for loss/perplexity")
    parser.add_argument("--quality-dataset", default=None, help="JSONL rows with prompt and reference fields")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt for latency measurement")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None, help="Optional JSON report output path")
    args = parser.parse_args()

    report = evaluate_model(
        EvaluationConfig(
            checkpoint_path=Path(args.checkpoint),
            tokenizer_path=Path(args.tokenizer),
            dataset_path=Path(args.dataset) if args.dataset else None,
            quality_dataset_path=Path(args.quality_dataset) if args.quality_dataset else None,
            prompts=tuple(args.prompt),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
    )
    payload = report.to_dict()
    output = json.dumps(payload, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
