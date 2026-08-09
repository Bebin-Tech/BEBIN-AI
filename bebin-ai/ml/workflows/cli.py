from pathlib import Path
import argparse
import json

from ml.workflows.training_run import TrainingRunConfig, run_training_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Bebin AI local training workflow")
    parser.add_argument("--input", nargs="+", required=True, help="Raw input files or directories")
    parser.add_argument("--output-dir", required=True, help="Directory for dataset, tokenizer, checkpoints, and reports")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=12_000)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=8_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-prompt", default="Bebin AI")
    parser.add_argument("--eval-max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    result = run_training_pipeline(
        TrainingRunConfig(
            raw_inputs=tuple(Path(item) for item in args.input),
            output_dir=Path(args.output_dir),
            text_field=args.text_field,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            keep_duplicates=args.keep_duplicates,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            context_length=args.context_length,
            layers=args.layers,
            heads=args.heads,
            dim=args.dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip_norm,
            validation_ratio=args.validation_ratio,
            checkpoint_every_steps=args.checkpoint_every_steps,
            use_mixed_precision=not args.no_mixed_precision,
            seed=args.seed,
            evaluate=not args.no_eval,
            eval_prompt=args.eval_prompt,
            eval_max_new_tokens=args.eval_max_new_tokens,
        )
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
