from pathlib import Path
import argparse
import json

from ml.training.trainer import TrainingConfig, train_language_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Bebin AI model training")
    parser.add_argument("--dataset", required=True, help="Processed JSONL dataset path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs")
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
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    result = train_language_model(
        TrainingConfig(
            dataset_path=Path(args.dataset),
            tokenizer_path=Path(args.tokenizer),
            output_dir=Path(args.output_dir),
            context_length=args.context_length,
            n_layers=args.layers,
            n_heads=args.heads,
            embedding_dim=args.dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip_norm,
            validation_ratio=args.validation_ratio,
            checkpoint_every_steps=args.checkpoint_every_steps,
            use_mixed_precision=not args.no_mixed_precision,
            resume_from=Path(args.resume_from) if args.resume_from else None,
            seed=args.seed,
        )
    )
    print(
        json.dumps(
            {
                "steps": result.steps,
                "train_loss": result.train_loss,
                "validation_loss": result.validation_loss,
                "best_checkpoint_path": str(result.best_checkpoint_path),
                "last_checkpoint_path": str(result.last_checkpoint_path),
                "log_path": str(result.log_path),
                "device": result.device,
                "mixed_precision": result.mixed_precision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

