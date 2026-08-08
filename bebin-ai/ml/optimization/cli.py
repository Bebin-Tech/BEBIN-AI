from dataclasses import asdict
from pathlib import Path
import argparse
import json

from ml.inference.generator import GenerationConfig, TextGenerator
from ml.optimization.lora import LoRAConfig, LoRATrainingConfig, train_lora_adapter
from ml.optimization.quantization import quantize_dynamic_cpu


def main() -> int:
    parser = argparse.ArgumentParser(description="Bebin AI fine-tuning and optimization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lora = subparsers.add_parser("lora-train", help="Fine-tune LoRA adapters")
    lora.add_argument("--base-checkpoint", required=True)
    lora.add_argument("--tokenizer", required=True)
    lora.add_argument("--dataset", required=True)
    lora.add_argument("--output-dir", required=True)
    lora.add_argument("--rank", type=int, default=8)
    lora.add_argument("--alpha", type=float, default=16.0)
    lora.add_argument("--dropout", type=float, default=0.0)
    lora.add_argument("--target-module", action="append", default=None)
    lora.add_argument("--batch-size", type=int, default=8)
    lora.add_argument("--epochs", type=int, default=1)
    lora.add_argument("--learning-rate", type=float, default=1e-3)
    lora.add_argument("--weight-decay", type=float, default=0.0)
    lora.add_argument("--gradient-clip-norm", type=float, default=1.0)
    lora.add_argument("--validation-ratio", type=float, default=0.1)
    lora.add_argument("--seed", type=int, default=1337)

    quantized = subparsers.add_parser("quantized-generate", help="Generate with dynamic CPU quantization")
    quantized.add_argument("--checkpoint", required=True)
    quantized.add_argument("--tokenizer", required=True)
    quantized.add_argument("--prompt", required=True)
    quantized.add_argument("--max-new-tokens", type=int, default=32)
    quantized.add_argument("--temperature", type=float, default=0.0)
    quantized.add_argument("--top-k", type=int, default=None)
    quantized.add_argument("--top-p", type=float, default=None)
    quantized.add_argument("--repetition-penalty", type=float, default=1.0)
    quantized.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()
    if args.command == "lora-train":
        result = train_lora_adapter(
            LoRATrainingConfig(
                base_checkpoint_path=Path(args.base_checkpoint),
                tokenizer_path=Path(args.tokenizer),
                dataset_path=Path(args.dataset),
                output_dir=Path(args.output_dir),
                lora=LoRAConfig(
                    rank=args.rank,
                    alpha=args.alpha,
                    dropout=args.dropout,
                    target_modules=tuple(args.target_module)
                    if args.target_module
                    else LoRAConfig().target_modules,
                ),
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                validation_ratio=args.validation_ratio,
                seed=args.seed,
            )
        )
        print(json.dumps(asdict(result), indent=2, default=str))
        return 0

    generator = TextGenerator.from_checkpoint(Path(args.checkpoint), Path(args.tokenizer))
    generator = quantize_dynamic_cpu(generator)
    result = generator.generate(
        args.prompt,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        ),
    )
    print(
        json.dumps(
            {
                "prompt": result.prompt,
                "text": result.text,
                "token_ids": result.token_ids,
                "new_token_ids": result.new_token_ids,
                "stopped_on_eos": result.stopped_on_eos,
                "optimization": "dynamic_cpu_int8",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
