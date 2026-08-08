from pathlib import Path
import argparse
import json

from ml.inference.generator import GenerationConfig, TextGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description="Bebin AI text generation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-eos-stop", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    args = parser.parse_args()

    generator = TextGenerator.from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        tokenizer_path=Path(args.tokenizer),
        context_length=args.context_length,
        n_layers=args.layers,
        n_heads=args.heads,
        embedding_dim=args.dim,
    )
    result = generator.generate(
        args.prompt,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_on_eos=not args.no_eos_stop,
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
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

