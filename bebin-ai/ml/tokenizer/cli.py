from pathlib import Path
import argparse
import json

from ml.tokenizer.bpe import TokenizerConfig, decode_ids, encode_text, load_tokenizer, train_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Bebin AI tokenizer tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a BPE tokenizer from processed JSONL")
    train.add_argument("--input", required=True, help="Processed JSONL dataset path")
    train.add_argument("--output", required=True, help="Tokenizer JSON output path")
    train.add_argument("--vocab-size", type=int, default=8_000)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--text-field", default="text")

    encode = subparsers.add_parser("encode", help="Encode text with a saved tokenizer")
    encode.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    encode.add_argument("--text", required=True)
    encode.add_argument("--no-special-tokens", action="store_true")

    decode = subparsers.add_parser("decode", help="Decode token IDs with a saved tokenizer")
    decode.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    decode.add_argument("--ids", nargs="+", type=int, required=True)

    args = parser.parse_args()

    if args.command == "train":
        tokenizer = train_tokenizer(
            dataset_path=Path(args.input),
            output_path=Path(args.output),
            config=TokenizerConfig(
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                text_field=args.text_field,
            ),
        )
        print(
            json.dumps(
                {
                    "output_path": args.output,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "pad_token_id": tokenizer.token_to_id("<pad>"),
                    "unk_token_id": tokenizer.token_to_id("<unk>"),
                    "bos_token_id": tokenizer.token_to_id("<bos>"),
                    "eos_token_id": tokenizer.token_to_id("<eos>"),
                },
                indent=2,
            )
        )
        return 0

    tokenizer = load_tokenizer(Path(args.tokenizer))
    if args.command == "encode":
        ids = encode_text(tokenizer, args.text, add_special_tokens=not args.no_special_tokens)
        print(json.dumps({"ids": ids}, indent=2))
        return 0

    text = decode_ids(tokenizer, args.ids)
    print(json.dumps({"text": text}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

