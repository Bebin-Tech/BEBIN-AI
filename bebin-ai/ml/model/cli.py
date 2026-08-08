from pathlib import Path
import argparse
import json

import torch

from ml.model.transformer import build_model_from_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Bebin AI model architecture tools")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer JSON path")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    model = build_model_from_tokenizer(
        tokenizer_path=Path(args.tokenizer),
        context_length=args.context_length,
        n_layers=args.layers,
        n_heads=args.heads,
        embedding_dim=args.dim,
        dropout=args.dropout,
    )
    model.eval()

    dummy_input = torch.zeros((1, min(4, args.context_length)), dtype=torch.long)
    with torch.no_grad():
        logits, loss = model(dummy_input)

    print(
        json.dumps(
            {
                "vocab_size": model.config.vocab_size,
                "context_length": model.config.context_length,
                "layers": model.config.n_layers,
                "heads": model.config.n_heads,
                "embedding_dim": model.config.embedding_dim,
                "parameters": model.count_parameters(),
                "logits_shape": list(logits.shape),
                "loss": loss,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

