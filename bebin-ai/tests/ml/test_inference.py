from pathlib import Path
import json

import torch

from ml.inference.generator import GenerationConfig, TextGenerator
from ml.inference.sampling import filter_top_k, filter_top_p, sample_next_token
from ml.tokenizer.bpe import TokenizerConfig, train_tokenizer
from ml.training.trainer import TrainingConfig, train_language_model


def test_top_k_filter_keeps_only_k_logits() -> None:
    logits = torch.tensor([1.0, 4.0, 3.0, 2.0])

    filtered = filter_top_k(logits, top_k=2)

    assert torch.all(filtered[[0, 3]] < -1e30)
    assert filtered[1].item() == 4.0
    assert filtered[2].item() == 3.0


def test_top_p_filter_keeps_at_least_one_token() -> None:
    logits = torch.tensor([10.0, 1.0, 0.5])

    filtered = filter_top_p(logits, top_p=0.1)

    assert filtered[0].item() == 10.0
    assert torch.all(filtered[1:] < -1e30)


def test_temperature_zero_uses_argmax() -> None:
    token_id = sample_next_token(
        logits=torch.tensor([0.1, 5.0, 2.0]),
        generated_ids=[],
        temperature=0.0,
    )

    assert token_id == 1


def test_text_generator_loads_checkpoint_and_generates_tokens(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    output_dir = tmp_path / "run"
    _write_dataset(dataset_path)
    train_tokenizer(dataset_path, tokenizer_path, TokenizerConfig(vocab_size=96, min_frequency=1))
    train_language_model(
        TrainingConfig(
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            context_length=8,
            n_layers=1,
            n_heads=2,
            embedding_dim=16,
            dropout=0.0,
            batch_size=1,
            epochs=1,
            learning_rate=1e-3,
            validation_ratio=0.0,
            checkpoint_every_steps=1,
            use_mixed_precision=False,
        )
    )

    generator = TextGenerator.from_checkpoint(output_dir / "last.pt", tokenizer_path)
    result = generator.generate(
        "Bebin AI",
        GenerationConfig(
            max_new_tokens=3,
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            seed=123,
        ),
    )

    assert len(result.new_token_ids) >= 1
    assert len(result.new_token_ids) <= 3
    assert result.text


def _write_dataset(path: Path) -> None:
    texts = [
        "Bebin AI learns to generate text from local training data.",
        "Inference samples tokens from model logits with configurable controls.",
        "The assistant uses its own tokenizer and Transformer checkpoint.",
    ]
    rows = [
        {"id": str(index), "text": text, "source": "test", "metadata": {}}
        for index, text in enumerate(texts)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
