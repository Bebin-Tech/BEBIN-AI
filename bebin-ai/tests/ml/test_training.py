from pathlib import Path
import json

import torch

from ml.tokenizer.bpe import TokenizerConfig, train_tokenizer
from ml.training.checkpoint import load_checkpoint
from ml.training.dataset import LanguageModelingDataset, load_token_ids, split_token_ids
from ml.training.trainer import TrainingConfig, train_language_model
from ml.model.transformer import build_model_from_tokenizer


def test_language_modeling_dataset_creates_shifted_examples() -> None:
    dataset = LanguageModelingDataset([1, 2, 3, 4, 5], context_length=3, stride=1)

    input_ids, labels = dataset[0]

    assert input_ids.tolist() == [1, 2, 3]
    assert labels.tolist() == [2, 3, 4]


def test_split_token_ids_keeps_train_and_validation_order() -> None:
    train_ids, validation_ids = split_token_ids(list(range(10)), validation_ratio=0.2)

    assert train_ids == list(range(8))
    assert validation_ids == [8, 9]


def test_training_pipeline_saves_checkpoints_and_logs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    output_dir = tmp_path / "runs"
    _write_dataset(dataset_path)
    train_tokenizer(dataset_path, tokenizer_path, TokenizerConfig(vocab_size=96, min_frequency=1))

    result = train_language_model(
        TrainingConfig(
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            context_length=8,
            n_layers=1,
            n_heads=2,
            embedding_dim=16,
            dropout=0.0,
            batch_size=2,
            epochs=1,
            learning_rate=1e-3,
            validation_ratio=0.0,
            checkpoint_every_steps=1,
            use_mixed_precision=True,
        )
    )

    assert result.steps > 0
    assert result.train_loss > 0
    assert result.last_checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert result.log_path.exists()
    assert result.mixed_precision is False
    assert "train_loss" in result.log_path.read_text(encoding="utf-8")

    model = build_model_from_tokenizer(tokenizer_path, context_length=8, n_layers=1, n_heads=2, embedding_dim=16)
    checkpoint = load_checkpoint(result.last_checkpoint_path, model)
    assert checkpoint["step"] == result.steps


def test_load_token_ids_rejects_empty_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_dataset(tmp_path / "source.jsonl")
    train_tokenizer(tmp_path / "source.jsonl", tokenizer_path, TokenizerConfig(vocab_size=64, min_frequency=1))
    dataset_path.write_text("", encoding="utf-8")

    try:
        load_token_ids(dataset_path, tokenizer_path)
    except ValueError as exc:
        assert "No tokens loaded" in str(exc)
    else:
        raise AssertionError("Expected empty dataset to fail")


def _write_dataset(path: Path) -> None:
    texts = [
        "Bebin AI trains a tiny model on local data for testing.",
        "The training loop saves checkpoints and writes CSV logs.",
        "Gradient clipping and optimizer steps are part of the pipeline.",
        "The model predicts the next token from previous tokens.",
    ]
    rows = [
        {"id": str(index), "text": text, "source": "test", "metadata": {}}
        for index, text in enumerate(texts)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

