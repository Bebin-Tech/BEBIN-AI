from pathlib import Path
import json

from ml.evaluation.metrics import EvaluationConfig, evaluate_model
from ml.tokenizer.bpe import TokenizerConfig, train_tokenizer
from ml.training.trainer import TrainingConfig, train_language_model


def test_evaluate_model_reports_loss_latency_and_quality(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    output_dir = tmp_path / "run"
    _write_dataset(dataset_path)
    _write_quality_dataset(quality_path)
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

    report = evaluate_model(
        EvaluationConfig(
            checkpoint_path=output_dir / "last.pt",
            tokenizer_path=tokenizer_path,
            dataset_path=dataset_path,
            quality_dataset_path=quality_path,
            prompts=("Bebin AI",),
            batch_size=1,
            max_new_tokens=2,
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
        )
    )

    assert report.loss is not None
    assert report.loss.loss > 0
    assert report.loss.perplexity > 1
    assert report.latency is not None
    assert report.latency.generated_tokens >= 1
    assert report.quality is not None
    assert report.quality.examples == 1
    assert 0 <= report.quality.unigram_f1 <= 1
    assert report.to_dict()["device"] in {"cpu", "cuda"}


def test_evaluate_model_can_run_latency_only(tmp_path: Path) -> None:
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

    report = evaluate_model(
        EvaluationConfig(
            checkpoint_path=output_dir / "last.pt",
            tokenizer_path=tokenizer_path,
            prompts=("local model",),
            max_new_tokens=1,
            temperature=0.0,
            top_k=None,
            top_p=None,
        )
    )

    assert report.loss is None
    assert report.quality is None
    assert report.latency is not None
    assert report.latency.prompts == 1


def _write_dataset(path: Path) -> None:
    texts = [
        "Bebin AI learns to evaluate local model quality and latency.",
        "The training loop writes checkpoints that evaluation can load.",
        "Perplexity is calculated from real language modeling loss.",
    ]
    rows = [
        {"id": str(index), "text": text, "source": "test", "metadata": {}}
        for index, text in enumerate(texts)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_quality_dataset(path: Path) -> None:
    row = {
        "prompt": "Bebin AI evaluates",
        "reference": "local model quality and latency",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
