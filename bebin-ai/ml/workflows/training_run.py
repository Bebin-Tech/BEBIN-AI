from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from ml.data_pipeline.pipeline import PipelineConfig, preprocess_dataset
from ml.data_pipeline.validation import validate_dataset
from ml.evaluation.metrics import EvaluationConfig, evaluate_model
from ml.tokenizer.bpe import TokenizerConfig, train_tokenizer
from ml.training.trainer import TrainingConfig, train_language_model


@dataclass(frozen=True)
class TrainingRunConfig:
    raw_inputs: tuple[Path, ...]
    output_dir: Path
    text_field: str = "text"
    min_chars: int = 20
    max_chars: int = 12_000
    keep_duplicates: bool = False
    vocab_size: int = 8_000
    min_frequency: int = 2
    context_length: int = 128
    layers: int = 4
    heads: int = 4
    dim: int = 256
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    validation_ratio: float = 0.1
    checkpoint_every_steps: int = 100
    use_mixed_precision: bool = True
    seed: int = 1337
    evaluate: bool = True
    eval_prompt: str = "Bebin AI"
    eval_max_new_tokens: int = 16


@dataclass(frozen=True)
class TrainingRunResult:
    processed_dataset_path: Path
    tokenizer_path: Path
    run_dir: Path
    last_checkpoint_path: Path
    best_checkpoint_path: Path
    training_log_path: Path
    manifest_path: Path
    evaluation_report_path: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "processed_dataset_path": str(self.processed_dataset_path),
            "tokenizer_path": str(self.tokenizer_path),
            "run_dir": str(self.run_dir),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "training_log_path": str(self.training_log_path),
            "manifest_path": str(self.manifest_path),
            "evaluation_report_path": str(self.evaluation_report_path)
            if self.evaluation_report_path
            else None,
        }


def run_training_pipeline(config: TrainingRunConfig) -> TrainingRunResult:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = config.output_dir / "dataset" / "train.jsonl"
    tokenizer_path = config.output_dir / "tokenizer" / "tokenizer.json"
    run_dir = config.output_dir / "checkpoints"
    evaluation_report_path = config.output_dir / "evaluation" / "report.json"
    manifest_path = config.output_dir / "manifest.json"

    preprocess_result = preprocess_dataset(
        PipelineConfig(
            input_paths=list(config.raw_inputs),
            output_path=dataset_path,
            text_field=config.text_field,
            min_chars=config.min_chars,
            max_chars=config.max_chars,
            deduplicate=not config.keep_duplicates,
        )
    )
    validation_report = validate_dataset(dataset_path, min_chars=config.min_chars)
    if not validation_report.passed:
        raise ValueError(f"Processed dataset failed validation: {validation_report.to_dict()}")

    tokenizer = train_tokenizer(
        dataset_path=dataset_path,
        output_path=tokenizer_path,
        config=TokenizerConfig(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            text_field=config.text_field,
        ),
    )
    training_result = train_language_model(
        TrainingConfig(
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            output_dir=run_dir,
            context_length=config.context_length,
            n_layers=config.layers,
            n_heads=config.heads,
            embedding_dim=config.dim,
            dropout=config.dropout,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            validation_ratio=config.validation_ratio,
            checkpoint_every_steps=config.checkpoint_every_steps,
            use_mixed_precision=config.use_mixed_precision,
            seed=config.seed,
        )
    )

    evaluation_payload: dict[str, object] | None = None
    if config.evaluate:
        evaluation = evaluate_model(
            EvaluationConfig(
                checkpoint_path=training_result.last_checkpoint_path,
                tokenizer_path=tokenizer_path,
                dataset_path=dataset_path,
                prompts=(config.eval_prompt,),
                batch_size=config.batch_size,
                max_new_tokens=config.eval_max_new_tokens,
                temperature=0.0,
                top_k=None,
                top_p=None,
                repetition_penalty=1.0,
                seed=config.seed,
            )
        )
        evaluation_payload = evaluation.to_dict()
        evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_report_path.write_text(json.dumps(evaluation_payload, indent=2) + "\n", encoding="utf-8")
    else:
        evaluation_report_path = None

    manifest = {
        "config": _config_to_dict(config),
        "preprocess": asdict(preprocess_result) | {"output_path": str(preprocess_result.output_path)},
        "validation": validation_report.to_dict(),
        "tokenizer": {
            "path": str(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(),
            "pad_token_id": tokenizer.token_to_id("<pad>"),
            "unk_token_id": tokenizer.token_to_id("<unk>"),
            "bos_token_id": tokenizer.token_to_id("<bos>"),
            "eos_token_id": tokenizer.token_to_id("<eos>"),
        },
        "training": {
            "steps": training_result.steps,
            "train_loss": training_result.train_loss,
            "validation_loss": training_result.validation_loss,
            "best_checkpoint_path": str(training_result.best_checkpoint_path),
            "last_checkpoint_path": str(training_result.last_checkpoint_path),
            "log_path": str(training_result.log_path),
            "device": training_result.device,
            "mixed_precision": training_result.mixed_precision,
        },
        "evaluation": evaluation_payload,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return TrainingRunResult(
        processed_dataset_path=dataset_path,
        tokenizer_path=tokenizer_path,
        run_dir=run_dir,
        last_checkpoint_path=training_result.last_checkpoint_path,
        best_checkpoint_path=training_result.best_checkpoint_path,
        training_log_path=training_result.log_path,
        manifest_path=manifest_path,
        evaluation_report_path=evaluation_report_path,
    )


def _config_to_dict(config: TrainingRunConfig) -> dict[str, object]:
    data = asdict(config)
    data["raw_inputs"] = [str(path) for path in config.raw_inputs]
    data["output_dir"] = str(config.output_dir)
    return data


def _validate_config(config: TrainingRunConfig) -> None:
    if not config.raw_inputs:
        raise ValueError("raw_inputs cannot be empty")
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if config.min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    if config.context_length < 2:
        raise ValueError("context_length must be at least 2")
    if config.layers <= 0:
        raise ValueError("layers must be positive")
    if config.heads <= 0:
        raise ValueError("heads must be positive")
    if config.dim <= 0:
        raise ValueError("dim must be positive")
    if config.dim % config.heads != 0:
        raise ValueError("dim must be divisible by heads")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.eval_max_new_tokens <= 0:
        raise ValueError("eval_max_new_tokens must be positive")
