from pathlib import Path
import json

from ml.inference.generator import GenerationConfig, TextGenerator
from ml.optimization.lora import (
    LoRAConfig,
    apply_lora,
    count_trainable_parameters,
    load_lora_adapter,
    train_lora_adapter,
    LoRATrainingConfig,
)
from ml.optimization.quantization import quantize_dynamic_cpu
from ml.tokenizer.bpe import TokenizerConfig, train_tokenizer
from ml.training.trainer import TrainingConfig, train_language_model


def test_lora_application_freezes_base_and_adds_trainable_adapters(tmp_path: Path) -> None:
    checkpoint_path, tokenizer_path, _ = _train_tiny_model(tmp_path)
    generator = TextGenerator.from_checkpoint(checkpoint_path, tokenizer_path)

    targets = apply_lora(generator.model, LoRAConfig(rank=2, alpha=4.0, target_modules=("qkv",)))

    assert targets
    assert count_trainable_parameters(generator.model) > 0
    assert count_trainable_parameters(generator.model) < sum(
        parameter.numel() for parameter in generator.model.parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in generator.model.named_parameters()
        if ".base." in name
    )


def test_lora_training_saves_reloadable_adapter(tmp_path: Path) -> None:
    checkpoint_path, tokenizer_path, dataset_path = _train_tiny_model(tmp_path)
    result = train_lora_adapter(
        LoRATrainingConfig(
            base_checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            dataset_path=dataset_path,
            output_dir=tmp_path / "lora",
            lora=LoRAConfig(rank=2, alpha=4.0, target_modules=("qkv",)),
            batch_size=1,
            epochs=1,
            learning_rate=1e-3,
            validation_ratio=0.0,
        )
    )

    assert result.adapter_path.exists()
    assert result.trainable_parameters > 0

    generator = TextGenerator.from_checkpoint(checkpoint_path, tokenizer_path)
    targets = load_lora_adapter(generator.model, result.adapter_path)
    generated = generator.generate(
        "Bebin AI",
        GenerationConfig(max_new_tokens=1, temperature=0.0, top_k=None, top_p=None),
    )

    assert targets
    assert generated.new_token_ids


def test_dynamic_cpu_quantization_can_generate(tmp_path: Path) -> None:
    checkpoint_path, tokenizer_path, _ = _train_tiny_model(tmp_path)
    generator = TextGenerator.from_checkpoint(checkpoint_path, tokenizer_path)
    quantized = quantize_dynamic_cpu(generator)

    result = quantized.generate(
        "Bebin AI",
        GenerationConfig(max_new_tokens=1, temperature=0.0, top_k=None, top_p=None),
    )

    assert quantized.device.type == "cpu"
    assert result.new_token_ids


def _train_tiny_model(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    return output_dir / "last.pt", tokenizer_path, dataset_path


def _write_dataset(path: Path) -> None:
    texts = [
        "Bebin AI fine tunes adapters for local model behavior.",
        "LoRA trains a small number of additional parameters.",
        "Dynamic quantization can optimize CPU inference.",
    ]
    rows = [
        {"id": str(index), "text": text, "source": "test", "metadata": {}}
        for index, text in enumerate(texts)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
