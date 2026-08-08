from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import json
import math
import re

import torch
from torch.utils.data import DataLoader

from ml.inference.generator import GenerationConfig, TextGenerator
from ml.training.dataset import LanguageModelingDataset, load_token_ids


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: Path
    tokenizer_path: Path
    dataset_path: Path | None = None
    quality_dataset_path: Path | None = None
    prompts: tuple[str, ...] = ()
    batch_size: int = 8
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    seed: int | None = 1337


@dataclass(frozen=True)
class LossMetrics:
    loss: float
    perplexity: float
    examples: int
    tokens: int


@dataclass(frozen=True)
class LatencyMetrics:
    prompts: int
    total_seconds: float
    average_latency_ms: float
    tokens_per_second: float
    generated_tokens: int


@dataclass(frozen=True)
class QualityMetrics:
    examples: int
    exact_match: float
    unigram_precision: float
    unigram_recall: float
    unigram_f1: float


@dataclass(frozen=True)
class EvaluationReport:
    loss: LossMetrics | None
    latency: LatencyMetrics | None
    quality: QualityMetrics | None
    device: str

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": asdict(self.loss) if self.loss else None,
            "latency": asdict(self.latency) if self.latency else None,
            "quality": asdict(self.quality) if self.quality else None,
            "device": self.device,
        }


def evaluate_model(config: EvaluationConfig) -> EvaluationReport:
    _validate_config(config)
    generator = TextGenerator.from_checkpoint(
        checkpoint_path=config.checkpoint_path,
        tokenizer_path=config.tokenizer_path,
    )
    generation_config = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        seed=config.seed,
    )

    loss = (
        evaluate_loss(
            generator=generator,
            dataset_path=config.dataset_path,
            tokenizer_path=config.tokenizer_path,
            batch_size=config.batch_size,
        )
        if config.dataset_path is not None
        else None
    )
    latency = (
        evaluate_latency(generator=generator, prompts=config.prompts, generation_config=generation_config)
        if config.prompts
        else None
    )
    quality = (
        evaluate_quality(
            generator=generator,
            quality_dataset_path=config.quality_dataset_path,
            generation_config=generation_config,
        )
        if config.quality_dataset_path is not None
        else None
    )

    return EvaluationReport(
        loss=loss,
        latency=latency,
        quality=quality,
        device=generator.device.type,
    )


def evaluate_loss(
    generator: TextGenerator,
    dataset_path: Path,
    tokenizer_path: Path,
    batch_size: int = 8,
) -> LossMetrics:
    token_ids = load_token_ids(dataset_path, tokenizer_path)
    dataset = LanguageModelingDataset(token_ids, generator.model.config.context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses: list[float] = []

    generator.model.eval()
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(generator.device)
            labels = labels.to(generator.device)
            _, loss = generator.model(input_ids, labels=labels)
            if loss is not None:
                losses.append(float(loss.detach().cpu()))

    if not losses:
        raise ValueError(f"No evaluation batches were produced from dataset: {dataset_path}")

    mean_loss = sum(losses) / len(losses)
    return LossMetrics(
        loss=mean_loss,
        perplexity=_safe_exp(mean_loss),
        examples=len(dataset),
        tokens=len(token_ids),
    )


def evaluate_latency(
    generator: TextGenerator,
    prompts: tuple[str, ...],
    generation_config: GenerationConfig,
) -> LatencyMetrics:
    if not prompts:
        raise ValueError("at least one prompt is required for latency evaluation")

    total_tokens = 0
    started = perf_counter()
    for prompt in prompts:
        result = generator.generate(prompt, generation_config)
        total_tokens += len(result.new_token_ids)
    total_seconds = perf_counter() - started

    return LatencyMetrics(
        prompts=len(prompts),
        total_seconds=total_seconds,
        average_latency_ms=(total_seconds / len(prompts)) * 1000,
        tokens_per_second=total_tokens / total_seconds if total_seconds > 0 else math.inf,
        generated_tokens=total_tokens,
    )


def evaluate_quality(
    generator: TextGenerator,
    quality_dataset_path: Path,
    generation_config: GenerationConfig,
) -> QualityMetrics:
    examples = _load_quality_examples(quality_dataset_path)
    scores = []
    exact_matches = 0

    for prompt, reference in examples:
        result = generator.generate(prompt, generation_config)
        candidate = _response_only(result.text, prompt)
        score = _unigram_scores(candidate, reference)
        scores.append(score)
        if _normalize(candidate) == _normalize(reference):
            exact_matches += 1

    return QualityMetrics(
        examples=len(examples),
        exact_match=exact_matches / len(examples),
        unigram_precision=sum(score["precision"] for score in scores) / len(scores),
        unigram_recall=sum(score["recall"] for score in scores) / len(scores),
        unigram_f1=sum(score["f1"] for score in scores) / len(scores),
    )


def _load_quality_examples(path: Path) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc

            prompt = record.get("prompt")
            reference = record.get("reference")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number}: prompt must be a non-empty string")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"{path}:{line_number}: reference must be a non-empty string")
            examples.append((prompt, reference))

    if not examples:
        raise ValueError(f"No quality examples found in dataset: {path}")
    return examples


def _unigram_scores(candidate: str, reference: str) -> dict[str, float]:
    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    reference_counts: dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    overlap = 0
    for token in candidate_tokens:
        count = reference_counts.get(token, 0)
        if count > 0:
            overlap += 1
            reference_counts[token] = count - 1

    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _response_only(text: str, prompt: str) -> str:
    return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _validate_config(config: EvaluationConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if config.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if config.repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
