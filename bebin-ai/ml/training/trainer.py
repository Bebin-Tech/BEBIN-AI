from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math

import torch
from torch.utils.data import DataLoader

from ml.model.transformer import BebinTransformerLM, build_model_from_tokenizer
from ml.training.checkpoint import load_checkpoint, save_checkpoint
from ml.training.dataset import LanguageModelingDataset, load_token_ids, split_token_ids


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    tokenizer_path: Path
    output_dir: Path
    context_length: int = 128
    n_layers: int = 4
    n_heads: int = 4
    embedding_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    validation_ratio: float = 0.1
    checkpoint_every_steps: int = 100
    use_mixed_precision: bool = True
    resume_from: Path | None = None
    seed: int = 1337


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    train_loss: float
    validation_loss: float | None
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    log_path: Path
    device: str
    mixed_precision: bool


def train_language_model(config: TrainingConfig) -> TrainingResult:
    _validate_training_config(config)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = config.use_mixed_precision and device.type == "cuda"

    token_ids = load_token_ids(config.dataset_path, config.tokenizer_path)
    train_ids, validation_ids = split_token_ids(token_ids, config.validation_ratio)

    train_dataset = LanguageModelingDataset(train_ids, config.context_length)
    validation_dataset = (
        LanguageModelingDataset(validation_ids, config.context_length)
        if len(validation_ids) > config.context_length
        else None
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = (
        DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
        if validation_dataset is not None
        else None
    )

    model = build_model_from_tokenizer(
        tokenizer_path=config.tokenizer_path,
        context_length=config.context_length,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    start_epoch = 0
    step = 0
    best_validation_loss: float | None = None
    if config.resume_from is not None:
        checkpoint = load_checkpoint(config.resume_from, model, optimizer)
        step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_validation_loss = checkpoint.get("best_validation_loss")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "training_log.csv"
    _ensure_log_header(log_path)

    last_train_loss = math.inf
    last_validation_loss: float | None = None
    best_checkpoint_path = config.output_dir / "best.pt"
    last_checkpoint_path = config.output_dir / "last.pt"

    for epoch in range(start_epoch, config.epochs):
        model.train()
        for input_ids, labels in train_loader:
            step += 1
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=mixed_precision):
                _, loss = model(input_ids, labels=labels)
                if loss is None:
                    raise RuntimeError("model did not return a training loss")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            last_train_loss = float(loss.detach().cpu())

            should_checkpoint = step % config.checkpoint_every_steps == 0
            if should_checkpoint:
                last_validation_loss = _evaluate(model, validation_loader, device, mixed_precision)
                _append_log(log_path, epoch + 1, step, last_train_loss, last_validation_loss)
                save_checkpoint(
                    last_checkpoint_path,
                    model,
                    optimizer,
                    step,
                    epoch + 1,
                    best_validation_loss,
                    _serializable_config(config),
                )
                if _is_best(last_validation_loss, best_validation_loss):
                    best_validation_loss = last_validation_loss
                    save_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        step,
                        epoch + 1,
                        best_validation_loss,
                        _serializable_config(config),
                    )

        last_validation_loss = _evaluate(model, validation_loader, device, mixed_precision)
        _append_log(log_path, epoch + 1, step, last_train_loss, last_validation_loss)
        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            step,
            epoch + 1,
            best_validation_loss,
            _serializable_config(config),
        )
        if _is_best(last_validation_loss, best_validation_loss):
            best_validation_loss = last_validation_loss
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                step,
                epoch + 1,
                best_validation_loss,
                _serializable_config(config),
            )

    if not best_checkpoint_path.exists():
        save_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            step,
            config.epochs,
            best_validation_loss,
            _serializable_config(config),
        )

    return TrainingResult(
        steps=step,
        train_loss=last_train_loss,
        validation_loss=last_validation_loss,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        log_path=log_path,
        device=device.type,
        mixed_precision=mixed_precision,
    )


def _evaluate(
    model: BebinTransformerLM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    mixed_precision: bool,
) -> float | None:
    if loader is None:
        return None

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=mixed_precision):
                _, loss = model(input_ids, labels=labels)
            if loss is not None:
                losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else None


def _validate_training_config(config: TrainingConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if config.checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")


def _ensure_log_header(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "step", "train_loss", "validation_loss"])


def _append_log(
    path: Path,
    epoch: int,
    step: int,
    train_loss: float,
    validation_loss: float | None,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, step, train_loss, "" if validation_loss is None else validation_loss])


def _is_best(validation_loss: float | None, best_validation_loss: float | None) -> bool:
    if validation_loss is None:
        return best_validation_loss is None
    return best_validation_loss is None or validation_loss < best_validation_loss


def _serializable_config(config: TrainingConfig) -> dict[str, object]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data

