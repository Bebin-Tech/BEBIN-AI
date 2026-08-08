from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.inference.generator import TextGenerator
from ml.training.dataset import LanguageModelingDataset, load_token_ids, split_token_ids


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("qkv", "output", "net.0", "net.2")


@dataclass(frozen=True)
class LoRATrainingConfig:
    base_checkpoint_path: Path
    tokenizer_path: Path
    dataset_path: Path
    output_dir: Path
    lora: LoRAConfig = LoRAConfig()
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    validation_ratio: float = 0.1
    seed: int = 1337


@dataclass(frozen=True)
class LoRATrainingResult:
    adapter_path: Path
    log_path: Path
    steps: int
    train_loss: float
    validation_loss: float | None
    trainable_parameters: int
    total_parameters: int
    device: str


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling


def apply_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    _validate_lora_config(config)
    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(name, config.target_modules):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, config.rank, config.alpha, config.dropout))
        replaced.append(name)

    if not replaced:
        raise ValueError(f"No linear modules matched LoRA targets: {config.target_modules}")
    return replaced


def train_lora_adapter(config: LoRATrainingConfig) -> LoRATrainingResult:
    _validate_training_config(config)
    torch.manual_seed(config.seed)
    generator = TextGenerator.from_checkpoint(config.base_checkpoint_path, config.tokenizer_path)
    model = generator.model
    device = generator.device
    targets = apply_lora(model, config.lora)

    token_ids = load_token_ids(config.dataset_path, config.tokenizer_path)
    train_ids, validation_ids = split_token_ids(token_ids, config.validation_ratio)
    train_dataset = LanguageModelingDataset(train_ids, model.config.context_length)
    validation_dataset = (
        LanguageModelingDataset(validation_ids, model.config.context_length)
        if len(validation_ids) > model.config.context_length
        else None
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = (
        DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
        if validation_dataset is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "lora_training_log.csv"
    _ensure_log_header(log_path)
    step = 0
    last_train_loss = math.inf
    validation_loss: float | None = None

    for epoch in range(config.epochs):
        model.train()
        for input_ids, labels in train_loader:
            step += 1
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels=labels)
            if loss is None:
                raise RuntimeError("model did not return a training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.gradient_clip_norm,
            )
            optimizer.step()
            last_train_loss = float(loss.detach().cpu())

        validation_loss = _evaluate(model, validation_loader, device)
        _append_log(log_path, epoch + 1, step, last_train_loss, validation_loss)

    adapter_path = config.output_dir / "adapter.pt"
    save_lora_adapter(adapter_path, model, config.lora, targets)

    return LoRATrainingResult(
        adapter_path=adapter_path,
        log_path=log_path,
        steps=step,
        train_loss=last_train_loss,
        validation_loss=validation_loss,
        trainable_parameters=count_trainable_parameters(model),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        device=device.type,
    )


def save_lora_adapter(path: Path, model: nn.Module, config: LoRAConfig, target_modules: list[str]) -> None:
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_a." in key or ".lora_b." in key
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": state,
            "config": asdict(config),
            "target_modules": target_modules,
        },
        path,
    )


def load_lora_adapter(model: nn.Module, adapter_path: Path) -> list[str]:
    payload = torch.load(adapter_path, map_location="cpu")
    raw_config = payload.get("config", {})
    if not isinstance(raw_config, dict):
        raise ValueError("adapter config is missing or invalid")
    config = LoRAConfig(
        rank=int(raw_config.get("rank", 8)),
        alpha=float(raw_config.get("alpha", 16.0)),
        dropout=float(raw_config.get("dropout", 0.0)),
        target_modules=tuple(raw_config.get("target_modules", ("qkv", "output", "net.0", "net.2"))),
    )
    targets = apply_lora(model, config)
    missing, unexpected = model.load_state_dict(payload["adapter_state_dict"], strict=False)
    unexpected = [key for key in unexpected if ".lora_" in key]
    missing = [key for key in missing if ".lora_" in key]
    if missing or unexpected:
        raise ValueError(f"LoRA adapter state mismatch: missing={missing}, unexpected={unexpected}")
    return targets


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
) -> float | None:
    if loader is None:
        return None
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            _, loss = model(input_ids, labels=labels)
            if loss is not None:
                losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else None


def _matches_target(module_name: str, targets: tuple[str, ...]) -> bool:
    return any(module_name.endswith(target) for target in targets)


def _ensure_log_header(path: Path) -> None:
    if path.exists():
        return
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


def _validate_lora_config(config: LoRAConfig) -> None:
    if config.rank <= 0:
        raise ValueError("rank must be positive")
    if config.alpha <= 0:
        raise ValueError("alpha must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in the range [0, 1)")
    if not config.target_modules:
        raise ValueError("target_modules cannot be empty")


def _validate_training_config(config: LoRATrainingConfig) -> None:
    _validate_lora_config(config.lora)
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if not 0 <= config.validation_ratio < 1:
        raise ValueError("validation_ratio must be in the range [0, 1)")
