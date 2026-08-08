from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    epoch: int,
    best_validation_loss: float | None,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optimizer | None = None) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

