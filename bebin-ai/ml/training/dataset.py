from pathlib import Path
import json

import torch
from torch.utils.data import Dataset

from ml.tokenizer.bpe import encode_text, load_tokenizer


class LanguageModelingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: list[int], context_length: int, stride: int | None = None) -> None:
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        if len(token_ids) <= context_length:
            raise ValueError("not enough tokens to create language modeling examples")

        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride or context_length
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.starts = list(range(0, len(token_ids) - context_length, self.stride))
        if not self.starts:
            self.starts = [0]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        end = start + self.context_length
        window = torch.tensor(self.token_ids[start : end + 1], dtype=torch.long)
        return window[:-1], window[1:]


def load_token_ids(dataset_path: Path, tokenizer_path: Path, text_field: str = "text") -> list[int]:
    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")
    token_ids: list[int] = []

    with dataset_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{dataset_path}:{line_number}: invalid JSONL row") from exc

            text = record.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue

            token_ids.extend(encode_text(tokenizer, text, add_special_tokens=True))
            if eos_id is not None:
                token_ids.append(eos_id)

    if not token_ids:
        raise ValueError(f"No tokens loaded from dataset: {dataset_path}")

    return token_ids


def split_token_ids(token_ids: list[int], validation_ratio: float) -> tuple[list[int], list[int]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in the range [0, 1)")
    if validation_ratio == 0:
        return token_ids, []

    validation_size = max(1, int(len(token_ids) * validation_ratio))
    if validation_size >= len(token_ids) - 1:
        validation_size = max(0, len(token_ids) - 2)

    if validation_size == 0:
        return token_ids, []

    return token_ids[:-validation_size], token_ids[-validation_size:]

