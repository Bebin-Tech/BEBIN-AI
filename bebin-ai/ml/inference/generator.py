from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterator

import torch
from tokenizers import Tokenizer

from ml.inference.sampling import sample_next_token
from ml.model.transformer import BebinTransformerLM, TransformerConfig
from ml.tokenizer.bpe import load_tokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.1
    stop_on_eos: bool = True
    seed: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    token_ids: list[int]
    new_token_ids: list[int]
    stopped_on_eos: bool


class TextGenerator:
    def __init__(
        self,
        model: BebinTransformerLM,
        tokenizer: Tokenizer,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.token_to_id("<eos>")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        tokenizer_path: Path,
        device: torch.device | None = None,
        context_length: int | None = None,
        n_layers: int | None = None,
        n_heads: int | None = None,
        embedding_dim: int | None = None,
        dropout: float | None = None,
    ) -> "TextGenerator":
        tokenizer = load_tokenizer(tokenizer_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = _model_config_from_checkpoint(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            context_length=context_length,
            n_layers=n_layers,
            n_heads=n_heads,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        model = BebinTransformerLM(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, tokenizer=tokenizer, device=device)

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        config = config or GenerationConfig()
        state = self._initial_state(prompt, config)
        for token_id, stopped_on_eos in self.iter_token_ids(prompt, config):
            state["token_ids"].append(token_id)
            state["new_token_ids"].append(token_id)
            state["stopped_on_eos"] = stopped_on_eos

        text = self.tokenizer.decode(state["token_ids"], skip_special_tokens=True)
        return GenerationResult(
            prompt=prompt,
            text=text,
            token_ids=state["token_ids"],
            new_token_ids=state["new_token_ids"],
            stopped_on_eos=state["stopped_on_eos"],
        )

    def iter_token_ids(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[tuple[int, bool]]:
        config = config or GenerationConfig()
        state = self._initial_state(prompt, config)
        token_ids: list[int] = state["token_ids"]

        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                context = token_ids[-self.model.config.context_length :]
                input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
                logits, _ = self.model(input_ids)
                next_logits = logits[0, -1, :].detach().cpu()
                next_token_id = sample_next_token(
                    logits=next_logits,
                    generated_ids=token_ids,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                )
                token_ids.append(next_token_id)
                stopped_on_eos = (
                    config.stop_on_eos
                    and self.eos_token_id is not None
                    and next_token_id == self.eos_token_id
                )
                yield next_token_id, stopped_on_eos
                if stopped_on_eos:
                    break

    def _initial_state(self, prompt: str, config: GenerationConfig) -> dict[str, Any]:
        _validate_generation_config(config)
        if config.seed is not None:
            torch.manual_seed(config.seed)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True).ids
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")

        return {
            "token_ids": list(prompt_ids),
            "new_token_ids": [],
            "stopped_on_eos": False,
        }


def _model_config_from_checkpoint(
    checkpoint: dict[str, Any],
    tokenizer: Tokenizer,
    context_length: int | None,
    n_layers: int | None,
    n_heads: int | None,
    embedding_dim: int | None,
    dropout: float | None,
) -> TransformerConfig:
    saved = checkpoint.get("config", {})
    if not isinstance(saved, dict):
        saved = {}

    return TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=context_length or int(saved.get("context_length", 256)),
        n_layers=n_layers or int(saved.get("n_layers", saved.get("layers", 4))),
        n_heads=n_heads or int(saved.get("n_heads", saved.get("heads", 4))),
        embedding_dim=embedding_dim or int(saved.get("embedding_dim", saved.get("dim", 256))),
        dropout=0.0 if dropout is None else dropout,
        pad_token_id=tokenizer.token_to_id("<pad>") or 0,
    )


def _validate_generation_config(config: GenerationConfig) -> None:
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if config.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if config.top_k is not None and config.top_k < 0:
        raise ValueError("top_k must be non-negative")
    if config.top_p is not None and not 0 < config.top_p <= 1:
        raise ValueError("top_p must be in the range (0, 1]")
    if config.repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
