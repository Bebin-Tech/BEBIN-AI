from dataclasses import dataclass
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    context_length: int = 256
    n_layers: int = 4
    n_heads: int = 4
    embedding_dim: int = 256
    dropout: float = 0.1
    pad_token_id: int = 0

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.embedding_dim % self.n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")


class BebinTransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")

        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context_length {self.config.context_length}"
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        hidden = self.dropout(token_embeddings + position_embeddings)

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        return hidden


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = hidden.shape
        qkv = self.qkv(hidden)
        query, key, value = qkv.chunk(3, dim=-1)

        query = self._split_heads(query, batch_size, sequence_length)
        key = self._split_heads(key, batch_size, sequence_length)
        value = self._split_heads(value, batch_size, sequence_length)

        scores = (query @ key.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        return self.residual_dropout(self.output(attended))

    def _split_heads(self, tensor: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
        return tensor.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def build_model_from_tokenizer(
    tokenizer_path: Path,
    context_length: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    embedding_dim: int = 256,
    dropout: float = 0.1,
) -> BebinTransformerLM:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_token_id = tokenizer.token_to_id("<pad>")
    return BebinTransformerLM(
        TransformerConfig(
            vocab_size=tokenizer.get_vocab_size(),
            context_length=context_length,
            n_layers=n_layers,
            n_heads=n_heads,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pad_token_id=pad_token_id if pad_token_id is not None else 0,
        )
    )

