import pytest
import torch

from ml.model.transformer import BebinTransformerLM, TransformerConfig


def test_transformer_forward_returns_logits_and_loss() -> None:
    config = TransformerConfig(
        vocab_size=32,
        context_length=8,
        n_layers=2,
        n_heads=4,
        embedding_dim=16,
        dropout=0.0,
        pad_token_id=0,
    )
    model = BebinTransformerLM(config)
    input_ids = torch.tensor([[2, 4, 5, 3]])
    labels = torch.tensor([[4, 5, 3, 0]])

    logits, loss = model(input_ids, labels=labels)

    assert logits.shape == (1, 4, 32)
    assert loss is not None
    assert loss.item() > 0
    assert model.count_parameters() > 0


def test_transformer_rejects_sequences_longer_than_context() -> None:
    model = BebinTransformerLM(
        TransformerConfig(vocab_size=16, context_length=4, n_layers=1, n_heads=2, embedding_dim=8)
    )
    input_ids = torch.ones((1, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds context_length"):
        model(input_ids)


def test_transformer_config_requires_heads_to_divide_embedding_dim() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BebinTransformerLM(
            TransformerConfig(vocab_size=16, context_length=4, n_layers=1, n_heads=3, embedding_dim=8)
        )


def test_causal_mask_prevents_future_tokens_from_affecting_past_logits() -> None:
    torch.manual_seed(7)
    config = TransformerConfig(
        vocab_size=64,
        context_length=6,
        n_layers=2,
        n_heads=2,
        embedding_dim=16,
        dropout=0.0,
    )
    model = BebinTransformerLM(config)
    model.eval()

    first = torch.tensor([[2, 10, 11, 12, 13, 3]])
    second = torch.tensor([[2, 10, 11, 44, 45, 3]])

    with torch.no_grad():
        first_logits, _ = model(first)
        second_logits, _ = model(second)

    torch.testing.assert_close(first_logits[:, :3, :], second_logits[:, :3, :], atol=1e-5, rtol=1e-5)

