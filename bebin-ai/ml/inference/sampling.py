import torch
from torch.nn import functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    if penalty == 1.0:
        return logits

    adjusted = logits.clone()
    for token_id in set(generated_ids):
        if adjusted[token_id] < 0:
            adjusted[token_id] *= penalty
        else:
            adjusted[token_id] /= penalty
    return adjusted


def filter_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.numel():
        return logits

    values, _ = torch.topk(logits, top_k)
    threshold = values[-1]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def filter_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    if top_p <= 0:
        raise ValueError("top_p must be greater than 0")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)

    remove_mask = cumulative > top_p
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False

    filtered = logits.clone()
    filtered[sorted_indices[remove_mask]] = torch.finfo(logits.dtype).min
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    generated_ids: list[int],
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> int:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    logits = filter_top_k(logits, top_k)
    logits = filter_top_p(logits, top_p)

    if temperature == 0:
        return int(torch.argmax(logits).item())

    probabilities = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())

