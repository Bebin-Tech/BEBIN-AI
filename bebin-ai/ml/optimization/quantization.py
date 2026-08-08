from __future__ import annotations

import torch
from torch import nn

from ml.inference.generator import TextGenerator


def quantize_dynamic_cpu(generator: TextGenerator) -> TextGenerator:
    model = generator.model.to("cpu").eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return TextGenerator(
        model=quantized_model,
        tokenizer=generator.tokenizer,
        device=torch.device("cpu"),
    )
