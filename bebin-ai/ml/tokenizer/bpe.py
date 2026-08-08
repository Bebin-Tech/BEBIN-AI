from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import json

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 8_000
    min_frequency: int = 2
    text_field: str = "text"
    special_tokens: tuple[str, ...] = tuple(SPECIAL_TOKENS)


def train_tokenizer(
    dataset_path: Path,
    output_path: Path,
    config: TokenizerConfig | None = None,
) -> Tokenizer:
    config = config or TokenizerConfig()
    texts = list(_iter_texts(dataset_path, config.text_field))
    if not texts:
        raise ValueError(f"No training text found in dataset: {dataset_path}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=list(config.special_tokens),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    _configure_post_processor(tokenizer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def load_tokenizer(path: Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.decoder = ByteLevelDecoder()
    _configure_post_processor(tokenizer)
    return tokenizer


def encode_text(tokenizer: Tokenizer, text: str, add_special_tokens: bool = True) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def decode_ids(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def _iter_texts(dataset_path: Path, text_field: str) -> Iterator[str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

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
            if not isinstance(text, str):
                continue
            if text.strip():
                yield text


def _configure_post_processor(tokenizer: Tokenizer) -> None:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        return

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )
