from pathlib import Path
import json

import pytest

from ml.tokenizer.bpe import TokenizerConfig, decode_ids, encode_text, load_tokenizer, train_tokenizer


def test_train_save_load_and_round_trip_tokenizer(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "tokenizer.json"
    _write_dataset(
        dataset_path,
        [
            "Bebin AI trains its own tokenizer from prepared text data.",
            "A small language model needs stable token IDs and special tokens.",
            "Tokenizer training should save and load without changing behavior.",
        ],
    )

    tokenizer = train_tokenizer(
        dataset_path,
        output_path,
        TokenizerConfig(vocab_size=80, min_frequency=1),
    )
    loaded = load_tokenizer(output_path)

    ids = encode_text(loaded, "Bebin AI trains tokenizers.")
    decoded = decode_ids(loaded, ids)

    assert output_path.exists()
    assert tokenizer.get_vocab_size() <= 80
    assert loaded.token_to_id("<pad>") is not None
    assert loaded.token_to_id("<unk>") is not None
    assert ids[0] == loaded.token_to_id("<bos>")
    assert ids[-1] == loaded.token_to_id("<eos>")
    assert "Bebin" in decoded


def test_train_tokenizer_rejects_empty_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty.jsonl"
    dataset_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No training text"):
        train_tokenizer(dataset_path, tmp_path / "tokenizer.json")


def _write_dataset(path: Path, texts: list[str]) -> None:
    rows = [
        {
            "id": str(index),
            "text": text,
            "source": "test",
            "metadata": {},
        }
        for index, text in enumerate(texts)
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

