from dataclasses import dataclass
import re
import unicodedata


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CleaningConfig:
    min_chars: int = 20
    max_chars: int = 12_000
    normalize_unicode: bool = True


def clean_text(text: str, config: CleaningConfig) -> str:
    """Normalize text while preserving paragraph boundaries."""
    if config.normalize_unicode:
        text = unicodedata.normalize("NFKC", text)

    text = CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    if len(text) > config.max_chars:
        text = text[: config.max_chars].rstrip()

    return text

