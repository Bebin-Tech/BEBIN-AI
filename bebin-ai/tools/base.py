from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolResult:
    name: str
    content: str
    metadata: dict[str, object]


class Tool(Protocol):
    name: str
    description: str

    def run(self, arguments: dict[str, object]) -> ToolResult:
        raise NotImplementedError

