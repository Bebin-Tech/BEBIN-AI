from collections.abc import Iterable

from tools.base import Tool, ToolCall, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def run(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ValueError(f"Unknown tool: {call.name}")
        return tool.run(call.arguments)

    def run_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        return [self.run(call) for call in calls]

