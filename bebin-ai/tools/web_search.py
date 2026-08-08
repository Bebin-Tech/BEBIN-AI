from tools.base import ToolResult


class WebSearchTool:
    name = "web_search"
    description = "Search the web when a provider is configured."

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def run(self, arguments: dict[str, object]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("web_search requires a query")
        if not self.enabled:
            return ToolResult(
                name=self.name,
                content="Web search is not configured. Set up a search provider before using this tool.",
                metadata={"query": query, "enabled": False},
            )
        raise NotImplementedError("web search provider integration is not configured")

