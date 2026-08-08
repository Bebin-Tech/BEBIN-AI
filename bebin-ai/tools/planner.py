import re

from tools.base import ToolCall


CALCULATE_RE = re.compile(r"(?:calculate|calculator|what is|solve)\s*[:\-]?\s*([0-9\s+\-*/().%^]+)", re.I)
DOCUMENT_RE = re.compile(r"(?:search documents|document search|search my documents|in my documents)\s*[:\-]?\s*(.+)", re.I)
WEB_RE = re.compile(r"(?:web search|search web|search the web)\s*[:\-]?\s*(.+)", re.I)


def plan_tool_calls(message: str, use_tools: bool = True) -> list[ToolCall]:
    if not use_tools:
        return []

    calls: list[ToolCall] = []
    calculator = CALCULATE_RE.search(message)
    if calculator:
        expression = calculator.group(1).replace("^", "**").strip()
        calls.append(ToolCall(name="calculator", arguments={"expression": expression}))

    document = DOCUMENT_RE.search(message)
    if document:
        calls.append(ToolCall(name="document_search", arguments={"query": document.group(1).strip()}))

    web = WEB_RE.search(message)
    if web:
        calls.append(ToolCall(name="web_search", arguments={"query": web.group(1).strip()}))

    return calls


def format_tool_context(results) -> str:
    if not results:
        return ""

    return "\n\n".join(
        f"[Tool: {result.name}]\n{result.content}"
        for result in results
    )

