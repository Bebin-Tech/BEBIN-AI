from tools.calculator import CalculatorTool
from tools.planner import plan_tool_calls
from tools.web_search import WebSearchTool


def test_calculator_tool_evaluates_safe_arithmetic() -> None:
    result = CalculatorTool().run({"expression": "2 + 3 * 4"})

    assert result.content == "2 + 3 * 4 = 14"
    assert result.metadata["value"] == 14


def test_calculator_rejects_unsafe_expression() -> None:
    try:
        CalculatorTool().run({"expression": "__import__('os').system('dir')"})
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("Expected unsafe calculator expression to fail")


def test_tool_planner_detects_calculator_and_document_search() -> None:
    calls = plan_tool_calls("Calculate 4 * 9 and search my documents: model training")

    assert [call.name for call in calls] == ["calculator", "document_search"]
    assert calls[0].arguments["expression"] == "4 * 9"
    assert calls[1].arguments["query"] == "model training"


def test_web_search_tool_is_honest_when_disabled() -> None:
    result = WebSearchTool(enabled=False).run({"query": "latest AI news"})

    assert "not configured" in result.content
    assert result.metadata["enabled"] is False

