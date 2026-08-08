import ast
import operator

from tools.base import ToolResult


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool:
    name = "calculator"
    description = "Evaluate arithmetic expressions safely."

    def run(self, arguments: dict[str, object]) -> ToolResult:
        expression = str(arguments.get("expression", "")).strip()
        if not expression:
            raise ValueError("calculator requires an expression")
        value = _evaluate(expression)
        return ToolResult(
            name=self.name,
            content=f"{expression} = {value}",
            metadata={"expression": expression, "value": value},
        )


def _evaluate(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported calculator expression")

