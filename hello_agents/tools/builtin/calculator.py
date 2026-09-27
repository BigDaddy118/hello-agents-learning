"""计算器工具（7.5.2）。"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ..base import Tool, ToolParameter


class CalculatorTool(Tool):
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    FUNCTIONS = {
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description="执行数学计算。支持基本运算与数学函数，如 2+3*4, sqrt(16), sin(pi/2)。",
        )

    def run(self, parameters: dict[str, Any]) -> str:
        expression = (
            parameters.get("input")
            or parameters.get("expression")
            or ""
        )
        if not str(expression).strip():
            return "错误：计算表达式不能为空"
        try:
            node = ast.parse(str(expression), mode="eval")
            return str(self._eval_node(node.body))
        except Exception as e:
            return f"计算失败: {e}"

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            op = self.OPERATORS[type(node.op)]
            return op(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            fn = self.FUNCTIONS.get(node.func.id)
            if not fn:
                raise ValueError(f"不支持的函数: {node.func.id}")
            return fn(*[self._eval_node(a) for a in node.args])
        if isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            raise ValueError(f"未定义的变量: {node.id}")
        raise ValueError(f"不支持的表达式类型: {type(node)}")

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="要计算的数学表达式",
                required=True,
            )
        ]


def calculate(expression: str) -> str:
    return CalculatorTool().run({"input": expression})
