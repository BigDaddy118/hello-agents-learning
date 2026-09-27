"""工具注册表（7.5.1）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool) -> None:
        if tool.name in self._tools:
            print(f"⚠️ 警告：工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def register_function(
        self, name: str, description: str, func: Callable[[str], str]
    ) -> None:
        if name in self._functions:
            print(f"⚠️ 警告：工具 '{name}' 已存在，将被覆盖。")
        self._functions[name] = {"description": description, "func": func}
        print(f"✅ 工具 '{name}' 已注册。")

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
        elif name in self._functions:
            del self._functions[name]

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def execute_tool(self, name: str, input_text: str) -> str:
        if name in self._tools:
            try:
                return self._tools[name].run({"input": input_text})
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {e}"
        if name in self._functions:
            try:
                return self._functions[name]["func"](input_text)
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {e}"
        return f"错误：未找到名为 '{name}' 的工具。"

    def get_tools_description(self) -> str:
        lines = [f"- {t.name}: {t.description}" for t in self._tools.values()]
        lines += [
            f"- {n}: {info['description']}" for n, info in self._functions.items()
        ]
        return "\n".join(lines) if lines else "暂无可用工具"

    def list_tools(self) -> list[str]:
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def clear(self) -> None:
        self._tools.clear()
        self._functions.clear()


global_registry = ToolRegistry()
