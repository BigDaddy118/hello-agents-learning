"""工具链（7.5.4）。"""

from __future__ import annotations

from typing import Any, Optional

from .registry import ToolRegistry


class ToolChain:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: list[dict[str, Any]] = []

    def add_step(
        self, tool_name: str, input_template: str, output_key: Optional[str] = None
    ) -> None:
        self.steps.append(
            {
                "tool_name": tool_name,
                "input_template": input_template,
                "output_key": output_key or f"step_{len(self.steps)}_result",
            }
        )

    def execute(
        self,
        registry: ToolRegistry,
        input_data: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        if not self.steps:
            return "❌ 工具链为空，无法执行"
        ctx = dict(context or {})
        ctx["input"] = input_data
        final = input_data
        for step in self.steps:
            try:
                actual = step["input_template"].format(**ctx)
            except KeyError as e:
                return f"❌ 模板变量替换失败: {e}"
            result = registry.execute_tool(step["tool_name"], actual)
            ctx[step["output_key"]] = result
            final = result
        return final


class ToolChainManager:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain) -> None:
        self.chains[chain.name] = chain

    def execute_chain(
        self,
        chain_name: str,
        input_data: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        if chain_name not in self.chains:
            return f"❌ 工具链 '{chain_name}' 不存在"
        return self.chains[chain_name].execute(self.registry, input_data, context)

    def list_chains(self) -> list[str]:
        return list(self.chains.keys())
