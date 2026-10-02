"""SimpleAgent（7.4.1）— 基础对话 + 可选 [TOOL_CALL:...] 工具循环。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


class SimpleAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
        tool_registry: ToolRegistry | None = None,
        enable_tool_calling: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    def run(
        self, input_text: str, max_tool_iterations: int = 3, **kwargs: Any
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._get_enhanced_system_prompt()}
        ]
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": input_text})

        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response

        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        base = self.system_prompt or "你是一个有用的AI助手。"
        if not self.enable_tool_calling or not self.tool_registry:
            return base
        desc = self.tool_registry.get_tools_description()
        if not desc or desc == "暂无可用工具":
            return base
        return (
            base
            + "\n\n## 可用工具\n你可以使用以下工具来帮助回答问题:\n"
            + desc
            + "\n\n## 工具调用格式\n"
            "当需要使用工具时，请使用以下格式:\n"
            "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
            "例如:`[TOOL_CALL:calculator:2+3*4]` 或 `[TOOL_CALL:search:Python编程]`\n"
            "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"
        )

    def _run_with_tools(
        self,
        messages: list[dict[str, str]],
        input_text: str,
        max_tool_iterations: int,
        **kwargs: Any,
    ) -> str:
        current = 0
        final = ""
        while current < max_tool_iterations:
            response = self.llm.invoke(messages, **kwargs)
            calls = self._parse_tool_calls(response)
            if not calls:
                final = response
                break
            clean = response
            results = []
            for call in calls:
                results.append(
                    self._execute_tool_call(call["tool_name"], call["parameters"])
                )
                clean = clean.replace(call["original"], "")
            messages.append({"role": "assistant", "content": clean})
            messages.append(
                {
                    "role": "user",
                    "content": "工具执行结果:\n"
                    + "\n\n".join(results)
                    + "\n\n请基于这些结果给出完整的回答。",
                }
            )
            current += 1
        if current >= max_tool_iterations and not final:
            final = self.llm.invoke(messages, **kwargs)
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final, "assistant"))
        return final

    def _parse_tool_calls(self, text: str) -> list[dict[str, str]]:
        matches = re.findall(r"\[TOOL_CALL:([^:]+):([^\]]+)\]", text)
        return [
            {
                "tool_name": name.strip(),
                "parameters": params.strip(),
                "original": f"[TOOL_CALL:{name}:{params}]",
            }
            for name, params in matches
        ]

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        if not self.tool_registry:
            return "❌ 错误:未配置工具注册表"
        try:
            # 字符串参数统一走 registry（含 calculator / search）
            if "=" not in parameters:
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    result = self.tool_registry.execute_tool(tool_name, parameters)
                else:
                    result = tool.run(self._parse_tool_parameters(tool_name, parameters))
            return f"🔧 工具 {tool_name} 执行结果:\n{result}"
        except Exception as e:  # noqa: BLE001
            return f"❌ 工具调用失败:{e}"

    def _parse_tool_parameters(
        self, tool_name: str, parameters: str
    ) -> dict[str, Any]:
        if "=" in parameters:
            out: dict[str, Any] = {}
            for pair in parameters.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    out[k.strip()] = v.strip()
            return self._convert_parameter_types(tool_name, out)
        if tool_name == "search":
            return {"input": parameters}
        return {"input": parameters}

    def _convert_parameter_types(
        self, tool_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.tool_registry:
            return params
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return params
        try:
            type_map = {p.name: p.type for p in tool.get_parameters()}
        except Exception:  # noqa: BLE001
            return params
        converted: dict[str, Any] = {}
        for key, value in params.items():
            ptype = type_map.get(key)
            if ptype in ("number", "integer") and isinstance(value, str):
                try:
                    converted[key] = (
                        float(value) if ptype == "number" else int(value)
                    )
                    continue
                except ValueError:
                    pass
            if ptype == "boolean" and isinstance(value, str):
                converted[key] = value.lower() in ("true", "1", "yes")
                continue
            converted[key] = value
        return converted

    def stream_run(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": input_text})
        full = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full += chunk
            yield chunk
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full, "assistant"))

    def add_tool(self, tool: Any) -> None:
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True
        # MCPTool auto_expand → 注册展开后的独立工具
        if getattr(tool, "auto_expand", False):
            expanded = tool.get_expanded_tools()
            if expanded:
                for t in expanded:
                    self.tool_registry.register_tool(t)
                return
        self.tool_registry.register_tool(tool)

    def has_tools(self) -> bool:
        return bool(self.enable_tool_calling and self.tool_registry)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False

    def list_tools(self) -> list[str]:
        return self.tool_registry.list_tools() if self.tool_registry else []
