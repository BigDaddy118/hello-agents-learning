"""将单个 MCP 工具包装成 HelloAgents Tool。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..base import Tool, ToolParameter

if TYPE_CHECKING:
    from .protocol_tools import MCPTool


class MCPWrappedTool(Tool):
    def __init__(
        self,
        mcp_tool: MCPTool,
        tool_info: dict[str, Any],
        prefix: str = "",
    ):
        self.mcp_tool = mcp_tool
        self.tool_info = tool_info
        self.mcp_tool_name = tool_info.get("name", "unknown")
        tool_name = f"{prefix}{self.mcp_tool_name}" if prefix else self.mcp_tool_name
        description = tool_info.get(
            "description", f"MCP工具: {self.mcp_tool_name}"
        )
        self._parameters = self._parse_input_schema(
            tool_info.get("input_schema", {})
        )
        super().__init__(name=tool_name, description=description)

    def _parse_input_schema(
        self, input_schema: dict[str, Any]
    ) -> list[ToolParameter]:
        properties = input_schema.get("properties", {})
        required_fields = input_schema.get("required", [])
        return [
            ToolParameter(
                name=param_name,
                type=param_info.get("type", "string"),
                description=param_info.get("description", ""),
                required=param_name in required_fields,
            )
            for param_name, param_info in properties.items()
        ]

    def get_parameters(self) -> list[ToolParameter]:
        return self._parameters

    def run(self, parameters: dict[str, Any]) -> str:
        return self.mcp_tool.run(
            {
                "action": "call_tool",
                "tool_name": self.mcp_tool_name,
                "arguments": parameters,
            }
        )
