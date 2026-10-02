"""基于 FastMCP 的 MCP 服务器封装。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

try:
    from fastmcp import FastMCP
except ImportError as e:
    raise ImportError(
        "MCP server requires fastmcp. Install: pip install 'hello-agents[protocols]'"
    ) from e

Transport = Literal["stdio", "http", "sse", "streamable-http"]


class MCPServer:
    def __init__(self, name: str, description: str | None = None):
        self.mcp = FastMCP(name=name)
        self.name = name
        self.description = description or f"{name} MCP Server"

    def add_tool(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if name or description:
            self.mcp.tool(name=name, description=description)(func)
        else:
            self.mcp.tool()(func)

    def add_resource(
        self,
        func: Callable[..., Any],
        uri: str,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.mcp.resource(uri, name=name, description=description)(func)

    def add_prompt(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if name or description:
            self.mcp.prompt(name=name, description=description)(func)
        else:
            self.mcp.prompt()(func)

    def run(self, transport: Transport | None = "stdio", **kwargs: Any) -> None:
        self.mcp.run(transport=transport, **kwargs)

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "protocol": "MCP",
        }


def create_demo_calculator_server() -> Any:
    """内置演示用计算器 FastMCP 实例（供 MCPTool 内存传输）。"""
    server = FastMCP("HelloAgents-BuiltinServer")

    @server.tool()
    def add(a: float, b: float) -> float:
        """加法计算器"""
        return a + b

    @server.tool()
    def subtract(a: float, b: float) -> float:
        """减法计算器"""
        return a - b

    @server.tool()
    def multiply(a: float, b: float) -> float:
        """乘法计算器"""
        return a * b

    @server.tool()
    def divide(a: float, b: float) -> float:
        """除法计算器"""
        if b == 0:
            raise ValueError("除数不能为零")
        return a / b

    @server.tool()
    def greet(name: str = "World") -> str:
        """友好问候"""
        return f"Hello, {name}! 欢迎使用 HelloAgents MCP 工具！"

    @server.tool()
    def get_system_info() -> dict[str, Any]:
        """获取系统信息"""
        import platform
        import sys

        return {
            "platform": platform.system(),
            "python_version": sys.version,
            "server_name": "HelloAgents-BuiltinServer",
            "tools_count": 6,
        }

    return server
