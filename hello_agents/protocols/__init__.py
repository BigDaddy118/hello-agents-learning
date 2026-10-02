"""通信协议模块（第 10 章）。当前已实现 MCP；A2A / ANP 见后续小节。"""

from .base import Protocol, ProtocolType
from .mcp import (
    MCPClient,
    MCPServer,
    create_context,
    create_demo_calculator_server,
    parse_context,
)

__all__ = [
    "MCPClient",
    "MCPServer",
    "Protocol",
    "ProtocolType",
    "create_context",
    "create_demo_calculator_server",
    "parse_context",
]
