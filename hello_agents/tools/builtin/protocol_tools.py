"""MCP 协议工具（Agent 侧）。"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from ..base import Tool, ToolParameter


def _run_async(coro: Any) -> Any:
    """在同步上下文中跑协程（已有事件循环时切线程）。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    def _in_thread() -> Any:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result()


class MCPTool(Tool):
    """连接 MCP 服务器并调用工具。

    Examples:
        >>> tool = MCPTool()  # 内置计算器
        >>> tool.run({"action": "call_tool", "tool_name": "add", "arguments": {"a": 10, "b": 20}})
        >>> tool = MCPTool(server_command="npx", server_args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
    """

    def __init__(
        self,
        server_command: str | list[str] | None = None,
        server_args: list[str] | None = None,
        name: str | None = None,
        description: str | None = None,
        auto_expand: bool = True,
        prefix: str = "",
    ):
        # 兼容教程写法: MCPTool(server_command=["python", "server.py"])
        if isinstance(server_command, list):
            if not server_command:
                raise ValueError("server_command list is empty")
            self.server_command: str | None = server_command[0]
            self.server_args = list(server_command[1:]) + list(server_args or [])
        else:
            self.server_command = server_command
            self.server_args = server_args or []
        self.auto_expand = auto_expand
        self.prefix = prefix
        self.server: Any = None
        self.env: dict[str, str] | None = None
        self._available_tools: list[dict[str, Any]] = []

        if server_command is None:
            from hello_agents.protocols.mcp.server import create_demo_calculator_server

            self.server = create_demo_calculator_server()
            default_name = "mcp_builtin"
            default_desc = "内置MCP服务器，提供基础计算和工具功能"
        else:
            label = (
                " ".join(server_command)
                if isinstance(server_command, list)
                else str(server_command)
            )
            default_name = name or "mcp"
            default_desc = f"连接到 MCP 服务器: {label}"
            env_map = {
                "github": {"GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN"},
                "brave-search": {"BRAVE_API_KEY": "BRAVE_API_KEY"},
                "postgres": {"POSTGRES_CONNECTION_STRING": "DATABASE_URL"},
            }
            haystack = f"{label} {' '.join(self.server_args)}".lower()
            for key, mapping in env_map.items():
                if key in haystack:
                    import os

                    self.env = {
                        k: os.getenv(v, "")
                        for k, v in mapping.items()
                        if os.getenv(v)
                    }
                    break

        super().__init__(
            name=name or default_name,
            description=description or default_desc,
        )
        self._discover_tools()
        if self._available_tools and description is None:
            self.description = self._generate_description()

    def _discover_tools(self) -> None:
        try:
            from hello_agents.protocols.mcp.client import MCPClient

            async def discover() -> list[dict[str, Any]]:
                source = self.server if self.server else self.server_command
                async with MCPClient(source, self.server_args, env=self.env) as client:
                    return await client.list_tools()

            self._available_tools = _run_async(discover())
        except Exception:
            self._available_tools = []

    def _generate_description(self) -> str:
        if not self._available_tools:
            return "连接到 MCP 服务器，调用工具、读取资源和获取提示词。"
        if self.auto_expand:
            return (
                f"MCP工具服务器，包含{len(self._available_tools)}个工具。"
                "这些工具会自动展开为独立的工具供Agent使用。"
            )
        lines = [f"MCP工具服务器，提供{len(self._available_tools)}个工具："]
        for tool in self._available_tools:
            short = (tool.get("description") or "无描述").split(".")[0]
            lines.append(f"  • {tool.get('name', 'unknown')}: {short}")
        lines.append(
            '\n调用格式：{"action": "call_tool", "tool_name": "工具名", "arguments": {...}}'
        )
        return "\n".join(lines)

    def get_expanded_tools(self) -> list[Tool]:
        if not self.auto_expand:
            return []
        from .mcp_wrapper_tool import MCPWrappedTool

        return [
            MCPWrappedTool(mcp_tool=self, tool_info=info, prefix=self.prefix)
            for info in self._available_tools
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        from hello_agents.protocols.mcp.client import MCPClient

        action = str(parameters.get("action", "")).lower()
        if not action and "tool_name" in parameters:
            action = "call_tool"
            parameters = {**parameters, "action": action}
        if not action:
            return "错误：必须指定 action 参数或 tool_name 参数"

        async def op() -> str:
            source = self.server if self.server else self.server_command
            async with MCPClient(source, self.server_args, env=self.env) as client:
                if action == "list_tools":
                    tools = await client.list_tools()
                    if not tools:
                        return "没有找到可用的工具"
                    out = f"找到 {len(tools)} 个工具:\n"
                    for t in tools:
                        out += f"- {t['name']}: {t['description']}\n"
                    return out
                if action == "call_tool":
                    tool_name = parameters.get("tool_name")
                    if not tool_name:
                        return "错误：必须指定 tool_name 参数"
                    result = await client.call_tool(
                        tool_name, parameters.get("arguments", {})
                    )
                    return f"工具 '{tool_name}' 执行结果:\n{result}"
                if action == "list_resources":
                    resources = await client.list_resources()
                    if not resources:
                        return "没有找到可用的资源"
                    out = f"找到 {len(resources)} 个资源:\n"
                    for r in resources:
                        out += f"- {r['uri']}: {r['name']}\n"
                    return out
                if action == "read_resource":
                    uri = parameters.get("uri")
                    if not uri:
                        return "错误：必须指定 uri 参数"
                    return f"资源 '{uri}' 内容:\n{await client.read_resource(uri)}"
                if action == "list_prompts":
                    prompts = await client.list_prompts()
                    if not prompts:
                        return "没有找到可用的提示词"
                    out = f"找到 {len(prompts)} 个提示词:\n"
                    for p in prompts:
                        out += f"- {p['name']}: {p['description']}\n"
                    return out
                if action == "get_prompt":
                    prompt_name = parameters.get("prompt_name")
                    if not prompt_name:
                        return "错误：必须指定 prompt_name 参数"
                    messages = await client.get_prompt(
                        prompt_name, parameters.get("prompt_arguments", {})
                    )
                    out = f"提示词 '{prompt_name}':\n"
                    for msg in messages:
                        out += f"[{msg['role']}] {msg['content']}\n"
                    return out
                return f"错误：不支持的操作 '{action}'"

        try:
            return _run_async(op())
        except Exception as e:
            return f"MCP 操作失败: {e}"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "操作类型: list_tools, call_tool, list_resources, "
                    "read_resource, list_prompts, get_prompt"
                ),
                required=True,
            ),
            ToolParameter(
                name="tool_name",
                type="string",
                description="工具名称（call_tool）",
                required=False,
            ),
            ToolParameter(
                name="arguments",
                type="object",
                description="工具参数（call_tool）",
                required=False,
            ),
            ToolParameter(
                name="uri",
                type="string",
                description="资源 URI（read_resource）",
                required=False,
            ),
            ToolParameter(
                name="prompt_name",
                type="string",
                description="提示词名称（get_prompt）",
                required=False,
            ),
            ToolParameter(
                name="prompt_arguments",
                type="object",
                description="提示词参数（get_prompt）",
                required=False,
            ),
        ]
