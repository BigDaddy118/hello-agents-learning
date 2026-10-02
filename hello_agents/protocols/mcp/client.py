"""MCP 客户端（基于 FastMCP Client）。"""

from __future__ import annotations

from typing import Any

try:
    from fastmcp import Client
except ImportError as e:
    raise ImportError(
        "MCP client requires fastmcp. Install: pip install 'hello-agents[protocols]'"
    ) from e


class MCPClient:
    def __init__(
        self,
        server_command_or_url: str | Any,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.server_command_or_url = server_command_or_url
        self.args = args or []
        self.env = env
        self.kwargs = kwargs
        self.client: Any = None
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return

        source = self.server_command_or_url
        if hasattr(source, "run") or hasattr(source, "_mcp_server"):
            self.client = Client(source)
        elif isinstance(source, str) and (
            source.startswith("http://") or source.startswith("https://")
        ):
            self.client = Client(source)
        else:
            from fastmcp.client.transports import StdioTransport

            if isinstance(source, list):
                if not source:
                    raise ValueError("MCP stdio command list is empty")
                command = str(source[0])
                cmd_args = [str(a) for a in source[1:]] + list(self.args)
            elif isinstance(source, str):
                command = source
                cmd_args = list(self.args)
            else:
                raise TypeError(f"Unsupported MCP server source: {type(source)!r}")
            self.client = Client(
                StdioTransport(command=command, args=cmd_args, env=self.env)
            )

        await self.client.__aenter__()
        self._connected = True

    async def disconnect(self) -> None:
        if self._connected and self.client:
            await self.client.__aexit__(None, None, None)
            self._connected = False

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("Not connected. Use 'async with MCPClient(...)'")
        result = await self.client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", {})
                or {},
            }
            for tool in result
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if not self._connected:
            raise RuntimeError("Not connected. Use 'async with MCPClient(...)'")
        result = await self.client.call_tool(name, arguments or {})
        if hasattr(result, "data") and result.data is not None:
            return result.data
        if hasattr(result, "content") and result.content:
            texts = [
                item.text
                for item in result.content
                if hasattr(item, "text")
            ]
            if texts:
                return "\n".join(texts)
        return str(result)

    async def list_resources(self) -> list[dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("Not connected. Use 'async with MCPClient(...)'")
        try:
            result = await self.client.list_resources()
            return [
                {
                    "uri": r.uri,
                    "name": r.name if hasattr(r, "name") else str(r.uri),
                    "description": r.description
                    if hasattr(r, "description")
                    else "",
                    "mimeType": r.mimeType if hasattr(r, "mimeType") else None,
                }
                for r in result
            ]
        except Exception:
            return []

    async def read_resource(self, uri: str) -> str:
        if not self._connected:
            raise RuntimeError("Not connected. Use 'async with MCPClient(...)'")
        result = await self.client.read_resource(uri)
        if hasattr(result, "contents") and result.contents:
            texts = [
                c.text for c in result.contents if hasattr(c, "text")
            ]
            return "\n".join(texts) if texts else str(result.contents)
        return str(result)

    async def list_prompts(self) -> list[dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("Not connected. Use 'async with MCPClient(...)'")
        try:
            result = await self.client.list_prompts()
            return [
                {
                    "name": p.name,
                    "description": p.description
                    if hasattr(p, "description")
                    else "",
                    "arguments": p.arguments if hasattr(p, "arguments") else [],
                }
                for p in result
            ]
        except Exception:
            return []

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("Not connected. Use 'async with MCPClient(...)'")
        result = await self.client.get_prompt(name, arguments or {})
        if hasattr(result, "messages"):
            return [
                {
                    "role": m.role if hasattr(m, "role") else "user",
                    "content": m.content.text
                    if hasattr(m.content, "text")
                    else str(m.content),
                }
                for m in result.messages
            ]
        return [{"role": "user", "content": str(result)}]
