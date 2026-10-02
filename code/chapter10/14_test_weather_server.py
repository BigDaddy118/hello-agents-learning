#!/usr/bin/env python3
"""测试天气查询 MCP 服务器（10.5.1，无需 LLM）。"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from hello_agents.protocols import MCPClient


async def test_weather_server() -> None:
    server_script = os.path.join(os.path.dirname(__file__), "14_weather_mcp_server.py")
    # Windows 下用当前解释器，避免 PATH 里没有 python
    client = MCPClient([sys.executable, server_script])

    try:
        async with client:
            info = json.loads(await client.call_tool("get_server_info", {}))
            print(f"服务器: {info['name']} v{info['version']}")

            cities = json.loads(await client.call_tool("list_supported_cities", {}))
            print(f"支持城市: {cities['count']} 个")

            weather = json.loads(
                await client.call_tool("get_weather", {"city": "北京"})
            )
            if "error" not in weather:
                print(f"\n北京天气: {weather['temperature']}°C, {weather['condition']}")
            else:
                print(f"\n北京天气查询失败: {weather['error']}")

            weather = json.loads(
                await client.call_tool("get_weather", {"city": "深圳"})
            )
            if "error" not in weather:
                print(f"深圳天气: {weather['temperature']}°C, {weather['condition']}")
            else:
                print(f"深圳天气查询失败: {weather['error']}")

            print("\nOK")
    except Exception as e:
        print(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_weather_server())
