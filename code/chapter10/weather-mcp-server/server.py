#!/usr/bin/env python3
"""天气查询 MCP 服务器（Streamable HTTP，供 Smithery 发布）。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import requests
from fastmcp import FastMCP

mcp = FastMCP("weather-server")

CITY_MAP = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "成都": "Chengdu",
    "重庆": "Chongqing",
    "武汉": "Wuhan",
    "西安": "Xi'an",
    "南京": "Nanjing",
    "天津": "Tianjin",
    "苏州": "Suzhou",
}


def get_weather_data(city: str) -> dict[str, Any]:
    city_en = CITY_MAP.get(city, city)
    url = f"https://wttr.in/{city_en}?format=j1"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    current = data["current_condition"][0]
    return {
        "city": city,
        "temperature": float(current["temp_C"]),
        "feels_like": float(current["FeelsLikeC"]),
        "humidity": int(current["humidity"]),
        "condition": current["weatherDesc"][0]["value"],
        "wind_speed": round(float(current["windspeedKmph"]) / 3.6, 1),
        "visibility": float(current["visibility"]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的当前天气"""
    try:
        return json.dumps(get_weather_data(city), ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "city": city}, ensure_ascii=False)


@mcp.tool()
def list_supported_cities() -> str:
    """列出所有支持的中文城市"""
    return json.dumps(
        {"cities": list(CITY_MAP.keys()), "count": len(CITY_MAP)},
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def get_server_info() -> str:
    """获取服务器信息"""
    return json.dumps(
        {
            "name": "Weather MCP Server",
            "version": "1.0.0",
            "tools": ["get_weather", "list_supported_cities", "get_server_info"],
        },
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8081"))
    host = os.getenv("HOST", "0.0.0.0")
    # Smithery 要求 Streamable HTTP
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    print(f"Weather MCP listening on http://{host}:{port}/mcp ({transport})")
    mcp.run(transport=transport, host=host, port=port)  # type: ignore[arg-type]
