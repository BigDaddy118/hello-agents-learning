#!/usr/bin/env python3
"""天气查询 MCP 服务器（10.5.1）。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hello_agents.protocols import MCPServer

weather_server = MCPServer(name="weather-server", description="真实天气查询服务")

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
    """从 wttr.in 获取天气数据。"""
    city_en = CITY_MAP.get(city, city)
    url = f"https://wttr.in/{city_en}?format=j1"
    req = Request(url, headers={"User-Agent": "hello-agents-weather-mcp/1.0"})
    with urlopen(req, timeout=10) as resp:  # noqa: S310 — 固定公开天气 API
        data = json.loads(resp.read().decode("utf-8"))
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


def get_weather(city: str) -> str:
    """获取指定城市的当前天气"""
    try:
        return json.dumps(get_weather_data(city), ensure_ascii=False, indent=2)
    except (HTTPError, URLError, TimeoutError, KeyError, ValueError, TypeError) as e:
        return json.dumps({"error": str(e), "city": city}, ensure_ascii=False)


def list_supported_cities() -> str:
    """列出所有支持的中文城市"""
    return json.dumps(
        {"cities": list(CITY_MAP.keys()), "count": len(CITY_MAP)},
        ensure_ascii=False,
        indent=2,
    )


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


weather_server.add_tool(get_weather)
weather_server.add_tool(list_supported_cities)
weather_server.add_tool(get_server_info)

if __name__ == "__main__":
    weather_server.run()
