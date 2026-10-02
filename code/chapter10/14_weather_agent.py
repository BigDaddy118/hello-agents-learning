"""在 Agent 中使用天气 MCP 服务器（10.5.1，需 LLM + .env）。"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.tools import MCPTool

load_dotenv()


def create_weather_assistant() -> SimpleAgent:
    assistant = SimpleAgent(
        name="天气助手",
        llm=HelloAgentsLLM(),
        system_prompt=(
            "你是天气助手，可以查询城市天气。\n"
            "使用 get_weather 工具查询天气，支持中文城市名。\n"
        ),
    )
    server_script = os.path.join(os.path.dirname(__file__), "14_weather_mcp_server.py")
    weather_tool = MCPTool(
        server_command=[sys.executable, server_script],
        name="weather",
        prefix="weather_",
    )
    assistant.add_tool(weather_tool)
    return assistant


def demo() -> None:
    assistant = create_weather_assistant()
    print("expanded tools:", assistant.list_tools())
    print("\n查询北京天气：")
    print(assistant.run("北京今天天气怎么样？"))


def interactive() -> None:
    assistant = create_weather_assistant()
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        print(f"助手: {assistant.run(user_input)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        interactive()
