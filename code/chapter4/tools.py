import os
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import requests
from dotenv import load_dotenv
from serpapi import SerpApiClient
from serpapi.serp_api_client_exception import SerpApiClientException

# 加载项目根目录 .env（chapter4 → code → 项目根）
_ = load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class ToolSpec(TypedDict):
    description: str
    func: Callable[..., str]


def _text_lines(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    lines: list[str] = []
    for item in cast(list[object], value):
        if isinstance(item, str):
            lines.append(item)
    return lines


def _text_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    mapped: dict[str, str] = {}
    for key, item in cast(dict[object, object], value).items():
        if isinstance(key, str) and isinstance(item, str):
            mapped[key] = item
    return mapped


def _result_rows(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, str]] = []
    for item in cast(list[object], value):
        row = _text_map(item)
        if row:
            rows.append(row)
    return rows


def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析:优先寻找最直接的答案
        answer_lines = _text_lines(results.get("answer_box_list"))
        if answer_lines:
            return "\n".join(answer_lines)
        answer = _text_map(results.get("answer_box")).get("answer")
        if answer:
            return answer
        description = _text_map(results.get("knowledge_graph")).get("description")
        if description:
            return description
        snippets: list[str] = []
        for index, item in enumerate(_result_rows(results.get("organic_results"))[:3]):
            snippets.append(f"[{index + 1}] {item.get('title', '')}\n{item.get('snippet', '')}")
        if snippets:
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except (SerpApiClientException, requests.RequestException, ValueError, TypeError) as e:
        return f"搜索时发生错误: {e}"


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """

    def __init__(self) -> None:
        self.tools: dict[str, ToolSpec] = {}

    def registerTool(self, name: str, description: str, func: Callable[..., str]) -> None:
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> Callable[..., str] | None:
        """
        根据名称获取一个工具的执行函数。
        """
        info = self.tools.get(name)
        if info is None:
            return None
        return info["func"]

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join(
            [
                f"- {name}: {info['description']}"
                for name, info in self.tools.items()
            ]
        )


# --- 工具初始化与使用示例 ---
if __name__ == "__main__":
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = (
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    )
    toolExecutor.registerTool("Search", search_description, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")
