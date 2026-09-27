"""多源搜索工具（7.5.3）。"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..base import Tool, ToolParameter


class SearchTool(Tool):
    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
    ):
        super().__init__(
            name="search",
            description="智能网页搜索。支持 hybrid / tavily / serpapi，缺资料或查时事时使用。",
        )
        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.available_backends: list[str] = []
        self.tavily_client = None
        self._setup_backends()

    def _setup_backends(self) -> None:
        if self.tavily_key:
            try:
                from tavily import TavilyClient

                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
            except ImportError:
                pass
        if self.serpapi_key:
            try:
                import serpapi  # noqa: F401

                self.available_backends.append("serpapi")
            except ImportError:
                pass

    def run(self, parameters: dict[str, Any]) -> str:
        query = str(
            parameters.get("input") or parameters.get("query") or ""
        ).strip()
        if not query:
            return "错误：搜索查询不能为空"
        try:
            if self.backend == "tavily":
                return (
                    self._search_tavily(query)
                    if "tavily" in self.available_backends
                    else self._config_hint()
                )
            if self.backend == "serpapi":
                return (
                    self._search_serpapi(query)
                    if "serpapi" in self.available_backends
                    else self._config_hint()
                )
            return self._search_hybrid(query)
        except Exception as e:
            return f"搜索时发生错误: {e}"

    def _search_hybrid(self, query: str) -> str:
        if not self.available_backends:
            return self._config_hint()
        if "tavily" in self.available_backends:
            try:
                return self._search_tavily(query)
            except Exception:
                if "serpapi" in self.available_backends:
                    return self._search_serpapi(query)
        elif "serpapi" in self.available_backends:
            try:
                return self._search_serpapi(query)
            except Exception:
                pass
        return "❌ 所有搜索源都失败了，请检查网络连接和API密钥配置"

    def _search_tavily(self, query: str) -> str:
        assert self.tavily_client is not None
        response = self.tavily_client.search(
            query=query, search_depth="basic", include_answer=True, max_results=3
        )
        result = f"🎯 Tavily AI搜索结果：{response.get('answer', '未找到直接答案')}\n\n"
        for i, item in enumerate(response.get("results", [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {str(item.get('content', ''))[:200]}...\n"
            result += f"    来源: {item.get('url', '')}\n\n"
        return result

    def _search_serpapi(self, query: str) -> str:
        results: dict[str, Any]
        try:
            from serpapi import SerpApiClient

            client = SerpApiClient(
                {
                    "engine": "google",
                    "q": query,
                    "api_key": self.serpapi_key,
                    "gl": "cn",
                    "hl": "zh-cn",
                }
            )
            results = client.get_dict()
        except Exception:
            import serpapi

            search = serpapi.GoogleSearch(
                {"q": query, "api_key": self.serpapi_key, "num": 3}
            )
            results = search.get_dict()

        text = "🔍 SerpApi Google搜索结果：\n\n"
        box = results.get("answer_box") or {}
        if "answer" in box:
            text += f"💡 直接答案：{box['answer']}\n\n"
        organic = results.get("organic_results") or []
        if organic:
            text += "🔗 相关结果：\n"
            for i, res in enumerate(organic[:3], 1):
                text += f"[{i}] {res.get('title', '')}\n"
                text += f"    {res.get('snippet', '')}\n"
                text += f"    来源: {res.get('link', '')}\n\n"
            return text
        return f"对不起，没有找到关于 '{query}' 的信息。"

    def _config_hint(self) -> str:
        return (
            "❌ 没有可用的搜索源。请配置 TAVILY_API_KEY 或 SERPAPI_API_KEY，"
            '并安装: pip install "hello-agents[search]"'
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="搜索查询关键词",
                required=True,
            )
        ]


def search(query: str, backend: str = "hybrid") -> str:
    return SearchTool(backend=backend).run({"input": query})
