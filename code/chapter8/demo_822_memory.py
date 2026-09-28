"""8.2.2 快速体验：MemoryTool add / search / summary。"""

from __future__ import annotations

from hello_agents import SimpleAgent, ToolRegistry
from hello_agents.tools import MemoryTool


def main() -> None:
    # 与章节一致：可挂到 Agent；本演示直接调 MemoryTool，无需 LLM
    agent = SimpleAgent(
        name="记忆助手",
        llm=_FakeLLM(),  # type: ignore[arg-type]
    )
    memory_tool = MemoryTool(user_id="user123")
    tool_registry = ToolRegistry()
    tool_registry.register_tool(memory_tool)
    agent.tool_registry = tool_registry
    agent.enable_tool_calling = True

    print("=== 添加多个记忆 ===")
    print(
        "记忆1:",
        memory_tool.execute(
            "add",
            content="用户张三是一名Python开发者，专注于机器学习和数据分析",
            memory_type="semantic",
            importance=0.8,
        ),
    )
    print(
        "记忆2:",
        memory_tool.execute(
            "add",
            content="李四是前端工程师，擅长React和Vue.js开发",
            memory_type="semantic",
            importance=0.7,
        ),
    )
    print(
        "记忆3:",
        memory_tool.execute(
            "add",
            content="王五是产品经理，负责用户体验设计和需求分析",
            memory_type="semantic",
            importance=0.6,
        ),
    )

    print("\n=== 搜索特定记忆 ===")
    print("🔍 搜索 '前端工程师':")
    print(memory_tool.execute("search", query="前端工程师", limit=3))

    print("\n=== 记忆摘要 ===")
    print(memory_tool.execute("summary"))


class _FakeLLM:
    provider = "fake"

    def invoke(self, *a, **k):
        return ""


if __name__ == "__main__":
    main()
