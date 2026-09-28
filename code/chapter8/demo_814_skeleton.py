"""8.1.4：确认记忆/RAG 目录骨架与工具占位可导入。

完整 MemoryTool / RAGTool 逻辑分别在 8.2、8.3 实现后再跑快速体验对话。
"""

from __future__ import annotations

from pathlib import Path

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool


def check_tree() -> None:
    root = Path(__file__).resolve().parents[2] / "hello_agents" / "memory"
    expected = [
        "base.py",
        "manager.py",
        "embedding.py",
        "types/working.py",
        "types/episodic.py",
        "types/semantic.py",
        "types/perceptual.py",
        "storage/qdrant_store.py",
        "storage/neo4j_store.py",
        "storage/document_store.py",
        "rag/pipeline.py",
        "rag/document.py",
    ]
    missing = [p for p in expected if not (root / p).exists()]
    assert not missing, f"缺少文件: {missing}"
    print(f"✅ memory 目录完整 ({len(expected)} 个目标文件)")


def check_tools() -> None:
    memory = MemoryTool(user_id="user123")
    rag = RAGTool(knowledge_base_path="./knowledge_base")
    reg = ToolRegistry()
    reg.register_tool(memory)
    reg.register_tool(rag)
    assert "memory" in reg.list_tools() and "rag" in reg.list_tools()
    print(f"✅ 工具已注册: {reg.list_tools()}")
    print(f"   描述:\n{reg.get_tools_description()}")


def check_agent_wiring() -> None:
    # 不调用 LLM；只验证章节示例里的挂载方式
    class _Fake:
        provider = "fake"

        def invoke(self, *a, **k):
            return "ok"

    agent = SimpleAgent(name="智能助手", llm=_Fake())  # type: ignore[arg-type]
    reg = ToolRegistry()
    reg.register_tool(MemoryTool(user_id="user123"))
    reg.register_tool(RAGTool(knowledge_base_path="./knowledge_base"))
    agent.tool_registry = reg
    agent.enable_tool_calling = True
    assert agent.has_tools()
    print(f"✅ Agent 已挂载工具: {agent.list_tools()}")


if __name__ == "__main__":
    check_tree()
    check_tools()
    check_agent_wiring()
    print(
        "\n📌 8.1.4 骨架就绪。"
        "配置见仓库根目录 .env.example；"
        "Memory/RAG 实现请继续 8.2 / 8.3。"
    )
