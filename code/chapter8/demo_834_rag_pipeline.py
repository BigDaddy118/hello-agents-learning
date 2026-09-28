"""8.3.4：五层管道 — 文档载入 → 分块 → 嵌入 → 检索 → 问答。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hello_agents.tools import RAGTool


def main() -> None:
    kb = Path(tempfile.mkdtemp(prefix="ha_rag834_"))
    doc = kb / "agent_memory.md"
    doc.write_text(
        """# Agent 记忆系统

## 工作记忆
工作记忆保存当前对话上下文，容量有限，会随时间衰减。

## 情景记忆
情景记忆记录带时间戳的具体事件，例如完成项目里程碑。

## 语义记忆
语义记忆存储概念与事实，例如 Python 是解释型语言。

## RAG
RAG（检索增强生成）先检索外部知识，再交给大模型生成答案。
""",
        encoding="utf-8",
    )

    rag = RAGTool(
        knowledge_base_path=str(kb / "kb"),
        collection_name="arch_demo",
        rag_namespace="demo834",
        llm=None,  # 摘录问答，不依赖 API
    )

    print(rag.execute("add_document", file_path=str(doc), chunk_size=80, chunk_overlap=16))
    print()
    print(rag.execute("search", query="情景记忆记录什么", limit=3, min_score=0.05))
    print()
    print(rag.execute("ask", question="什么是 RAG？", limit=2))
    print()
    print(rag.execute("stats"))
    print(f"\n临时目录: {kb}")


if __name__ == "__main__":
    main()
