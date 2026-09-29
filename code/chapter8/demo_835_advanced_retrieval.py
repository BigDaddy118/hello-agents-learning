"""8.3.5：MQE / HyDE / 扩展检索。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hello_agents.memory.rag.retrieval import prompt_hyde, prompt_mqe
from hello_agents.tools import RAGTool


class _StubLLM:
    """演示用：模拟 MQE / HyDE 输出，不调真实 API。"""

    def invoke(self, messages):
        sys_c = (messages[0].get("content") or "") if messages else ""
        user = (messages[-1].get("content") or "") if messages else ""
        if "查询扩展" in sys_c:
            return "Python语言发展历史\nGuido van Rossum 与 Python 发布\nPython 设计哲学"
        if "答案性段落" in sys_c or "假设" in sys_c:
            return (
                "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。"
                "其设计强调可读性与简洁语法。"
            )
        return user


def main() -> None:
    kb = Path(tempfile.mkdtemp(prefix="ha_rag835_"))
    rag = RAGTool(
        knowledge_base_path=str(kb),
        collection_name="adv",
        rag_namespace="demo835",
        llm=_StubLLM(),
    )

    rag.execute(
        "add_text",
        text=(
            "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。"
            "Python的设计哲学强调代码的可读性和简洁的语法。"
        ),
        document_id="python_intro",
    )
    rag.execute(
        "add_text",
        text=(
            "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。"
            "主要包括监督学习、无监督学习和强化学习三种类型。"
        ),
        document_id="ml_basics",
    )
    rag.execute(
        "add_text",
        text=(
            "RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术。"
            "它通过检索相关知识来增强大语言模型的生成能力。"
        ),
        document_id="rag_concept",
    )

    q = "Python是谁发明的，什么时候发布的？"
    print("=== MQE 扩展 ===")
    print(prompt_mqe(q, 3, llm=_StubLLM()))
    print("\n=== HyDE 假设文档 ===")
    hyde = prompt_hyde(q, llm=_StubLLM()) or ""
    print(hyde[:80], "...")

    print("\n=== 基础检索 ===")
    print(rag.execute("search", query=q, limit=3, min_score=0.05))

    print("\n=== MQE+HyDE 扩展检索 ===")
    print(
        rag.execute(
            "search",
            query=q,
            limit=3,
            min_score=0.05,
            enable_mqe=True,
            enable_hyde=True,
            mqe_expansions=2,
        )
    )

    print("\n=== ask(MQE) ===")
    print(rag.execute("ask", question=q, enable_mqe=True, limit=2))
    print(f"\n临时库: {kb}")


if __name__ == "__main__":
    main()
