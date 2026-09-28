"""8.4.2 演示：加载文档 + 情景记忆记录（不依赖真实 PDF/API）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pdf_learning_assistant import PDFLearningAssistant


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="ha_842_"))
    sample = root / "Happy-LLM-demo.md"
    sample.write_text(
        """# Happy-LLM 导读

## 什么是大语言模型
大语言模型（LLM）是一种基于海量文本训练的神经网络，能够理解和生成自然语言。

## Transformer
Transformer 通过自注意力机制建模序列依赖，是现代 LLM 的核心结构。
""",
        encoding="utf-8",
    )

    assistant = PDFLearningAssistant(
        user_id="learner_842",
        knowledge_base_path=str(root / "kb"),
        memory_data_path=str(root / "mem"),
        report_dir=str(root / "reports"),
        llm=None,
    )

    print("session:", assistant.session_id)
    loaded = assistant.load_document(str(sample))
    print("load:", loaded)

    print("\n=== 情景记忆检索 ===")
    print(
        assistant.memory_tool.execute(
            "search",
            query="加载了文档",
            memory_type="episodic",
            limit=3,
        )
    )

    print("\n=== RAG 统计 ===")
    print(assistant.rag_tool.execute("stats"))
    print("\nstats:", {k: v for k, v in assistant.stats.items() if k != "session_start"})
    print("current_document:", assistant.current_document)
    print("临时目录:", root)


if __name__ == "__main__":
    main()
