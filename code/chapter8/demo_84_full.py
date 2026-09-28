"""8.4 完整 CLI：加载 → 问答 → 笔记 → 回顾 → 统计 → 报告。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pdf_learning_assistant import PDFLearningAssistant


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="ha_84_"))
    sample = root / "Happy-LLM-demo.md"
    sample.write_text(
        """# Happy-LLM 导读

## 什么是大语言模型
大语言模型（LLM）是一种基于海量文本训练的神经网络，能够理解和生成自然语言。

## Transformer
Transformer 通过自注意力机制建模序列依赖，是现代 LLM 的核心结构。

## RAG
检索增强生成先从知识库检索相关片段，再交给大模型生成答案。
""",
        encoding="utf-8",
    )

    assistant = PDFLearningAssistant(
        user_id="learner_84",
        knowledge_base_path=str(root / "kb"),
        memory_data_path=str(root / "mem"),
        report_dir=str(root / "reports"),
        llm=None,
    )

    print("1) 加载", assistant.load_document(str(sample)))
    print("\n2) 问答", assistant.ask("什么是大语言模型？", use_advanced_search=True))
    print("\n3) 笔记", assistant.add_note("LLM = 大语言模型", concept="LLM"))
    print("\n4) 回顾", assistant.recall("文档"))
    print("\n5) 统计", assistant.get_stats())
    report = assistant.generate_report(save_to_file=True)
    print("\n6) 报告文件:", report.get("report_file"))
    print("临时目录:", root)


if __name__ == "__main__":
    main()
