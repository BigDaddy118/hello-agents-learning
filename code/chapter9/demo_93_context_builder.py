"""9.3 ContextBuilder 自检：GSSC 流水线，不依赖 LLM / 向量库。"""

from __future__ import annotations

from datetime import datetime

from hello_agents.context import ContextBuilder, ContextConfig, ContextPacket
from hello_agents.core.message import Message


def main() -> None:
    builder = ContextBuilder(
        config=ContextConfig(
            max_tokens=800,
            reserve_ratio=0.2,
            min_relevance=0.0,
            enable_compression=True,
        )
    )

    from datetime import timedelta

    t0 = datetime.now() - timedelta(minutes=3)
    history = [
        Message("我正在开发一个数据分析工具", "user", timestamp=t0),
        Message("您计划使用什么技术栈?", "assistant", timestamp=t0 + timedelta(minutes=1)),
        Message(
            "Python和Pandas,已完成CSV读取模块",
            "user",
            timestamp=t0 + timedelta(minutes=2),
        ),
    ]

    evidence = ContextPacket(
        content="Pandas内存优化: 用category代替object; 分块读取大文件。",
        relevance_score=0.9,
        metadata={"type": "rag_result"},
    )

    ctx = builder.build(
        user_query="如何优化Pandas的内存占用?",
        conversation_history=history,
        system_instructions="你是Python数据工程顾问。给出可行建议与代码示例。",
        custom_packets=[evidence],
    )

    assert "[Role & Policies]" in ctx
    assert "[Task]" in ctx
    assert "Pandas" in ctx
    assert "[Evidence]" in ctx
    assert "[Context]" in ctx
    assert "[Output]" in ctx
    print(ctx)
    print("\n[OK] 9.3 ContextBuilder GSSC self-check passed")


if __name__ == "__main__":
    main()
