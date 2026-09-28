"""8.2.3 MemoryTool 完整操作演示（对齐官方 01_MemoryTool_Basic_Operations）。"""

from __future__ import annotations

from hello_agents.tools import MemoryTool


def main() -> None:
    memory_tool = MemoryTool(
        user_id="demo_user",
        memory_types=["working", "episodic", "semantic", "perceptual"],
    )
    print("✅ MemoryTool初始化完成")
    print(
        "支持: add, search, summary, stats, update, remove, forget, consolidate, clear_all"
    )

    print("\n📝 添加四种记忆")
    print(
        memory_tool.run(
            {
                "action": "add",
                "content": "正在学习HelloAgents框架的记忆系统",
                "memory_type": "working",
                "importance": 0.7,
                "task_type": "learning",
            }
        )
    )
    print(
        memory_tool.run(
            {
                "action": "add",
                "content": "2024年开始深入研究AI Agent技术",
                "memory_type": "episodic",
                "importance": 0.8,
                "event_type": "milestone",
            }
        )
    )
    print(
        memory_tool.run(
            {
                "action": "add",
                "content": "记忆系统包括工作记忆、情景记忆、语义记忆和感知记忆四种类型",
                "memory_type": "semantic",
                "importance": 0.9,
                "concept": "memory_types",
            }
        )
    )
    print(
        memory_tool.run(
            {
                "action": "add",
                "content": "查看了记忆系统的架构图和实现代码",
                "memory_type": "perceptual",
                "importance": 0.6,
                "modality": "document",
            }
        )
    )

    print("\n🔍 搜索")
    print(memory_tool.run({"action": "search", "query": "记忆系统", "limit": 3}))
    print(
        memory_tool.run(
            {
                "action": "search",
                "query": "记忆",
                "memory_type": "semantic",
                "limit": 2,
            }
        )
    )

    print("\n📋 摘要 / 统计")
    print(memory_tool.run({"action": "summary", "limit": 5}))
    print(memory_tool.run({"action": "stats"}))

    print("\n⚙️ 遗忘 / 整合")
    memory_tool.run(
        {
            "action": "add",
            "content": "这是一个临时的测试记忆，重要性很低",
            "memory_type": "working",
            "importance": 0.1,
        }
    )
    print(
        memory_tool.run(
            {"action": "forget", "strategy": "importance_based", "threshold": 0.2}
        )
    )
    print(
        memory_tool.run(
            {
                "action": "consolidate",
                "from_type": "working",
                "to_type": "episodic",
                "importance_threshold": 0.6,
            }
        )
    )
    print(memory_tool.run({"action": "stats"}))
    print("\n🎉 8.2.3 演示完成")


if __name__ == "__main__":
    main()
