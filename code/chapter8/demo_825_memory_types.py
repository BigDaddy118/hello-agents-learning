"""8.2.5 四种记忆类型：评分差异与持久化冒烟。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hello_agents.memory import MemoryConfig, MemoryManager
from hello_agents.tools import MemoryTool


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ha_mem_"))
    cfg = MemoryConfig(database_path=str(tmp / "memory.db"), working_memory_ttl=60)
    tool = MemoryTool(
        user_id="u825",
        memory_config=cfg,
        memory_types=["working", "episodic", "semantic", "perceptual"],
    )

    tool.execute(
        "add",
        content="今天上午讨论了 Python 异步编程",
        memory_type="working",
        importance=0.6,
    )
    tool.execute(
        "add",
        content="2024-03-15 完成了第一个 Agent 项目里程碑",
        memory_type="episodic",
        importance=0.85,
        event_type="milestone",
    )
    tool.execute(
        "add",
        content="Python 是解释型面向对象语言；Agent 需要 Memory 与 RAG",
        memory_type="semantic",
        importance=0.9,
    )
    tool.execute(
        "add",
        content="用户上传了一张 Python 代码截图，含 async 函数定义",
        memory_type="perceptual",
        importance=0.7,
        modality="image",
        file_path="./uploads/code.png",
    )

    print("=== working 检索 Python ===")
    print(tool.execute("search", query="Python", memory_type="working", limit=3))

    print("\n=== episodic 检索 项目 ===")
    print(tool.execute("search", query="项目里程碑", memory_type="episodic", limit=3))

    print("\n=== semantic 检索 Agent Memory ===")
    print(tool.execute("search", query="Agent Memory", memory_type="semantic", limit=3))

    print("\n=== perceptual 检索 截图 ===")
    print(
        tool.execute(
            "search",
            query="代码截图",
            memory_type="perceptual",
            limit=3,
        )
    )

    print("\n=== stats ===")
    print(tool.execute("stats"))

    # 情景/感知落盘后可再开一个 Manager 读到
    mm2 = MemoryManager(
        config=cfg,
        user_id="u825",
        enable_working=False,
        enable_episodic=True,
        enable_semantic=False,
        enable_perceptual=True,
    )
    epi = mm2.retrieve_memories("里程碑", limit=2, memory_types=["episodic"])
    assert epi, "episodic 应能从 SQLite 恢复"
    print(f"\n✅ SQLite 恢复 episodic: {epi[0].content[:40]}...")
    print(f"✅ 临时库: {tmp}")


if __name__ == "__main__":
    main()
