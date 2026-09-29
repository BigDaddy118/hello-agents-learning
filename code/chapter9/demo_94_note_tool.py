"""9.4 NoteTool 自检：CRUD / search / summary，并注入 ContextBuilder。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hello_agents.context import ContextBuilder, ContextConfig, ContextPacket
from hello_agents.tools import NoteTool


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="note_tool_"))
    try:
        notes = NoteTool(workspace=str(root / "project_notes"))

        nid = notes.create_note(
            title="重构项目 - 第一阶段",
            content="已完成数据模型层重构，测试覆盖率 85%。下一步重构业务逻辑层。",
            note_type="task_state",
            tags=["refactoring", "phase1"],
        )
        assert nid.startswith("note_")

        blocker = notes.run(
            {
                "action": "create",
                "title": "依赖冲突问题",
                "content": "第三方库版本不兼容，影响业务逻辑层 3 个模块。",
                "note_type": "blocker",
                "tags": ["dependency", "urgent"],
            }
        )
        assert "创建成功" in blocker

        detail = notes.run({"action": "read", "note_id": nid})
        assert "数据模型层" in detail

        updated = notes.run(
            {
                "action": "update",
                "note_id": nid,
                "content": "第一阶段完成；覆盖率 85%；准备进入第二阶段。",
            }
        )
        assert "已更新" in updated

        found = notes.search_notes("依赖冲突", limit=5)
        assert len(found) >= 1
        assert found[0]["type"] == "blocker"

        listed = notes.list_notes(note_type="task_state", limit=5)
        assert any(n["id"] == nid for n in listed)

        summary = notes.summary_dict()
        assert summary["total_notes"] == 2
        assert summary["type_distribution"].get("blocker") == 1

        # 与 ContextBuilder 集成：笔记 → ContextPacket
        builder = ContextBuilder(config=ContextConfig(max_tokens=1500, min_relevance=0.0))
        packets = [
            ContextPacket(
                content=f"[笔记:{n['title']}]\n{n['content']}",
                relevance_score=0.9 if n["type"] == "blocker" else 0.75,
                metadata={"type": "note", "note_type": n["type"]},
            )
            for n in notes.search_notes("重构", limit=3)
        ]
        # note 类型走 Context 区：先标成 memory 便于进 Evidence，或保持 note
        for p in packets:
            p.metadata["type"] = "memory"

        ctx = builder.build(
            user_query="业务逻辑层依赖冲突怎么处理？",
            system_instructions="你是项目长期助手，优先关注 blocker。",
            custom_packets=packets,
        )
        assert "[Evidence]" in ctx
        assert "依赖" in ctx or "冲突" in ctx

        deleted = notes.run({"action": "delete", "note_id": nid})
        assert "已删除" in deleted
        assert notes.summary_dict()["total_notes"] == 1

        print(notes.run({"action": "summary"}))
        print("\n[OK] 9.4 NoteTool self-check passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
