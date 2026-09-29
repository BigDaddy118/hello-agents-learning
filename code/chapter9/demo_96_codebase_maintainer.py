"""9.6 CodebaseMaintainer 自检：探索/分析/规划 + 笔记/报告（Stub LLM，无 API）。"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from hello_agents.agents.codebase_maintainer import CodebaseMaintainer


class _StubLLM:
    def invoke(self, messages, **kwargs):
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        bits = []
        if "[代码库结构]" in system or "[代码文件]" in system:
            bits.append("已扫描到若干 Python 文件。")
        if "[待办/标记]" in system or "TODO" in system:
            bits.append("发现 TODO/FIXME 标记，建议清理或转成正式任务。")
        if "[笔记:" in system or "[Evidence]" in system:
            bits.append("已结合历史笔记作答。")
        if "规划" in user or "下一步" in user:
            bits.append("建议优先处理 blocker，再重构高复杂度模块。")
        if "问题" in user or "分析" in user:
            bits.append("存在代码质量问题，建议补充测试。")
        if not bits:
            bits.append("基于当前上下文给出维护建议。")
        return " ".join(bits)


def _seed_codebase(root: Path) -> None:
    app = root / "app"
    (app / "models").mkdir(parents=True)
    (app / "services").mkdir(parents=True)
    (app / "models" / "user.py").write_text(
        "class User:\n    # TODO: add email unique constraint\n    pass\n",
        encoding="utf-8",
    )
    (app / "services" / "order_service.py").write_text(
        "def process_order(order_id):\n"
        "    # FIXME: nested logic too deep\n"
        "    if order_id:\n"
        "        return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# demo flask-like app\n", encoding="utf-8")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="maintainer_"))
    notes = root / "notes"
    reports = root / "reports"
    try:
        _seed_codebase(root / "codebase")
        m = CodebaseMaintainer(
            project_name="demo_app",
            codebase_path=str(root / "codebase"),
            llm=_StubLLM(),
            notes_workspace=str(notes),
        )

        r1 = m.explore()
        assert "Python" in r1 or "扫描" in r1 or "建议" in r1

        r2 = m.analyze()
        assert "TODO" in r2 or "问题" in r2 or "质量" in r2

        m.create_note(
            title="本周重构计划",
            content="优先修复 User.email 唯一约束与 process_order 嵌套。",
            note_type="task_state",
            tags=["week1"],
        )
        r3 = m.plan_next_steps()
        assert len(r3) > 0

        listed = m.execute_command("dir" if __import__("os").name == "nt" else "ls")
        assert "README" in listed or "app" in listed.lower() or "目录" in listed

        report = m.generate_report(save_to_file=True, report_dir=str(reports))
        assert report["activity"]["commands_executed"] >= 1
        assert report["notes"]["total_notes"] >= 1
        assert Path(report["report_file"]).exists()

        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print("\n[OK] 9.6 CodebaseMaintainer self-check passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
