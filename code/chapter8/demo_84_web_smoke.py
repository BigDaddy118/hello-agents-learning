"""Web 层回调冒烟测试（不启 HTTP，直接调 qa_assistant_app 函数）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import qa_assistant_app as app


def main() -> None:
    print(app.init_assistant("smoke_user"))
    sample = Path(tempfile.gettempdir()) / "ha_smoke_doc.md"
    sample.write_text(
        "# 测试\n\nGradio 冒烟：RAG 与 Memory 应能工作。\n",
        encoding="utf-8",
    )
    print("load:", app.load_doc(str(sample)))
    print("ask:", app.ask_question("什么是 Gradio 冒烟？", True)[:120], "...")
    print("note:", app.save_note("冒烟通过", "test"))
    print("recall:", app.recall_mem("文档")[:120], "...")
    print("stats:\n", app.show_stats())
    print("report:\n", app.make_report()[:200], "...")
    print("OK")


if __name__ == "__main__":
    main()
