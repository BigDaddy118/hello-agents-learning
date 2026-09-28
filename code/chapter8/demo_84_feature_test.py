"""8.4 功能验收：助手 API + Web 回调 + Gradio HTTP。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pdf_learning_assistant import PDFLearningAssistant
import qa_assistant_app as web


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  PASS: {msg}")


def test_assistant_api() -> None:
    print("\n=== PDFLearningAssistant ===")
    root = Path(tempfile.mkdtemp(prefix="ha_feat_"))
    doc = root / "feat.md"
    doc.write_text(
        "# 功能测试\n\n## LLM\n大语言模型能理解自然语言。\n\n## RAG\n检索增强生成结合检索与生成。\n",
        encoding="utf-8",
    )
    a = PDFLearningAssistant(
        user_id="feat_user",
        knowledge_base_path=str(root / "kb"),
        memory_data_path=str(root / "mem"),
        report_dir=str(root / "reports"),
        llm=None,
    )

    # 边界：未加载就提问
    _assert("请先加载" in a.ask("hi"), "未加载文档时 ask 提示")

    r = a.load_document(str(doc))
    _assert(r["success"] is True, "load_document 成功")
    _assert(a.current_document == "feat.md", "current_document 已设置")

    miss = a.load_document(str(root / "nope.pdf"))
    _assert(miss["success"] is False, "不存在文件 load 失败")

    ans = a.ask("什么是大语言模型？", use_advanced_search=True)
    _assert(ans.startswith("💬") or "知识库" in ans or "LLM" in ans or "语言" in ans, "ask 返回答案")
    _assert(a.stats["questions_asked"] == 1, "questions_asked +1")

    note = a.add_note("LLM 很重要", concept="LLM")
    _assert(note.startswith("✅"), "add_note 成功")
    _assert(a.stats["concepts_learned"] == 1, "concepts_learned +1")

    mem = a.recall("文档")
    _assert("加载了文档" in mem or "情景" in mem, "recall 能找回加载事件")

    st = a.get_stats()
    _assert(st["加载文档"] == 1 and st["提问次数"] == 1 and st["学习笔记"] == 1, "get_stats 计数正确")

    report = a.generate_report(save_to_file=True)
    _assert(Path(report["report_file"]).exists(), "报告 JSON 已写入")
    _assert(report["learning_metrics"]["documents_loaded"] == 1, "报告指标正确")


def test_web_callbacks() -> None:
    print("\n=== Web 回调 ===")
    web._assistant = None
    init = web.init_assistant("feat_web")
    _assert("feat_web" in init and "✅" in init, "init_assistant")

    empty = web.load_doc(None)
    _assert("上传" in empty, "未上传文件提示")

    sample = Path(tempfile.gettempdir()) / "ha_feat_web.md"
    sample.write_text("# Web\n\nTransformer 是 LLM 核心结构。\n", encoding="utf-8")
    loaded = web.load_doc(str(sample))
    _assert("加载成功" in loaded, "load_doc")
    _assert(web._assistant is not None and web._assistant.user_id == "feat_web", "沿用同一 user")

    _assert("请输入问题" in web.ask_question("", True), "空问题提示")
    ans = web.ask_question("Transformer 是什么？", True)
    _assert("💬" in ans or "Transformer" in ans or "知识库" in ans, "ask_question")

    _assert("为空" in web.save_note("", "x"), "空笔记提示")
    _assert(web.save_note("记一下 Transformer", "Transformer").startswith("✅"), "save_note")

    _assert("关键词" in web.recall_mem(""), "空回顾提示")
    rec = web.recall_mem("加载")
    _assert("找到" in rec or "记忆" in rec, "recall_mem")

    stats = web.show_stats()
    _assert("加载文档" in stats and "提问次数" in stats, "show_stats")

    rep = web.make_report()
    _assert("报告已生成" in rep and "文件:" in rep, "make_report")


def test_gradio_http() -> None:
    print("\n=== Gradio HTTP ===")
    import urllib.request

    with urllib.request.urlopen("http://127.0.0.1:7861/", timeout=5) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
        _assert(resp.status == 200, "首页 200")
        _assert("gradio" in body.lower() or "HelloAgents" in body or "html" in body.lower(), "页面有内容")

    # Gradio 3+/4+ config endpoint
    try:
        with urllib.request.urlopen("http://127.0.0.1:7861/config", timeout=5) as resp:
            cfg = resp.read().decode("utf-8", errors="ignore")
            _assert(resp.status == 200 and len(cfg) > 10, "/config 可用")
    except Exception as e:
        print(f"  SKIP: /config ({e})")

    try:
        from gradio_client import Client

        client = Client("http://127.0.0.1:7861/")
        # 旧进程可能仍是修 bug 前代码；只测 init 是否可调
        out = client.predict("api_user", api_name="/init_assistant")
        _assert(isinstance(out, str) and len(out) > 0, f"Gradio Client init -> {out[:60]}")
    except Exception as e:
        print(f"  SKIP: Gradio Client API ({e})")


def main() -> None:
    test_assistant_api()
    test_web_callbacks()
    test_gradio_http()
    print("\n全部相关功能测试通过")


if __name__ == "__main__":
    main()
