"""8.4 Gradio Web 问答助手。默认从 7860 起自动选空闲端口。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from pdf_learning_assistant import PDFLearningAssistant

_assistant: Optional[PDFLearningAssistant] = None
_data_root = Path(tempfile.gettempdir()) / "hello_agents_qa"


def _ensure_assistant(user_id: str = "web_user") -> PDFLearningAssistant:
    global _assistant
    if _assistant is None or _assistant.user_id != user_id:
        root = _data_root / user_id
        _assistant = PDFLearningAssistant(
            user_id=user_id,
            knowledge_base_path=str(root / "kb"),
            memory_data_path=str(root / "mem"),
            report_dir=str(root / "reports"),
            llm=None,
        )
    return _assistant


def init_assistant(user_id: str) -> str:
    a = _ensure_assistant(user_id.strip() or "web_user")
    return f"✅ 助手已初始化\nuser={a.user_id}\nsession={a.session_id}"


def load_doc(file_obj: Any) -> str:
    if file_obj is None:
        return "⚠️ 请先上传文件（PDF / MD / TXT）"
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        return "⚠️ 无法读取上传路径"
    # 沿用已初始化助手，避免默认回到 web_user
    a = _assistant or _ensure_assistant()
    return a.load_document(path).get("message", "")


def ask_question(question: str, advanced: bool) -> str:
    if not question.strip():
        return "⚠️ 请输入问题"
    a = _assistant or _ensure_assistant()
    return a.ask(question.strip(), use_advanced_search=bool(advanced))


def save_note(content: str, concept: str) -> str:
    if not content.strip():
        return "⚠️ 笔记内容为空"
    a = _assistant or _ensure_assistant()
    return a.add_note(content.strip(), concept=concept.strip() or "general")


def recall_mem(query: str) -> str:
    if not query.strip():
        return "⚠️ 请输入回顾关键词"
    a = _assistant or _ensure_assistant()
    return a.recall(query.strip())


def show_stats() -> str:
    a = _assistant or _ensure_assistant()
    s = a.get_stats()
    return "\n".join(f"{k}: {v}" for k, v in s.items())


def make_report() -> str:
    a = _assistant or _ensure_assistant()
    r = a.generate_report(save_to_file=True)
    return (
        f"✅ 报告已生成\n文件: {r.get('report_file')}\n"
        f"指标: {r['learning_metrics']}\n\n{r['memory_summary']}\n\n{r['rag_status']}"
    )


def build_ui():
    import gradio as gr

    with gr.Blocks(title="HelloAgents 文档问答助手") as demo:
        gr.Markdown("# HelloAgents 智能文档问答助手\nMemory + RAG（第八章 8.4）")
        with gr.Row():
            user_in = gr.Textbox(label="用户 ID", value="web_user")
            init_btn = gr.Button("初始化助手")
            init_out = gr.Textbox(label="状态", lines=3)
        init_btn.click(init_assistant, inputs=user_in, outputs=init_out)

        with gr.Tab("加载文档"):
            file_in = gr.File(label="上传 PDF / Markdown / TXT")
            load_btn = gr.Button("加载文档")
            load_out = gr.Textbox(label="结果", lines=4)
            load_btn.click(load_doc, inputs=file_in, outputs=load_out)

        with gr.Tab("智能问答"):
            q_in = gr.Textbox(label="问题", lines=2)
            adv = gr.Checkbox(label="高级检索 (MQE + HyDE)", value=True)
            ask_btn = gr.Button("提问")
            ask_out = gr.Textbox(label="回答", lines=10)
            ask_btn.click(ask_question, inputs=[q_in, adv], outputs=ask_out)

        with gr.Tab("学习笔记"):
            note_concept = gr.Textbox(label="概念标签", value="general")
            note_body = gr.Textbox(label="笔记内容", lines=4)
            note_btn = gr.Button("保存笔记")
            note_out = gr.Textbox(label="结果", lines=2)
            note_btn.click(save_note, inputs=[note_body, note_concept], outputs=note_out)
            recall_q = gr.Textbox(label="回顾关键词")
            recall_btn = gr.Button("回顾学习历程")
            recall_out = gr.Textbox(label="记忆检索", lines=8)
            recall_btn.click(recall_mem, inputs=recall_q, outputs=recall_out)

        with gr.Tab("统计与报告"):
            stats_btn = gr.Button("查看统计")
            stats_out = gr.Textbox(label="学习统计", lines=6)
            report_btn = gr.Button("生成学习报告")
            report_out = gr.Textbox(label="报告", lines=12)
            stats_btn.click(show_stats, outputs=stats_out)
            report_btn.click(make_report, outputs=report_out)
    return demo


def main() -> None:
    try:
        import gradio  # noqa: F401
    except ImportError as e:
        raise SystemExit("需要 gradio：pip install gradio") from e

    import socket

    def free_port(start: int = 7860, span: int = 20) -> int:
        for port in range(start, start + span):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise SystemExit("7860–7879 均被占用，请先结束旧 Gradio 进程")

    port = free_port()
    print(f"启动 Gradio: http://127.0.0.1:{port}/")
    build_ui().launch(server_name="127.0.0.1", server_port=port)


if __name__ == "__main__":
    main()
