"""8.4：PDFLearningAssistant — MemoryTool + RAGTool 学习助手。"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hello_agents.memory import MemoryConfig
from hello_agents.tools import MemoryTool, RAGTool


class PDFLearningAssistant:
    """智能文档问答助手。"""

    def __init__(
        self,
        user_id: str = "default_user",
        knowledge_base_path: str = "./knowledge_base",
        memory_data_path: Optional[str] = None,
        llm: Any = None,
        report_dir: str = ".",
    ):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        mem_cfg = MemoryConfig()
        if memory_data_path:
            Path(memory_data_path).mkdir(parents=True, exist_ok=True)
            mem_cfg.database_path = str(Path(memory_data_path) / "memory.db")

        self.memory_tool = MemoryTool(
            user_id=user_id,
            memory_config=mem_cfg,
            memory_types=["working", "episodic", "semantic", "perceptual"],
        )
        self.memory_tool.current_session_id = self.session_id

        self.rag_tool = RAGTool(
            knowledge_base_path=knowledge_base_path,
            rag_namespace=f"pdf_{user_id}",
            collection_name=f"pdf_{user_id}",
            llm=llm,
        )

        self.stats: dict[str, Any] = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
        }
        self.current_document: Optional[str] = None

    def load_document(self, pdf_path: str) -> dict[str, Any]:
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"文件不存在: {pdf_path}"}

        start_time = time.time()
        result = self.rag_tool.execute(
            "add_document",
            file_path=pdf_path,
            chunk_size=1000,
            chunk_overlap=200,
        )
        process_time = time.time() - start_time
        ok = isinstance(result, str) and result.startswith("✅")
        if ok:
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1
            self.memory_tool.execute(
                "add",
                content=f"加载了文档《{self.current_document}》",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id,
            )
            return {
                "success": True,
                "message": f"加载成功！(耗时: {process_time:.1f}秒)",
                "document": self.current_document,
                "detail": result,
            }
        return {"success": False, "message": f"加载失败: {result}"}

    def ask(self, question: str, use_advanced_search: bool = True) -> str:
        if not self.current_document:
            return "⚠️ 请先加载文档！"

        self.memory_tool.execute(
            "add",
            content=f"提问: {question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id,
        )
        answer = self.rag_tool.execute(
            "ask",
            question=question,
            limit=5,
            enable_mqe=use_advanced_search,
            enable_hyde=use_advanced_search,
        )
        self.memory_tool.execute(
            "add",
            content=f"关于'{question}'的学习",
            memory_type="episodic",
            importance=0.7,
            event_type="qa_interaction",
            session_id=self.session_id,
        )
        self.stats["questions_asked"] += 1
        return answer

    def add_note(self, content: str, concept: Optional[str] = None) -> str:
        msg = self.memory_tool.execute(
            "add",
            content=content,
            memory_type="semantic",
            importance=0.8,
            concept=concept or "general",
            session_id=self.session_id,
        )
        self.stats["concepts_learned"] += 1
        return msg

    def recall(self, query: str, limit: int = 5) -> str:
        return self.memory_tool.execute("search", query=query, limit=limit)

    def get_stats(self) -> dict[str, Any]:
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "学习笔记": self.stats["concepts_learned"],
            "当前文档": self.current_document or "未加载",
        }

    def generate_report(self, save_to_file: bool = True) -> dict[str, Any]:
        memory_summary = self.memory_tool.execute("summary", limit=10)
        rag_stats = self.rag_tool.execute("stats")
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        report: dict[str, Any] = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration,
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"],
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats,
        }
        if save_to_file:
            report_file = self.report_dir / f"learning_report_{self.session_id}.json"
            report_file.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            report["report_file"] = str(report_file)
        return report
