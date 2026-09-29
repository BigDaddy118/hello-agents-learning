"""CodebaseMaintainer — 长程代码库维护助手（9.6）。

整合 ContextBuilder + NoteTool + TerminalTool + MemoryTool。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..context import ContextBuilder, ContextConfig, ContextPacket
from ..core.message import Message
from ..tools.builtin.memory_tool import MemoryTool
from ..tools.builtin.note_tool import NoteTool
from ..tools.builtin.terminal_tool import TerminalTool


def _now() -> datetime:
    return datetime.now()  # noqa: DTZ005


class CodebaseMaintainer:
    """跨会话代码库维护助手。"""

    def __init__(
        self,
        project_name: str,
        codebase_path: str,
        llm: Any = None,
        notes_workspace: str | None = None,
    ):
        self.project_name = project_name
        self.codebase_path = str(Path(codebase_path).resolve())
        self.session_id = f"session_{_now().strftime('%Y%m%d_%H%M%S')}"
        self.llm = llm if llm is not None else self._default_llm()

        notes_dir = notes_workspace or f"./{project_name}_notes"
        self.memory_tool = MemoryTool(
            user_id=project_name, memory_types=["working"]
        )
        self.note_tool = NoteTool(workspace=notes_dir)
        self.terminal_tool = TerminalTool(workspace=self.codebase_path, timeout=60)

        self.context_builder = ContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=None,
            config=ContextConfig(
                max_tokens=4000,
                reserve_ratio=0.15,
                min_relevance=0.0,
                enable_compression=True,
            ),
        )
        self.conversation_history: list[Message] = []
        self.stats: dict[str, Any] = {
            "session_start": _now(),
            "commands_executed": 0,
            "notes_created": 0,
            "issues_found": 0,
        }
        print(f"✅ 代码库维护助手已初始化: {project_name}")
        print(f"📁 工作目录: {self.codebase_path}")
        print(f"🆔 会话ID: {self.session_id}")

    @staticmethod
    def _default_llm() -> Any:
        from ..core.llm import HelloAgentsLLM

        return HelloAgentsLLM()

    def run(self, user_input: str, mode: str = "auto") -> str:
        print(f"\n{'=' * 80}")
        print(f"👤 用户: {user_input}")
        print(f"{'=' * 80}\n")

        pre = self._preprocess_by_mode(user_input, mode)
        notes = self._retrieve_relevant_notes(user_input)
        note_packets = self._notes_to_packets(notes)

        context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.conversation_history,
            system_instructions=self._build_system_instructions(mode),
            custom_packets=note_packets + pre,
        )

        print("🤖 正在思考...")
        response = self.llm.invoke(
            [
                {"role": "system", "content": context},
                {"role": "user", "content": user_input},
            ]
        )
        self._postprocess_response(user_input, response)
        self._update_history(user_input, response)

        print(f"\n🤖 助手: {response}\n")
        print(f"{'=' * 80}\n")
        return response

    def _preprocess_by_mode(
        self, user_input: str, mode: str
    ) -> list[ContextPacket]:
        packets: list[ContextPacket] = []
        if mode in ("explore", "auto"):
            print("🔍 探索代码库结构...")
            structure = self._run_term(self._cmd_list_py())
            packets.append(
                ContextPacket(
                    content=f"[代码库结构]\n{structure}",
                    relevance_score=0.6,
                    metadata={"type": "code_structure", "source": "terminal"},
                )
            )
        if mode == "analyze":
            print("📊 分析代码质量...")
            loc = self._run_term(self._cmd_list_py())
            todos = self._run_term(self._cmd_find_todos())
            packets.append(
                ContextPacket(
                    content=f"[代码文件]\n{loc}\n\n[待办/标记]\n{todos}",
                    relevance_score=0.7,
                    metadata={"type": "code_analysis", "source": "terminal"},
                )
            )
        if mode == "plan":
            print("📋 加载任务规划...")
            task_notes = self.note_tool.list_notes(note_type="task_state", limit=3)
            if task_notes:
                content = "\n".join(f"- {n['title']}" for n in task_notes)
                packets.append(
                    ContextPacket(
                        content=f"[当前任务]\n{content}",
                        relevance_score=0.8,
                        metadata={"type": "task_plan", "source": "notes"},
                    )
                )
        return packets

    def _cmd_list_py(self) -> str:
        if os.name == "nt":
            return "dir /s /b *.py"
        return 'find . -type f -name "*.py"'

    def _cmd_find_todos(self) -> str:
        if os.name == "nt":
            return 'findstr /s /n /i "TODO FIXME" *.py'
        return "grep -rn 'TODO\\|FIXME' --include='*.py' ."

    def _run_term(self, command: str) -> str:
        out = self.terminal_tool.run({"command": command})
        self.stats["commands_executed"] += 1
        return out

    def _retrieve_relevant_notes(
        self, query: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        try:
            blockers = self.note_tool.list_notes(note_type="blocker", limit=2)
            # list 无正文，补全 content 供 Context
            enriched: list[dict[str, Any]] = []
            for meta in blockers:
                try:
                    enriched.append(self.note_tool.get_note(meta["id"]))
                except (OSError, ValueError, KeyError):
                    continue
            searched = self.note_tool.search_notes(query, limit=limit)
            merged: dict[str, dict[str, Any]] = {}
            for n in enriched + searched:
                nid = n.get("note_id") or n.get("id")
                if nid:
                    merged[str(nid)] = n
            return list(merged.values())[:limit]
        except (OSError, ValueError, TypeError) as e:
            print(f"[WARNING] 笔记检索失败: {e}")
            return []

    def _notes_to_packets(self, notes: list[dict[str, Any]]) -> list[ContextPacket]:
        relevance_map = {
            "blocker": 0.9,
            "action": 0.8,
            "task_state": 0.75,
            "conclusion": 0.7,
        }
        packets: list[ContextPacket] = []
        for note in notes:
            ntype = note.get("type", "general")
            content = (
                f"[笔记:{note.get('title', 'Untitled')}]\n"
                f"类型: {ntype}\n\n{note.get('content', '')}"
            )
            ts_raw = note.get("updated_at")
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else _now()
            except (TypeError, ValueError):
                ts = _now()
            packets.append(
                ContextPacket(
                    content=content,
                    timestamp=ts,
                    relevance_score=relevance_map.get(str(ntype), 0.6),
                    metadata={
                        "type": "memory",  # 进入 Evidence 分区
                        "note_type": ntype,
                        "note_id": note.get("note_id") or note.get("id"),
                    },
                )
            )
        return packets

    def _build_system_instructions(self, mode: str) -> str:
        base = f"""你是 {self.project_name} 项目的代码库维护助手。

你的核心能力:
1. 结合终端探索结果理解代码库结构
2. 结合笔记追踪 blocker / action / task_state
3. 给出可执行的维护与重构建议

当前会话ID: {self.session_id}
工作目录: {self.codebase_path}
"""
        hints = {
            "explore": "\n当前模式: 探索 — 总结结构、关键模块与建议下一步。",
            "analyze": "\n当前模式: 分析 — 指出重复、复杂度、TODO/缺少测试等问题。",
            "plan": "\n当前模式: 规划 — 综合笔记给出优先级任务清单。",
            "auto": "\n当前模式: 自动 — 按用户问题灵活回答。",
        }
        return base + hints.get(mode, hints["auto"])

    def _postprocess_response(self, user_input: str, response: str) -> None:
        low = response.lower()
        if any(k in response for k in ("问题", "bug", "错误", "阻塞")) or "bug" in low:
            try:
                self.note_tool.create_note(
                    title=f"发现问题: {user_input[:30]}...",
                    content=f"## 用户输入\n{user_input}\n\n## 问题分析\n{response[:500]}...",
                    note_type="blocker",
                    tags=[self.project_name, "auto_detected", self.session_id],
                )
                self.stats["notes_created"] += 1
                self.stats["issues_found"] += 1
                print("📝 已自动创建问题笔记")
            except ValueError as e:
                print(f"[WARNING] 创建笔记失败: {e}")
        elif any(k in user_input for k in ("计划", "下一步", "任务", "todo", "TODO")):
            try:
                self.note_tool.create_note(
                    title=f"任务规划: {user_input[:30]}...",
                    content=f"## 讨论\n{user_input}\n\n## 行动计划\n{response[:500]}...",
                    note_type="action",
                    tags=[self.project_name, "planning", self.session_id],
                )
                self.stats["notes_created"] += 1
                print("📝 已自动创建行动计划笔记")
            except ValueError as e:
                print(f"[WARNING] 创建笔记失败: {e}")

    def _update_history(self, user_input: str, response: str) -> None:
        self.conversation_history.append(
            Message(user_input, "user", timestamp=_now())
        )
        self.conversation_history.append(
            Message(response, "assistant", timestamp=_now())
        )
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def explore(self, target: str = ".") -> str:
        return self.run(f"请探索 {target} 的代码结构", mode="explore")

    def analyze(self, focus: str = "") -> str:
        q = "请分析代码质量" + (f",重点关注{focus}" if focus else "")
        return self.run(q, mode="analyze")

    def plan_next_steps(self) -> str:
        return self.run("根据当前进度,规划下一步任务", mode="plan")

    def execute_command(self, command: str) -> str:
        return self._run_term(command)

    def create_note(
        self,
        title: str,
        content: str,
        note_type: str = "general",
        tags: list[str] | None = None,
    ) -> str:
        nid = self.note_tool.create_note(
            title=title,
            content=content,
            note_type=note_type,
            tags=tags or [self.project_name],
        )
        self.stats["notes_created"] += 1
        return nid

    def get_stats(self) -> dict[str, Any]:
        duration = (_now() - self.stats["session_start"]).total_seconds()
        return {
            "session_info": {
                "session_id": self.session_id,
                "project": self.project_name,
                "duration_seconds": duration,
            },
            "activity": {
                "commands_executed": self.stats["commands_executed"],
                "notes_created": self.stats["notes_created"],
                "issues_found": self.stats["issues_found"],
            },
            "notes": self.note_tool.summary_dict(),
        }

    def generate_report(
        self, save_to_file: bool = True, report_dir: str | None = None
    ) -> dict[str, Any]:
        report = self.get_stats()
        if save_to_file:
            out_dir = Path(report_dir or ".")
            out_dir.mkdir(parents=True, exist_ok=True)
            report_file = out_dir / f"maintainer_report_{self.session_id}.json"
            report_file.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            report["report_file"] = str(report_file)
            print(f"📄 报告已保存: {report_file}")
        return report
