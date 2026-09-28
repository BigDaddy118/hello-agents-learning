"""记忆工具 MemoryTool（8.2.3：完整生命周期接口）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hello_agents.memory import MemoryConfig, MemoryManager

from ..base import Tool, ToolParameter

_TYPE_LABEL = {
    "working": "工作记忆",
    "episodic": "情景记忆",
    "semantic": "语义记忆",
    "perceptual": "感知记忆",
}


class MemoryTool(Tool):
    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: Optional[MemoryConfig] = None,
        memory_types: Optional[list[str]] = None,
    ):
        super().__init__(
            name="memory",
            description="记忆工具 - 可以存储和检索对话历史、知识和经验",
        )
        self.user_id = user_id
        self.current_session_id: str | None = None
        types = memory_types or ["working", "episodic", "semantic"]
        self.memory_manager = MemoryManager(
            config=memory_config or MemoryConfig(),
            user_id=user_id,
            enable_working="working" in types,
            enable_episodic="episodic" in types,
            enable_semantic="semantic" in types,
            enable_perceptual="perceptual" in types,
        )

    def run(self, parameters: dict[str, Any]) -> str:
        action = parameters.get("action") or parameters.get("input") or "summary"
        kwargs = {k: v for k, v in parameters.items() if k not in ("action", "input")}
        return self.execute(str(action), **kwargs)

    def execute(self, action: str, **kwargs: Any) -> str:
        handlers = {
            "add": self._add_memory,
            "search": self._search_memory,
            "summary": self._get_summary,
            "stats": self._get_stats,
            "update": self._update_memory,
            "remove": self._remove_memory,
            "forget": self._forget,
            "consolidate": self._consolidate,
            "clear_all": self._clear_all,
        }
        fn = handlers.get(action)
        if not fn:
            return (
                f"❌ 不支持的操作: {action}。"
                "支持: add, search, summary, stats, update, remove, forget, consolidate, clear_all"
            )
        return fn(**kwargs)

    def _add_memory(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        file_path: Optional[str] = None,
        modality: Optional[str] = None,
        **metadata: Any,
    ) -> str:
        try:
            if self.current_session_id is None:
                self.current_session_id = (
                    f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            if memory_type == "perceptual" and file_path:
                metadata.setdefault(
                    "modality", modality or self._infer_modality(file_path)
                )
                metadata.setdefault("raw_data", file_path)
            elif modality:
                metadata.setdefault("modality", modality)
            metadata.update(
                {
                    "session_id": self.current_session_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata,
                auto_classify=False,
            )
            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"
        except Exception as e:
            return f"❌ 添加记忆失败: {e}"

    def _search_memory(
        self,
        query: str = "",
        limit: int = 5,
        memory_types: Optional[list[str]] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.1,
    ) -> str:
        try:
            if memory_type and not memory_types:
                memory_types = [memory_type]
            results = self.memory_manager.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=memory_types,
                min_importance=min_importance,
            )
            if not results:
                return f"🔍 未找到与 '{query}' 相关的记忆"
            lines = [f"🔍 找到 {len(results)} 条相关记忆:"]
            for i, memory in enumerate(results, 1):
                label = _TYPE_LABEL.get(memory.memory_type, memory.memory_type)
                preview = (
                    memory.content[:80] + "..."
                    if len(memory.content) > 80
                    else memory.content
                )
                lines.append(
                    f"{i}. [{label}] {preview} (重要性: {memory.importance:.2f})"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 搜索记忆失败: {e}"

    def _get_summary(self, limit: int = 10, **_: Any) -> str:
        try:
            data = self.memory_manager.get_summary(limit=limit)
            lines = [
                f"📊 记忆摘要 (user={data['user_id']}, total={data['total']})",
                "类型统计: "
                + ", ".join(f"{k}={v}" for k, v in data["counts"].items()),
            ]
            if data["recent"]:
                lines.append("最近记忆:")
                for i, m in enumerate(data["recent"], 1):
                    label = _TYPE_LABEL.get(m.memory_type, m.memory_type)
                    preview = m.content[:60] + ("..." if len(m.content) > 60 else "")
                    lines.append(f"  {i}. [{label}] {preview}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 获取摘要失败: {e}"

    def _get_stats(self, **_: Any) -> str:
        try:
            s = self.memory_manager.get_stats()
            lines = [
                f"📈 统计 (user={s['user_id']})",
                f"总数: {s['total']}，平均重要性: {s['avg_importance']}",
                "类型: " + ", ".join(f"{k}={v}" for k, v in s["counts"].items()),
                "启用: " + ", ".join(s["enabled_types"]),
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 获取统计失败: {e}"

    def _update_memory(
        self,
        memory_id: str = "",
        content: Optional[str] = None,
        importance: Optional[float] = None,
        **metadata: Any,
    ) -> str:
        try:
            ok = self.memory_manager.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                **metadata,
            )
            return (
                f"✅ 记忆已更新 (ID: {memory_id[:8]}...)"
                if ok
                else f"❌ 未找到记忆: {memory_id}"
            )
        except Exception as e:
            return f"❌ 更新记忆失败: {e}"

    def _remove_memory(self, memory_id: str = "", **_: Any) -> str:
        try:
            ok = self.memory_manager.remove_memory(memory_id)
            return (
                f"🗑️ 记忆已删除 (ID: {memory_id[:8]}...)"
                if ok
                else f"❌ 未找到记忆: {memory_id}"
            )
        except Exception as e:
            return f"❌ 删除记忆失败: {e}"

    def _forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
        **_: Any,
    ) -> str:
        try:
            count = self.memory_manager.forget_memories(
                strategy=strategy,
                threshold=threshold,
                max_age_days=max_age_days,
            )
            return f"🧹 已遗忘 {count} 条记忆（策略: {strategy}）"
        except Exception as e:
            return f"❌ 遗忘记忆失败: {e}"

    def _consolidate(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
        **_: Any,
    ) -> str:
        try:
            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold,
            )
            return (
                f"🔄 已整合 {count} 条记忆为长期记忆"
                f"（{from_type} → {to_type}，阈值={importance_threshold}）"
            )
        except Exception as e:
            return f"❌ 整合记忆失败: {e}"

    def _clear_all(self, **_: Any) -> str:
        n = self.memory_manager.clear_all()
        return f"🧹 已清空 {n} 条记忆"

    @staticmethod
    def _infer_modality(file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return "image"
        if ext in {".mp3", ".wav", ".flac", ".m4a"}:
            return "audio"
        return "text"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description=(
                    "记忆操作: add/search/summary/stats/update/remove/"
                    "forget/consolidate/clear_all"
                ),
                required=True,
            )
        ]
