"""记忆管理器：调度四种记忆类型（8.2.4–8.2.5）。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from .base import BaseMemory, MemoryConfig, MemoryItem
from .storage.document_store import SQLiteDocumentStore
from .types.episodic import EpisodicMemory
from .types.perceptual import PerceptualMemory
from .types.semantic import SemanticMemory
from .types.working import WorkingMemory


class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False,
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.memory_types: dict[str, BaseMemory] = {}

        shared_store: SQLiteDocumentStore | None = None
        if enable_episodic or enable_perceptual:
            shared_store = SQLiteDocumentStore(self.config.database_path)

        if enable_working:
            self.memory_types["working"] = WorkingMemory(self.config)
        if enable_episodic:
            self.memory_types["episodic"] = EpisodicMemory(self.config, shared_store)
        if enable_semantic:
            self.memory_types["semantic"] = SemanticMemory(self.config)
        if enable_perceptual:
            self.memory_types["perceptual"] = PerceptualMemory(self.config, shared_store)

        self.enabled = {k: True for k in self.memory_types}
        print(
            "MemoryManager初始化完成，启用记忆类型: "
            + str(list(self.memory_types.keys()))
        )

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
        auto_classify: bool = False,
    ) -> str:
        store = self.memory_types.get(memory_type)
        if not store:
            raise ValueError(f"记忆类型未启用: {memory_type}")
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            importance=float(importance),
            user_id=self.user_id,
            metadata=dict(metadata or {}),
        )
        return store.add(item)

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[list[str]] = None,
        min_importance: float = 0.1,
    ) -> list[MemoryItem]:
        types = memory_types or list(self.memory_types.keys())
        pooled: list[tuple[float, MemoryItem]] = []
        # 各类型各自检索后合并（用重要性做粗排序键，类型内已排好）
        per_type = max(1, limit)
        for t in types:
            store = self.memory_types.get(t)
            if not store:
                continue
            hits = store.retrieve(
                query,
                limit=per_type * 2,
                user_id=self.user_id,
                min_importance=min_importance,
            )
            for i, m in enumerate(hits):
                if m.importance < min_importance:
                    continue
                # 名次靠前加分
                pooled.append((m.importance + (len(hits) - i) * 0.01, m))
        # 去重
        seen: set[str] = set()
        out: list[MemoryItem] = []
        for _, m in sorted(pooled, key=lambda x: x[0], reverse=True):
            if m.id in seen:
                continue
            seen.add(m.id)
            out.append(m)
            if len(out) >= limit:
                break
        return out

    def get_summary(self, limit: int = 10) -> dict[str, Any]:
        counts = {t: len(s.all_items()) for t, s in self.memory_types.items()}
        recent: list[MemoryItem] = []
        for s in self.memory_types.values():
            recent.extend(s.all_items())
        recent.sort(key=lambda m: m.timestamp, reverse=True)
        return {
            "user_id": self.user_id,
            "counts": counts,
            "total": sum(counts.values()),
            "recent": recent[:limit],
        }

    def get_stats(self) -> dict[str, Any]:
        counts = {t: len(s.all_items()) for t, s in self.memory_types.items()}
        all_items = [m for s in self.memory_types.values() for m in s.all_items()]
        total = len(all_items)
        avg = sum(m.importance for m in all_items) / total if total else 0.0
        return {
            "user_id": self.user_id,
            "counts": counts,
            "total": total,
            "avg_importance": round(avg, 3),
            "enabled_types": list(self.memory_types.keys()),
        }

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        **metadata: Any,
    ) -> bool:
        for s in self.memory_types.values():
            if s.update(memory_id, content=content, importance=importance, **metadata):
                return True
        return False

    def remove_memory(self, memory_id: str) -> bool:
        for s in self.memory_types.values():
            if s.remove(memory_id):
                return True
        return False

    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        if strategy == "importance_based":
            return self._forget_by_importance(threshold)
        if strategy == "time_based":
            return self._forget_by_time(max_age_days)
        if strategy == "capacity_based":
            return self._forget_by_capacity(threshold)
        raise ValueError(f"未知遗忘策略: {strategy}")

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
    ) -> int:
        src = self.memory_types.get(from_type)
        dst = self.memory_types.get(to_type)
        if not src or not dst:
            raise ValueError(f"记忆类型未启用: {from_type} → {to_type}")
        moved = 0
        for m in list(src.all_items()):
            if m.importance < importance_threshold:
                continue
            src.remove(m.id)
            m.memory_type = to_type
            m.metadata["consolidated_from"] = from_type
            m.metadata["consolidated_at"] = datetime.now().isoformat()
            dst.add(m)
            moved += 1
        return moved

    def clear_all(self) -> int:
        return sum(s.clear() for s in self.memory_types.values())

    def _forget_by_importance(self, threshold: float) -> int:
        removed = 0
        for s in self.memory_types.values():
            for m in list(s.all_items()):
                if m.importance < threshold:
                    if s.remove(m.id):
                        removed += 1
        return removed

    def _forget_by_time(self, max_age_days: int) -> int:
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed = 0
        for s in self.memory_types.values():
            for m in list(s.all_items()):
                if m.timestamp < cutoff:
                    if s.remove(m.id):
                        removed += 1
        return removed

    def _forget_by_capacity(self, threshold: float) -> int:
        cap = self.config.working_memory_capacity
        all_items: list[tuple[BaseMemory, MemoryItem]] = []
        for s in self.memory_types.values():
            for m in s.all_items():
                all_items.append((s, m))
        if len(all_items) <= cap:
            return self._forget_by_importance(threshold)
        all_items.sort(key=lambda x: x[1].importance, reverse=True)
        keep = {m.id for _, m in all_items[:cap]}
        removed = 0
        for s, m in all_items:
            if m.id not in keep or m.importance < threshold:
                if s.remove(m.id):
                    removed += 1
        return removed
