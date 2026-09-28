"""工作记忆（8.2.5）：TTL + 容量 + TF-IDF/关键词混合检索。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .._scoring import (
    importance_weight,
    keyword_score,
    tfidf_scores,
    time_decay,
)
from ..base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    def __init__(self, config: MemoryConfig | None = None):
        super().__init__(config)
        self.max_capacity = self.config.working_memory_capacity or 50
        self.max_age_minutes = self.config.working_memory_ttl or 60
        self.memories: list[MemoryItem] = []

    def add(self, memory_item: MemoryItem) -> str:
        self._expire_old_memories()
        if len(self.memories) >= self.max_capacity:
            self._remove_lowest_priority_memory()
        self.memories.append(memory_item)
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs: Any) -> list[MemoryItem]:
        self._expire_old_memories()
        docs = {m.id: m.content for m in self.memories}
        vector_scores = tfidf_scores(query, docs)
        scored: list[tuple[float, MemoryItem]] = []
        for memory in self.memories:
            v = vector_scores.get(memory.id, 0.0)
            k = keyword_score(query, memory.content)
            base = v * 0.7 + k * 0.3 if v > 0 else k
            final = base * time_decay(memory.timestamp) * importance_weight(
                memory.importance
            )
            if final > 0:
                scored.append((final, memory))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def all_items(self) -> list[MemoryItem]:
        self._expire_old_memories()
        return list(self.memories)

    def remove(self, memory_id: str) -> bool:
        for i, m in enumerate(self.memories):
            if m.id == memory_id or m.id.startswith(memory_id):
                self.memories.pop(i)
                return True
        return False

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        **metadata: Any,
    ) -> bool:
        for m in self.memories:
            if m.id == memory_id or m.id.startswith(memory_id):
                if content is not None:
                    m.content = content
                if importance is not None:
                    m.importance = float(importance)
                if metadata:
                    m.metadata.update(metadata)
                return True
        return False

    def clear(self) -> int:
        n = len(self.memories)
        self.memories.clear()
        return n

    def _expire_old_memories(self) -> None:
        cutoff = datetime.now() - timedelta(minutes=self.max_age_minutes)
        self.memories = [m for m in self.memories if m.timestamp >= cutoff]

    def _remove_lowest_priority_memory(self) -> None:
        if not self.memories:
            return
        self.memories.sort(key=lambda m: m.importance)
        self.memories.pop(0)
