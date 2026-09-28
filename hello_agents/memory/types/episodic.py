"""情景记忆（8.2.5）：SQLite 持久化 + TF-IDF/近因性混合检索。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from .._scoring import importance_weight, keyword_score, recency_score, tfidf_scores
from ..base import BaseMemory, MemoryConfig, MemoryItem
from ..storage.document_store import SQLiteDocumentStore


class EpisodicMemory(BaseMemory):
    def __init__(
        self,
        config: MemoryConfig | None = None,
        doc_store: SQLiteDocumentStore | None = None,
    ):
        super().__init__(config)
        self.doc_store = doc_store or SQLiteDocumentStore(self.config.database_path)
        self.sessions: dict[str, list[str]] = {}
        self._cache: dict[str, MemoryItem] = {}
        self._load()

    def _load(self) -> None:
        for row in self.doc_store.list_by_type("episodic"):
            item = _row_to_item(row)
            self._cache[item.id] = item
            sid = item.metadata.get("session_id", "default")
            self.sessions.setdefault(sid, []).append(item.id)

    def add(self, memory_item: MemoryItem) -> str:
        memory_item.memory_type = "episodic"
        sid = memory_item.metadata.get("session_id", "default")
        self.sessions.setdefault(sid, []).append(memory_item.id)
        self._cache[memory_item.id] = memory_item
        self.doc_store.upsert(
            memory_item.id,
            memory_item.user_id,
            "episodic",
            memory_item.content,
            memory_item.importance,
            memory_item.timestamp,
            memory_item.metadata,
        )
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs: Any) -> list[MemoryItem]:
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        min_importance = float(kwargs.get("min_importance", 0.0))
        candidates = list(self._cache.values())
        if user_id:
            candidates = [m for m in candidates if m.user_id == user_id]
        if session_id:
            ids = set(self.sessions.get(session_id, []))
            candidates = [m for m in candidates if m.id in ids]
        candidates = [m for m in candidates if m.importance >= min_importance]

        docs = {m.id: m.content for m in candidates}
        vec = tfidf_scores(query, docs)
        scored: list[tuple[float, MemoryItem]] = []
        for m in candidates:
            v = vec.get(m.id, 0.0)
            k = keyword_score(query, m.content)
            # 无向量命中时用关键词当 vec 近似
            vec_score = v if v > 0 else k
            base = vec_score * 0.8 + recency_score(m.timestamp) * 0.2
            score = base * importance_weight(m.importance)
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def all_items(self) -> list[MemoryItem]:
        return list(self._cache.values())

    def remove(self, memory_id: str) -> bool:
        mid = self._resolve_id(memory_id)
        if not mid:
            return False
        self._cache.pop(mid, None)
        for sid, ids in self.sessions.items():
            self.sessions[sid] = [i for i in ids if i != mid]
        return self.doc_store.delete(mid)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        **metadata: Any,
    ) -> bool:
        mid = self._resolve_id(memory_id)
        if not mid or mid not in self._cache:
            return False
        m = self._cache[mid]
        if content is not None:
            m.content = content
        if importance is not None:
            m.importance = float(importance)
        if metadata:
            m.metadata.update(metadata)
        self.doc_store.upsert(
            m.id, m.user_id, "episodic", m.content, m.importance, m.timestamp, m.metadata
        )
        return True

    def clear(self) -> int:
        n = len(self._cache)
        user = next(iter(self._cache.values())).user_id if self._cache else None
        self._cache.clear()
        self.sessions.clear()
        self.doc_store.clear_type("episodic", user)
        return n

    def _resolve_id(self, memory_id: str) -> Optional[str]:
        if memory_id in self._cache:
            return memory_id
        for i in self._cache:
            if i.startswith(memory_id):
                return i
        return None


def _row_to_item(row: dict[str, Any]) -> MemoryItem:
    ts = row["timestamp"]
    try:
        timestamp = datetime.fromisoformat(ts)
    except Exception:
        timestamp = datetime.now()
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        memory_type=row["memory_type"],
        importance=float(row["importance"]),
        user_id=row["user_id"],
        timestamp=timestamp,
        metadata=row.get("metadata") or {},
    )
