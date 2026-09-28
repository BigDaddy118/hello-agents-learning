"""感知记忆（8.2.5）：按模态分桶 + 文本检索/近因性（图/音频以元数据为主）。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from .._scoring import importance_weight, keyword_score, recency_score, tfidf_scores
from ..base import BaseMemory, MemoryConfig, MemoryItem
from ..storage.document_store import SQLiteDocumentStore


class PerceptualMemory(BaseMemory):
    def __init__(
        self,
        config: MemoryConfig | None = None,
        doc_store: SQLiteDocumentStore | None = None,
    ):
        super().__init__(config)
        self.doc_store = doc_store or SQLiteDocumentStore(self.config.database_path)
        self._by_modality: dict[str, dict[str, MemoryItem]] = defaultdict(dict)
        self._load()

    def _load(self) -> None:
        for row in self.doc_store.list_by_type("perceptual"):
            item = _row_to_item(row)
            mod = item.metadata.get("modality", "text")
            self._by_modality[mod][item.id] = item

    def add(self, memory_item: MemoryItem) -> str:
        memory_item.memory_type = "perceptual"
        mod = memory_item.metadata.get("modality", "text")
        self._by_modality[mod][memory_item.id] = memory_item
        self.doc_store.upsert(
            memory_item.id,
            memory_item.user_id,
            "perceptual",
            memory_item.content,
            memory_item.importance,
            memory_item.timestamp,
            memory_item.metadata,
        )
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs: Any) -> list[MemoryItem]:
        user_id = kwargs.get("user_id")
        target = kwargs.get("target_modality") or kwargs.get("query_modality")
        modalities = [target] if target else list(self._by_modality.keys())
        pool: list[MemoryItem] = []
        for mod in modalities:
            for m in self._by_modality.get(mod, {}).values():
                if user_id and m.user_id != user_id:
                    continue
                pool.append(m)

        docs = {m.id: m.content for m in pool}
        vec = tfidf_scores(query, docs)
        scored: list[tuple[float, MemoryItem]] = []
        for m in pool:
            v = vec.get(m.id, 0.0)
            k = keyword_score(query, m.content)
            vec_score = v if v > 0 else k
            base = vec_score * 0.8 + recency_score(m.timestamp) * 0.2
            score = base * importance_weight(m.importance)
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def all_items(self) -> list[MemoryItem]:
        out: list[MemoryItem] = []
        for bucket in self._by_modality.values():
            out.extend(bucket.values())
        return out

    def remove(self, memory_id: str) -> bool:
        mid = self._resolve(memory_id)
        if not mid:
            return False
        for bucket in self._by_modality.values():
            if mid in bucket:
                bucket.pop(mid)
                break
        return self.doc_store.delete(mid)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        **metadata: Any,
    ) -> bool:
        mid = self._resolve(memory_id)
        if not mid:
            return False
        m = None
        for bucket in self._by_modality.values():
            if mid in bucket:
                m = bucket[mid]
                break
        if not m:
            return False
        if content is not None:
            m.content = content
        if importance is not None:
            m.importance = float(importance)
        if metadata:
            m.metadata.update(metadata)
        self.doc_store.upsert(
            m.id, m.user_id, "perceptual", m.content, m.importance, m.timestamp, m.metadata
        )
        return True

    def clear(self) -> int:
        n = sum(len(b) for b in self._by_modality.values())
        user = None
        for b in self._by_modality.values():
            if b:
                user = next(iter(b.values())).user_id
                break
        self._by_modality.clear()
        self.doc_store.clear_type("perceptual", user)
        return n

    def _resolve(self, memory_id: str) -> Optional[str]:
        for bucket in self._by_modality.values():
            if memory_id in bucket:
                return memory_id
            for i in bucket:
                if i.startswith(memory_id):
                    return i
        return None


def _row_to_item(row: dict[str, Any]) -> MemoryItem:
    from datetime import datetime

    try:
        timestamp = datetime.fromisoformat(row["timestamp"])
    except Exception:
        timestamp = datetime.now()
    return MemoryItem(
        id=row["id"],
        content=row["content"],
        memory_type="perceptual",
        importance=float(row["importance"]),
        user_id=row["user_id"],
        timestamp=timestamp,
        metadata=row.get("metadata") or {},
    )
