"""语义记忆（8.2.5）：本地实体图 + TF-IDF 向量混合检索。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from .._scoring import (
    extract_entities,
    importance_weight,
    keyword_score,
    tfidf_scores,
)
from ..base import BaseMemory, MemoryConfig, MemoryItem


class SemanticMemory(BaseMemory):
    def __init__(self, config: MemoryConfig | None = None, storage_backend: Any = None):
        super().__init__(config, storage_backend)
        self._items: dict[str, MemoryItem] = {}
        # entity -> set(memory_id)
        self.entity_index: dict[str, set[str]] = defaultdict(set)
        # (src, dst) -> count
        self.relations: dict[tuple[str, str], int] = defaultdict(int)

    def add(self, memory_item: MemoryItem) -> str:
        memory_item.memory_type = "semantic"
        entities = extract_entities(memory_item.content)
        memory_item.metadata["entities"] = entities
        memory_item.metadata["entity_count"] = len(entities)
        self._items[memory_item.id] = memory_item
        for e in entities:
            self.entity_index[e].add(memory_item.id)
        # 共现当作无向关系边
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                self.relations[(a, b)] += 1
                self.relations[(b, a)] += 1
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs: Any) -> list[MemoryItem]:
        user_id = kwargs.get("user_id")
        items = [
            m
            for m in self._items.values()
            if not user_id or m.user_id == user_id
        ]
        docs = {m.id: m.content for m in items}
        vector_hits = tfidf_scores(query, docs)

        # 图检索：查询实体命中的记忆
        q_ents = extract_entities(query) or []
        # 若抽不出实体，用分词当弱实体
        if not q_ents:
            from .._scoring import tokenize

            q_ents = [t for t in tokenize(query) if len(t) > 1][:8]

        graph_scores: dict[str, float] = defaultdict(float)
        for e in q_ents:
            for mid in self.entity_index.get(e, ()):
                graph_scores[mid] += 1.0
            # 一跳邻居
            for (a, b), w in self.relations.items():
                if a == e:
                    for mid in self.entity_index.get(b, ()):
                        graph_scores[mid] += 0.3 * min(w, 3)

        max_g = max(graph_scores.values()) if graph_scores else 1.0
        combined: dict[str, dict[str, Any]] = {}
        for mid, vs in vector_hits.items():
            if vs <= 0 and mid not in graph_scores:
                continue
            combined[mid] = {
                "memory_id": mid,
                "vector_score": vs,
                "graph_score": 0.0,
                "importance": self._items[mid].importance,
            }
        for mid, gs in graph_scores.items():
            if mid not in self._items:
                continue
            norm_g = gs / max_g
            if mid in combined:
                combined[mid]["graph_score"] = norm_g
            else:
                combined[mid] = {
                    "memory_id": mid,
                    "vector_score": keyword_score(query, self._items[mid].content),
                    "graph_score": norm_g,
                    "importance": self._items[mid].importance,
                }

        ranked: list[tuple[float, MemoryItem]] = []
        for mid, r in combined.items():
            base = r["vector_score"] * 0.7 + r["graph_score"] * 0.3
            score = base * importance_weight(r["importance"])
            if score > 0:
                ranked.append((score, self._items[mid]))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:limit]]

    def all_items(self) -> list[MemoryItem]:
        return list(self._items.values())

    def remove(self, memory_id: str) -> bool:
        mid = self._resolve(memory_id)
        if not mid:
            return False
        item = self._items.pop(mid)
        for e in item.metadata.get("entities", []):
            self.entity_index[e].discard(mid)
        return True

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
        m = self._items[mid]
        # 先清旧实体
        for e in m.metadata.get("entities", []):
            self.entity_index[e].discard(mid)
        if content is not None:
            m.content = content
            ents = extract_entities(content)
            m.metadata["entities"] = ents
            for e in ents:
                self.entity_index[e].add(mid)
        if importance is not None:
            m.importance = float(importance)
        if metadata:
            m.metadata.update(metadata)
        return True

    def clear(self) -> int:
        n = len(self._items)
        self._items.clear()
        self.entity_index.clear()
        self.relations.clear()
        return n

    def _resolve(self, memory_id: str) -> Optional[str]:
        if memory_id in self._items:
            return memory_id
        for i in self._items:
            if i.startswith(memory_id):
                return i
        return None
