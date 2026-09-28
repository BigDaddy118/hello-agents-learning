"""可选 Qdrant 向量存储；未配置时返回 None，走本地 JSON。"""

from __future__ import annotations

import os
from typing import Any, Optional


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        collection_name: str,
        dimension: int = 384,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        self.collection_name = collection_name
        self.dimension = dimension
        self._qm = qm
        self.client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=int(os.getenv("QDRANT_TIMEOUT") or "30"),
        )
        self._ensure_collection()
        print(f"[OK] 使用云端 Qdrant 集合: {collection_name}")

    def _ensure_collection(self) -> None:
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in names:
            return
        dist = (os.getenv("QDRANT_DISTANCE") or "cosine").lower()
        distance = {
            "cosine": self._qm.Distance.COSINE,
            "euclid": self._qm.Distance.EUCLID,
            "dot": self._qm.Distance.DOT,
        }.get(dist, self._qm.Distance.COSINE)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._qm.VectorParams(
                size=self.dimension, distance=distance
            ),
        )

    def upsert(self, points: list[dict[str, Any]]) -> None:
        from qdrant_client.http import models as qm

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qm.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p.get("payload") or {},
                )
                for p in points
            ],
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        must = []
        if where:
            for k, v in where.items():
                must.append(
                    self._qm.FieldCondition(
                        key=k, match=self._qm.MatchValue(value=v)
                    )
                )
        qfilter = self._qm.Filter(must=must) if must else None
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qfilter,
        )
        out = []
        for h in hits:
            meta = dict(h.payload or {})
            out.append(
                {
                    "id": str(h.id),
                    "score": float(h.score),
                    "content": meta.get("content", ""),
                    "metadata": meta,
                }
            )
        return out


def try_create_qdrant_store(
    collection_name: str,
    dimension: int = 384,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[QdrantVectorStore]:
    url = url or os.getenv("QDRANT_URL")
    api_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")
    if not url or "your-cluster" in url or "your_" in (api_key or ""):
        return None
    try:
        return QdrantVectorStore(url, api_key, collection_name, dimension=dimension)
    except Exception as e:
        print(f"[WARNING] Qdrant 不可用，回退本地存储: {e}")
        return None
