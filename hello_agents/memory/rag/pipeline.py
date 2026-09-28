"""RAG 管道：文档 → Markdown → 分块 → 嵌入 → 存储/检索/问答。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from ..embedding import (
    cosine,
    embed_query,
    get_dimension,
    get_text_embedder,
    preprocess_markdown_for_embedding,
)
from ..storage.qdrant_store import try_create_qdrant_store
from .document import convert_to_markdown, text_to_chunks
from .retrieval import search_expanded


class LocalRAGPipeline:
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        llm: Any = None,
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        self.llm = llm
        self.store_path = (
            self.knowledge_base_path / collection_name / f"{rag_namespace}.json"
        )
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, dict[str, Any]] = self._load()
        self.embedder = get_text_embedder()
        self.vector_store = try_create_qdrant_store(
            collection_name=collection_name,
            dimension=get_dimension(),
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.store_path.exists():
            return {}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        # 向量很大时仍落盘，便于无 Qdrant 复用
        self.store_path.write_text(
            json.dumps(self._docs, ensure_ascii=False),
            encoding="utf-8",
        )

    def index_chunks(self, chunks: list[dict[str, Any]], document_id: str) -> list[str]:
        if not chunks:
            print("[RAG] No chunks to index")
            return []
        texts = [
            preprocess_markdown_for_embedding(c["content"]) for c in chunks
        ]
        print(f"[RAG] Embedding start: total_texts={len(texts)} backend={self.embedder.name}")
        vecs = self.embedder.encode(texts)
        created: list[str] = []
        q_points: list[dict[str, Any]] = []
        for i, (ch, vec) in enumerate(zip(chunks, vecs)):
            cid = f"{document_id}#{i}"
            # Qdrant 需要 uuid/int id；本地用字符串
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, cid))
            payload = {
                "memory_id": cid,
                "document_id": document_id,
                "content": ch["content"],
                "heading_path": ch.get("heading_path"),
                "rag_namespace": self.rag_namespace,
                "memory_type": "rag_chunk",
                "is_rag_data": True,
                "data_source": "rag_pipeline",
            }
            self._docs[cid] = {
                "id": cid,
                "point_id": point_id,
                "vector": vec,
                **payload,
            }
            created.append(cid)
            q_points.append({"id": point_id, "vector": vec, "payload": payload})
        if self.vector_store is not None:
            try:
                self.vector_store.upsert(q_points)
            except Exception as e:
                print(f"[WARNING] Qdrant upsert 失败，仅本地: {e}")
        self._save()
        return created

    def add_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("text 不能为空")
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        chunks = text_to_chunks(
            text, chunk_tokens=chunk_size, overlap_tokens=chunk_overlap
        )
        if not chunks:
            raise ValueError("分块结果为空，未修改知识库")
        # 先备份；成功入库后再替换，避免 index 半路失败丢旧文档
        previous = dict(self._docs)
        try:
            self._docs = {
                k: v for k, v in previous.items() if v.get("document_id") != doc_id
            }
            created = self.index_chunks(chunks, doc_id)
        except Exception:
            self._docs = previous
            try:
                self._save()
            except Exception:
                pass
            raise
        return {
            "success": True,
            "document_id": doc_id,
            "chunks": len(created),
            "chunk_ids": created,
            "embedder": self.embedder.name,
        }

    def add_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ) -> dict[str, Any]:
        md = convert_to_markdown(file_path)
        if not md.strip():
            return {
                "success": False,
                "error": f"无法解析文档或内容为空: {file_path}",
            }
        doc_id = document_id or Path(file_path).stem
        info = self.add_text(
            md,
            document_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        info["source"] = file_path
        info["chars"] = len(md)
        return info

    def _search_once(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        if not query or not self._docs:
            return []

        if self.vector_store is not None:
            try:
                qv = embed_query(query)
                hits = self.vector_store.search(
                    query_vector=qv,
                    limit=limit,
                    score_threshold=min_score or None,
                    where={
                        "rag_namespace": self.rag_namespace,
                        "is_rag_data": True,
                        "memory_type": "rag_chunk",
                    },
                )
                if hits:
                    return [
                        {
                            "id": h["metadata"].get("memory_id", h["id"]),
                            "document_id": h["metadata"].get("document_id"),
                            "content": h.get("content")
                            or h["metadata"].get("content", ""),
                            "score": h["score"],
                            "heading_path": h["metadata"].get("heading_path"),
                        }
                        for h in hits
                    ]
            except Exception as e:
                print(f"[WARNING] Qdrant 检索失败，回退本地: {e}")

        qv = embed_query(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._docs.values():
            vec = item.get("vector")
            if not vec:
                continue
            s = cosine(qv, vec)
            if s >= min_score:
                row = {k: v for k, v in item.items() if k != "vector"}
                row["score"] = s
                scored.append((s, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        enable_mqe: bool = False,
        mqe_expansions: int = 2,
        enable_hyde: bool = False,
        candidate_pool_multiplier: int = 4,
    ) -> list[dict[str, Any]]:
        if enable_mqe or enable_hyde:
            return search_expanded(
                self._search_once,
                query=query,
                top_k=limit,
                min_score=min_score,
                enable_mqe=enable_mqe,
                mqe_expansions=mqe_expansions,
                enable_hyde=enable_hyde,
                candidate_pool_multiplier=candidate_pool_multiplier,
                llm=self.llm,
            )
        return self._search_once(query, limit=limit, min_score=min_score)

    def ask(
        self,
        question: str,
        limit: int = 4,
        min_score: float = 0.05,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        mqe_expansions: int = 2,
    ) -> dict[str, Any]:
        hits = self.search(
            question,
            limit=limit,
            min_score=min_score,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
            mqe_expansions=mqe_expansions,
        )
        context = "\n\n---\n\n".join(
            f"[{h.get('document_id')}] {h['content']}" for h in hits
        )
        if not hits:
            return {
                "success": True,
                "answer": "知识库中未找到相关内容。",
                "sources": [],
            }
        if self.llm is not None:
            try:
                prompt = [
                    {
                        "role": "system",
                        "content": "根据给定知识片段回答问题，不要编造。",
                    },
                    {
                        "role": "user",
                        "content": f"问题：{question}\n\n知识：\n{context}\n\n请作答：",
                    },
                ]
                answer = self.llm.invoke(prompt)
                return {"success": True, "answer": answer, "sources": hits}
            except Exception as e:
                print(f"[WARNING] LLM 问答失败，返回摘录: {e}")
        # extractive fallback
        top = hits[0]["content"]
        return {
            "success": True,
            "answer": f"根据知识库：{top}",
            "sources": hits,
        }

    def stats(self) -> dict[str, Any]:
        doc_ids = {v["document_id"] for v in self._docs.values()}
        return {
            "namespace": self.rag_namespace,
            "collection": self.collection_name,
            "documents": len(doc_ids),
            "chunks": len(self._docs),
            "store_path": str(self.store_path),
            "embedder": self.embedder.name,
            "vector_backend": "qdrant" if self.vector_store else "local_json",
        }


def create_rag_pipeline(
    knowledge_base_path: str = "./knowledge_base",
    collection_name: str = "rag_knowledge_base",
    rag_namespace: str = "default",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    llm: Any = None,
    **_: Any,
) -> LocalRAGPipeline:
    return LocalRAGPipeline(
        knowledge_base_path=knowledge_base_path,
        collection_name=collection_name,
        rag_namespace=rag_namespace,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        llm=llm,
    )
