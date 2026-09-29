"""统一嵌入：DashScope → sentence-transformers → TF-IDF 兜底。"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from typing import Any, Optional, Protocol

from ._scoring import tokenize


class TextEmbedder(Protocol):
    name: str
    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class TfidfEmbedder:
    """无依赖稀疏嵌入：固定哈希桶，可做余弦检索。"""

    def __init__(self, dimension: int = 384):
        self.name = "tfidf"
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        toks = tokenize(text)
        if not toks:
            return vec
        tf = Counter(toks)
        n = len(toks)
        for t, c in tf.items():
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign * (c / n)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class DashScopeEmbedder:
    def __init__(self, api_key: str, model: str, base_url: str, dimension: int = 1024):
        from openai import OpenAI

        self.name = "dashscope"
        self.dimension = dimension
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def encode(self, texts: list[str]) -> list[list[float]]:
        # OpenAI-compatible embeddings；失败则抛出让上层 fallback
        resp = self._client.embeddings.create(model=self.model, input=texts)
        out = [list(d.embedding) for d in resp.data]
        if out:
            self.dimension = len(out[0])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self.name = "sentence-transformers"
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension() or 384)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]


_embedder: Optional[TextEmbedder] = None


def get_dimension(default: int = 384) -> int:
    return get_text_embedder().dimension or default


def get_text_embedder(force_reload: bool = False) -> TextEmbedder:
    global _embedder
    if _embedder is not None and not force_reload:
        return _embedder

    kind = (os.getenv("EMBED_MODEL_TYPE") or "auto").lower()
    dim = int(os.getenv("QDRANT_VECTOR_SIZE") or "384")

    if kind in ("dashscope", "auto"):
        key = os.getenv("EMBED_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if key:
            try:
                model = os.getenv("EMBED_MODEL_NAME") or "text-embedding-v3"
                base = (
                    os.getenv("EMBED_BASE_URL")
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                _embedder = DashScopeEmbedder(key, model, base, dimension=dim)
                # 探活
                _embedder.encode(["ping"])
                print(f"[RAG] 嵌入后端: dashscope ({model})")
                return _embedder
            except Exception as e:
                print(f"[WARNING] DashScope 嵌入不可用: {e}")

    if kind in ("local", "sentence-transformers", "auto"):
        try:
            name = os.getenv("EMBED_MODEL_NAME") or "all-MiniLM-L6-v2"
            _embedder = SentenceTransformerEmbedder(name)
            print(f"[RAG] 嵌入后端: sentence-transformers ({name})")
            return _embedder
        except Exception as e:
            print(f"[WARNING] sentence-transformers 不可用: {e}")

    _embedder = TfidfEmbedder(dimension=dim)
    print(f"[RAG] 嵌入后端: tfidf (dim={dim})")
    return _embedder


def embed_query(query: str) -> list[float]:
    return get_text_embedder().encode([query])[0]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return max(0.0, min(1.0, dot / (na * nb)))


def preprocess_markdown_for_embedding(text: str) -> str:
    """去掉纯装饰性 Markdown 标记，保留正文。"""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        s = s.replace("**", "").replace("__", "").replace("`", "")
        if s:
            lines.append(s)
    return "\n".join(lines)
