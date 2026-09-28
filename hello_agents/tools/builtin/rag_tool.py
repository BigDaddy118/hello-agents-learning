"""RAG 工具（8.3.4：文档载入 / 分块 / 嵌入 / 检索 / 问答）。"""

from __future__ import annotations

from typing import Any, Optional

from hello_agents.memory.rag.pipeline import LocalRAGPipeline, create_rag_pipeline

from ..base import Tool, ToolParameter

_UNSET = object()


class RAGTool(Tool):
    """提供完整 RAG：多格式文档、检索、LLM 增强问答、知识库管理。"""

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        llm: Any = _UNSET,
        **kwargs: Any,
    ):
        super().__init__(
            name="rag",
            description="RAG 工具：文档入库与知识库问答。",
        )
        self.knowledge_base_path = knowledge_base_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        # llm=_UNSET → 自动探测；显式 None → 摘录问答
        self.llm = self._try_llm() if llm is _UNSET else llm
        self._pipelines: dict[str, LocalRAGPipeline] = {}
        self._pipelines[self.rag_namespace] = create_rag_pipeline(
            knowledge_base_path=self.knowledge_base_path,
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            rag_namespace=self.rag_namespace,
            llm=self.llm,
        )
        print(
            f"✅ RAG工具初始化成功: namespace={self.rag_namespace}, "
            f"collection={self.collection_name}"
        )

    @staticmethod
    def _try_llm() -> Any:
        try:
            from hello_agents.core.llm import HelloAgentsLLM

            return HelloAgentsLLM()
        except Exception:
            return None

    def _pipeline(self, namespace: Optional[str] = None) -> LocalRAGPipeline:
        ns = namespace or self.rag_namespace
        if ns not in self._pipelines:
            self._pipelines[ns] = create_rag_pipeline(
                knowledge_base_path=self.knowledge_base_path,
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                rag_namespace=ns,
                llm=self.llm,
            )
        return self._pipelines[ns]

    def run(self, parameters: dict[str, Any]) -> str:
        action = parameters.get("input") or parameters.get("action") or "stats"
        kwargs = {k: v for k, v in parameters.items() if k not in ("action", "input")}
        return self.execute(str(action), **kwargs)

    def execute(self, action: str, **kwargs: Any) -> str:
        handlers = {
            "add_text": self._add_text,
            "add_document": self._add_document,
            "search": self._search,
            "ask": self._ask,
            "stats": self._stats,
        }
        fn = handlers.get(action)
        if not fn:
            return (
                f"❌ 不支持的操作: {action}。"
                "支持: add_text, add_document, search, ask, stats"
            )
        return fn(**kwargs)

    def _add_text(
        self,
        text: str = "",
        document_id: Optional[str] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        rag_namespace: Optional[str] = None,
        **_: Any,
    ) -> str:
        try:
            info = self._pipeline(rag_namespace).add_text(
                text,
                document_id=document_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return (
                f"✅ 已入库 document_id={info['document_id']}，"
                f"分块数={info['chunks']}，embedder={info['embedder']}"
            )
        except Exception as e:
            return f"❌ 添加文本失败: {e}"

    def _add_document(
        self,
        file_path: str = "",
        document_id: Optional[str] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        rag_namespace: Optional[str] = None,
        **_: Any,
    ) -> str:
        try:
            info = self._pipeline(rag_namespace).add_document(
                file_path,
                document_id=document_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if not info.get("success"):
                return f"❌ 添加文档失败: {info.get('error', '未知错误')}"
            return (
                f"✅ 文档已入库 document_id={info['document_id']}，"
                f"chars={info.get('chars')}，分块数={info['chunks']}，"
                f"embedder={info['embedder']}"
            )
        except Exception as e:
            return f"❌ 添加文档失败: {e}"

    def _search(
        self,
        query: str = "",
        limit: int = 5,
        min_score: float = 0.0,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        mqe_expansions: int = 2,
        rag_namespace: Optional[str] = None,
        **_: Any,
    ) -> str:
        try:
            hits = self._pipeline(rag_namespace).search(
                query=query,
                limit=limit,
                min_score=min_score,
                enable_mqe=bool(enable_mqe),
                enable_hyde=bool(enable_hyde),
                mqe_expansions=int(mqe_expansions),
            )
            if not hits:
                return "🔍 未找到相关知识。"
            mode = []
            if enable_mqe:
                mode.append("MQE")
            if enable_hyde:
                mode.append("HyDE")
            tag = f"（{'+'.join(mode)}）" if mode else ""
            lines = [f"🔍 找到 {len(hits)} 条相关知识{tag}:"]
            for i, h in enumerate(hits, 1):
                preview = h["content"][:120].replace("\n", " ")
                path = h.get("heading_path") or "-"
                lines.append(
                    f"{i}. [{h.get('document_id')}|{path}] {preview} "
                    f"(score: {h['score']:.2f})"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 搜索失败: {e}"

    def _ask(
        self,
        question: str = "",
        query: str = "",
        limit: int = 4,
        min_score: float = 0.05,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        mqe_expansions: int = 2,
        rag_namespace: Optional[str] = None,
        **_: Any,
    ) -> str:
        q = question or query
        try:
            result = self._pipeline(rag_namespace).ask(
                q,
                limit=limit,
                min_score=min_score,
                enable_mqe=bool(enable_mqe),
                enable_hyde=bool(enable_hyde),
                mqe_expansions=int(mqe_expansions),
            )
            srcs = result.get("sources") or []
            tail = ""
            if srcs:
                ids = ", ".join(
                    str(s.get("document_id")) for s in srcs[:3] if s.get("document_id")
                )
                tail = f"\n📚 来源: {ids}"
            return f"💬 {result.get('answer', '')}{tail}"
        except Exception as e:
            return f"❌ 问答失败: {e}"

    def _stats(self, rag_namespace: Optional[str] = None, **_: Any) -> str:
        try:
            s = self._pipeline(rag_namespace).stats()
            return (
                f"📈 知识库统计\n"
                f"namespace: {s['namespace']}\n"
                f"collection: {s['collection']}\n"
                f"documents: {s['documents']}\n"
                f"chunks: {s['chunks']}\n"
                f"embedder: {s['embedder']}\n"
                f"vector_backend: {s['vector_backend']}\n"
                f"store: {s['store_path']}"
            )
        except Exception as e:
            return f"❌ 统计失败: {e}"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="RAG 操作：add_text / add_document / search / ask / stats",
                required=True,
            )
        ]
