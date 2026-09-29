"""ContextBuilder — GSSC（Gather-Select-Structure-Compress）流水线（9.3）.

cSpell:ignore GSSC
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from ..core.message import Message


def _now() -> datetime:
    return datetime.now()  # noqa: DTZ005 — 与 Message / MemoryItem 的 naive 时间戳一致


def count_tokens(text: str) -> int:
    """估算 token 数。有 tiktoken 则精确，否则中英混合粗估。"""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except (ImportError, ModuleNotFoundError, OSError, ValueError, AttributeError):
        # ponytail: 无 tiktoken 时粗估；精确计数装 tiktoken 即可
        chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        english = len([w for w in text.split() if w])
        return max(1, int(chinese + english * 1.3)) if text else 0


@dataclass
class ContextPacket:
    """候选信息包。"""

    content: str
    timestamp: datetime = field(default_factory=_now)
    token_count: int = 0
    relevance_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            self.token_count = count_tokens(self.content)
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class ContextConfig:
    """上下文构建配置。"""

    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7

    def __post_init__(self) -> None:
        assert 0.0 <= self.reserve_ratio <= 1.0
        assert 0.0 <= self.min_relevance <= 1.0
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6

    def get_available_tokens(self) -> int:
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """GSSC 上下文构建器。"""

    def __init__(
        self,
        memory_tool: Any = None,
        rag_tool: Any = None,
        config: ContextConfig | None = None,
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()

    def build(
        self,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        custom_packets: list[ContextPacket] | None = None,
        additional_packets: list[ContextPacket] | None = None,
    ) -> str:
        extra = list(custom_packets or []) + list(additional_packets or [])
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            custom_packets=extra,
        )
        selected = self._select(packets, user_query)
        structured = self._structure(selected, user_query)
        if self.config.enable_compression:
            return self._compress(structured, self.config.get_available_tokens())
        return structured

    def _gather(
        self,
        user_query: str,
        conversation_history: list[Message],
        system_instructions: str | None,
        custom_packets: list[ContextPacket],
    ) -> list[ContextPacket]:
        packets: list[ContextPacket] = []

        if system_instructions:
            packets.append(
                ContextPacket(
                    content=system_instructions,
                    relevance_score=1.0,
                    metadata={"type": "system_instruction", "priority": "high"},
                )
            )

        if self.memory_tool:
            try:
                packets.extend(self._gather_memory(user_query))
            except Exception as e:  # noqa: BLE001 — 外部工具任意失败都不应打断流水线
                print(f"[WARNING] 记忆检索失败: {e}")

        if self.rag_tool:
            try:
                packets.extend(self._gather_rag(user_query))
            except Exception as e:  # noqa: BLE001 — 同上
                print(f"[WARNING] RAG 检索失败: {e}")

        if conversation_history:
            for msg in conversation_history[-5:]:
                packets.append(
                    ContextPacket(
                        content=f"{msg.role}: {msg.content}",
                        timestamp=getattr(msg, "timestamp", None) or _now(),
                        relevance_score=0.6,
                        metadata={"type": "conversation_history", "role": msg.role},
                    )
                )

        packets.extend(custom_packets)
        print(f"[ContextBuilder] 汇集了 {len(packets)} 个候选信息包")
        return packets

    def _gather_memory(self, user_query: str) -> list[ContextPacket]:
        """优先走 memory_manager 结构化结果，否则解析 execute 文本。"""
        mgr = getattr(self.memory_tool, "memory_manager", None)
        if mgr is not None:
            items = mgr.retrieve_memories(
                query=user_query, limit=10, min_importance=0.3
            )
            out: list[ContextPacket] = []
            for item in items:
                out.append(
                    ContextPacket(
                        content=f"记忆: {item.content}",
                        timestamp=getattr(item, "timestamp", None) or _now(),
                        relevance_score=float(getattr(item, "importance", 0.5)),
                        metadata={
                            "type": "memory",
                            "memory_type": getattr(item, "memory_type", "unknown"),
                        },
                    )
                )
            return out

        text = self.memory_tool.run(
            {
                "action": "search",
                "query": user_query,
                "limit": 10,
                "min_importance": 0.3,
            }
        )
        if not text or "未找到" in text or text.startswith("❌"):
            return []
        return [
            ContextPacket(
                content=text,
                relevance_score=0.6,
                metadata={"type": "memory"},
            )
        ]

    def _gather_rag(self, user_query: str) -> list[ContextPacket]:
        """优先走 pipeline.search 结构化命中。"""
        pipeline_fn = getattr(self.rag_tool, "_pipeline", None)
        if callable(pipeline_fn):
            search = getattr(pipeline_fn(), "search", None)
            if callable(search):
                hits = cast(list[dict[str, Any]], search(
                    query=user_query, limit=5, min_score=0.3
                ))
                return [
                    ContextPacket(
                        content=str(h["content"]),
                        relevance_score=float(h.get("score", 0.5)),
                        metadata={
                            "type": "rag_result",
                            "document_id": h.get("document_id"),
                        },
                    )
                    for h in hits
                ]

        text = self.rag_tool.run(
            {
                "action": "search",
                "query": user_query,
                "limit": 5,
                "min_score": 0.3,
            }
        )
        if not text or "未找到" in text or text.startswith("❌"):
            return []
        return [
            ContextPacket(
                content=text,
                relevance_score=0.6,
                metadata={"type": "rag_result"},
            )
        ]

    def _select(
        self,
        packets: list[ContextPacket],
        user_query: str,
    ) -> list[ContextPacket]:
        available = self.config.get_available_tokens()
        system = [
            p for p in packets if p.metadata.get("type") == "system_instruction"
        ]
        others = [
            p for p in packets if p.metadata.get("type") != "system_instruction"
        ]

        system_tokens = sum(p.token_count for p in system)
        if system_tokens >= available:
            print("[WARNING] 系统指令已占满所有 token 预算")
            return system

        scored: list[tuple[float, ContextPacket]] = []
        for packet in others:
            if abs(packet.relevance_score - 0.5) < 1e-9:
                packet.relevance_score = self._calculate_relevance(
                    packet.content, user_query
                )
            recency = self._calculate_recency(packet.timestamp)
            combined = (
                self.config.relevance_weight * packet.relevance_score
                + self.config.recency_weight * recency
            )
            if packet.relevance_score >= self.config.min_relevance:
                scored.append((combined, packet))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = list(system)
        used = system_tokens
        for _, packet in scored:
            if used + packet.token_count <= available:
                selected.append(packet)
                used += packet.token_count

        print(f"[ContextBuilder] 选择了 {len(selected)} 个信息包,共 {used} tokens")
        return selected

    def _calculate_relevance(self, content: str, query: str) -> float:
        # ponytail: 词集重叠相似度；要向量相似度再说
        c_words = set(content.lower().split())
        q_words = set(query.lower().split())
        if not q_words:
            return 0.0
        inter = c_words & q_words
        union = c_words | q_words
        return len(inter) / len(union) if union else 0.0

    def _calculate_recency(self, timestamp: datetime) -> float:
        age_hours = max((_now() - timestamp).total_seconds(), 0) / 3600
        return max(0.1, min(1.0, math.exp(-0.1 * age_hours / 24)))

    def _structure(self, selected: list[ContextPacket], user_query: str) -> str:
        system_instructions: list[str] = []
        evidence: list[str] = []
        history: list[ContextPacket] = []
        other: list[str] = []

        for packet in selected:
            ptype = packet.metadata.get("type", "general")
            if ptype == "system_instruction":
                system_instructions.append(packet.content)
            elif ptype in ("rag_result", "knowledge", "memory"):
                evidence.append(packet.content)
            elif ptype == "conversation_history":
                history.append(packet)
            else:
                other.append(packet.content)

        history.sort(key=lambda p: p.timestamp)
        context = [p.content for p in history] + other

        sections: list[str] = []
        if system_instructions:
            sections.append("[Role & Policies]\n" + "\n".join(system_instructions))
        sections.append(f"[Task]\n{user_query}")
        if evidence:
            sections.append("[Evidence]\n" + "\n---\n".join(evidence))
        if context:
            sections.append("[Context]\n" + "\n".join(context))
        sections.append("[Output]\n请基于以上信息,提供准确、有据的回答。")
        return "\n\n".join(sections)

    def _compress(self, context: str, max_tokens: int) -> str:
        current = count_tokens(context)
        if current <= max_tokens:
            return context

        print(f"[ContextBuilder] 上下文超限({current} > {max_tokens}),执行压缩")
        sections = context.split("\n\n")
        kept: list[str] = []
        total = 0
        for section in sections:
            n = count_tokens(section)
            if total + n <= max_tokens:
                kept.append(section)
                total += n
                continue
            remain = max_tokens - total
            if remain > 50:
                kept.append(self._truncate_text(section, remain) + "\n[... 内容已压缩 ...]")
            break

        out = "\n\n".join(kept)
        print(f"[ContextBuilder] 压缩完成: {current} -> {count_tokens(out)} tokens")
        return out

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        total = count_tokens(text)
        if total <= 0:
            return text
        max_chars = int(len(text) * max_tokens / total)
        return text[:max_chars]
