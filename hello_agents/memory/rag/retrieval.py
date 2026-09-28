"""高级检索：MQE / HyDE / 扩展合并（8.3.5）。"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional


def prompt_mqe(query: str, n: int, llm: Any = None) -> list[str]:
    """多查询扩展；无 LLM 时用轻量启发式。"""
    if n <= 0:
        return []
    if llm is not None:
        try:
            text = llm.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是检索查询扩展助手。生成语义等价或互补的多样化查询。"
                            "使用中文，简短，避免标点。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。",
                    },
                ]
            )
            lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
            outs = [ln for ln in lines if ln and ln != query]
            if outs:
                return outs[:n]
        except Exception:
            pass
    return _heuristic_mqe(query, n)


def prompt_hyde(query: str, llm: Any = None) -> Optional[str]:
    """假设文档；无 LLM 则返回 None。"""
    if llm is not None:
        try:
            text = llm.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "根据用户问题，先写一段可能的答案性段落，"
                            "用于向量检索的查询文档（不要分析过程）。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"
                        ),
                    },
                ]
            )
            if text and str(text).strip():
                return str(text).strip()
        except Exception:
            pass
    # ponytail: 无 LLM 时把疑问句改成陈述式骨架，略缩语义鸿沟
    q = re.sub(r"[？?]+$", "", query).strip()
    q = re.sub(r"^(什么是|如何|怎样|怎么|为什么|哪些|哪个)", "", q).strip()
    if not q:
        return None
    return f"{q}是相关领域中的重要概念，通常涉及定义、原理、应用场景与实现方法。"


def _heuristic_mqe(query: str, n: int) -> list[str]:
    base = re.sub(r"[？?！!。．]+$", "", query).strip()
    variants = [
        base,
        re.sub(r"^(如何|怎样|怎么|什么是|为什么)", "", base).strip(),
        f"{base} 定义",
        f"{base} 原理",
        f"{base} 应用",
        f"{base} 入门",
    ]
    out: list[str] = []
    for v in variants:
        if v and v != query and v not in out:
            out.append(v)
        if len(out) >= n:
            break
    return out[:n]


SearchFn = Callable[..., list[dict[str, Any]]]


def search_expanded(
    search_fn: SearchFn,
    query: str,
    top_k: int = 8,
    min_score: float = 0.0,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
    llm: Any = None,
) -> list[dict[str, Any]]:
    """扩展 → 多次检索 → 按 id 去重取最高分 → top_k。"""
    if not query:
        return []

    expansions: list[str] = [query]
    if enable_mqe and mqe_expansions > 0:
        expansions.extend(prompt_mqe(query, mqe_expansions, llm=llm))
    if enable_hyde:
        hyde = prompt_hyde(query, llm=llm)
        if hyde:
            expansions.append(hyde)

    uniq: list[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq

    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))

    agg: dict[str, dict[str, Any]] = {}
    for q in expansions:
        hits = search_fn(query=q, limit=per, min_score=min_score)
        for h in hits:
            mid = str(h.get("id") or h.get("memory_id") or id(h))
            s = float(h.get("score", 0.0))
            prev = agg.get(mid)
            if prev is None or s > float(prev.get("score", 0.0)):
                row = dict(h)
                row["_expansion"] = q if q != query else None
                agg[mid] = row

    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]
