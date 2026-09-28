"""检索评分工具：关键词 + 简易 TF-IDF（无第三方依赖）。"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Iterable


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())


def keyword_score(query: str, content: str) -> float:
    tokens = tokenize(query)
    if not tokens:
        return 0.0
    c = content.lower()
    return sum(1 for t in tokens if t in c) / len(tokens)


def tfidf_scores(query: str, docs: dict[str, str]) -> dict[str, float]:
    """docs: id -> content；返回 id -> 余弦相似度 [0,1]。"""
    if not docs:
        return {}
    q_tokens = tokenize(query)
    if not q_tokens:
        return {i: 0.0 for i in docs}
    doc_tokens = {i: tokenize(t) for i, t in docs.items()}
    df: Counter[str] = Counter()
    for toks in doc_tokens.values():
        df.update(set(toks))
    n = len(docs)
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}

    def vec(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        length = len(tokens) or 1
        return {t: (c / length) * idf.get(t, 0.0) for t, c in tf.items()}

    qv = vec(q_tokens)
    q_norm = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    out: dict[str, float] = {}
    for i, toks in doc_tokens.items():
        dv = vec(toks)
        d_norm = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        dot = sum(qv.get(t, 0.0) * dv.get(t, 0.0) for t in set(qv) | set(dv))
        out[i] = max(0.0, min(1.0, dot / (q_norm * d_norm)))
    return out


def recency_score(ts: datetime | str, decay: float = 0.1) -> float:
    try:
        t = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        age_hours = (datetime.now() - t).total_seconds() / 3600
        return max(0.1, math.exp(-decay * age_hours / 24))
    except Exception:
        return 0.5


def time_decay(ts: datetime, half_life_minutes: float = 30.0) -> float:
    age_min = max(0.0, (datetime.now() - ts).total_seconds() / 60)
    return math.exp(-math.log(2) * age_min / half_life_minutes)


def importance_weight(importance: float) -> float:
    return 0.8 + (float(importance) * 0.4)


def extract_entities(text: str) -> list[str]:
    """轻量实体抽取：英文专名 + 连续中文词段（2–6 字）。"""
    ents: list[str] = []
    ents.extend(re.findall(r"\b[A-Z][a-zA-Z0-9_]{1,}\b", text))
    ents.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", text))
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for e in ents:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out[:20]
