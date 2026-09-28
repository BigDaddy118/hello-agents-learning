"""文档载入 + Markdown 标题感知分块。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

_markitdown = None
_markitdown_tried = False


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def approx_token_len(text: str) -> int:
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk = len([t for t in text.split() if t])
    return cjk + non_cjk


def split_paragraphs_with_headings(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    heading_stack: list[str] = []
    paragraphs: list[dict[str, Any]] = []
    buf: list[str] = []
    char_pos = 0

    def flush_buf(end_pos: int) -> None:
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append(
            {
                "content": content,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos,
            }
        )

    for ln in lines:
        raw = ln
        if raw.strip().startswith("#"):
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip("#"))
            title = raw.lstrip("#").strip()
            level = max(1, level)
            if level <= len(heading_stack):
                heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            char_pos += len(raw) + 1
            continue
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1

    flush_buf(char_pos)
    if not paragraphs:
        paragraphs = [
            {"content": text, "heading_path": None, "start": 0, "end": len(text)}
        ]
    return paragraphs


def chunk_paragraphs(
    paragraphs: list[dict[str, Any]],
    chunk_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    cur_tokens = 0
    i = 0

    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = approx_token_len(p["content"]) or 1
        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
            continue
        chunks.append(_pack_chunk(cur))
        if overlap_tokens > 0 and cur:
            kept: list[dict[str, Any]] = []
            kept_tokens = 0
            for x in reversed(cur):
                t = approx_token_len(x["content"]) or 1
                if kept_tokens + t > overlap_tokens:
                    break
                kept.append(x)
                kept_tokens += t
            cur = list(reversed(kept))
            cur_tokens = kept_tokens
        else:
            cur, cur_tokens = [], 0

    if cur:
        chunks.append(_pack_chunk(cur))
    return chunks


def _pack_chunk(cur: list[dict[str, Any]]) -> dict[str, Any]:
    content = "\n\n".join(x["content"] for x in cur)
    return {
        "content": content,
        "start": cur[0]["start"],
        "end": cur[-1]["end"],
        "heading_path": next(
            (x["heading_path"] for x in reversed(cur) if x.get("heading_path")),
            None,
        ),
    }


def text_to_chunks(
    text: str, chunk_tokens: int = 256, overlap_tokens: int = 32
) -> list[dict[str, Any]]:
    return chunk_paragraphs(
        split_paragraphs_with_headings(text),
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )


def _get_markitdown_instance() -> Any:
    global _markitdown, _markitdown_tried
    if _markitdown_tried:
        return _markitdown
    _markitdown_tried = True
    try:
        from markitdown import MarkItDown

        _markitdown = MarkItDown()
    except Exception:
        _markitdown = None
    return _markitdown


def _fallback_text_reader(path: str) -> str:
    p = Path(path)
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def _enhanced_pdf_processing(path: str) -> str:
    """ponytail: 优先 pypdf；无则走 MarkItDown / 空。"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(p.strip() for p in parts if p and p.strip())
        if text.strip():
            print(f"[RAG] pypdf 提取成功: {path} -> {len(text)} chars")
            return text
    except Exception as e:
        print(f"[WARNING] pypdf 失败: {e}")
    md = _get_markitdown_instance()
    if md is None:
        return ""
    try:
        result = md.convert(path)
        text = getattr(result, "text_content", "") or ""
        return text if isinstance(text, str) else ""
    except Exception as e:
        print(f"[WARNING] MarkItDown PDF 失败: {e}")
        return ""


def convert_to_markdown(path: str) -> str:
    """任意常见格式 → Markdown/纯文本。"""
    if not os.path.exists(path):
        return ""
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext == ".pdf":
        return _enhanced_pdf_processing(path)
    if ext in {".md", ".txt", ".csv", ".json", ".xml", ".html", ".htm", ".py", ".js"}:
        return _fallback_text_reader(path)

    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)
    try:
        result = md_instance.convert(path)
        markdown_text = getattr(result, "text_content", None)
        if isinstance(markdown_text, str) and markdown_text.strip():
            print(
                f"[RAG] MarkItDown转换成功: {path} -> {len(markdown_text)} chars Markdown"
            )
            return markdown_text
        return ""
    except Exception as e:
        print(f"[WARNING] MarkItDown转换失败 {path}: {e}")
        return _fallback_text_reader(path)
