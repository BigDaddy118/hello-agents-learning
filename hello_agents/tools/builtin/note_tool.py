"""NoteTool — 结构化笔记（9.4）：Markdown + YAML，持久化到工作目录。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import Tool, ToolParameter

_NOTE_TYPES = frozenset(
    {"task_state", "conclusion", "blocker", "action", "reference", "general"}
)


def _now_iso() -> str:
    return datetime.now().isoformat()  # noqa: DTZ005


class NoteTool(Tool):
    """结构化笔记：create / read / update / delete / list / search / summary。"""

    def __init__(self, workspace: str = "./notes", max_notes: int = 1000):
        super().__init__(
            name="note",
            description=(
                "结构化笔记工具：创建/读取/更新/删除笔记，"
                "支持 task_state、conclusion、blocker、action、reference、general"
            ),
        )
        self.workspace = Path(workspace)
        self.max_notes = max_notes
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.index_file = self.workspace / "notes_index.json"
        self.index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def run(self, parameters: dict[str, Any]) -> str:
        action = parameters.get("action") or parameters.get("input") or "summary"
        kwargs = {k: v for k, v in parameters.items() if k not in ("action", "input")}
        return self.execute(str(action), **kwargs)

    def execute(self, action: str, **kwargs: Any) -> str:
        handlers = {
            "create": self._create,
            "read": self._read,
            "update": self._update,
            "delete": self._delete,
            "list": self._list,
            "search": self._search,
            "summary": self._summary,
        }
        fn = handlers.get(action)
        if not fn:
            return (
                f"❌ 不支持的操作: {action}。"
                "支持: create, read, update, delete, list, search, summary"
            )
        return fn(**kwargs)

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="create|read|update|delete|list|search|summary",
                required=True,
            ),
            ToolParameter(name="title", type="string", description="标题", required=False),
            ToolParameter(
                name="content", type="string", description="正文", required=False
            ),
            ToolParameter(
                name="note_type",
                type="string",
                description="task_state|conclusion|blocker|action|reference|general",
                required=False,
                default="general",
            ),
            ToolParameter(name="tags", type="array", description="标签", required=False),
            ToolParameter(
                name="note_id", type="string", description="笔记 ID", required=False
            ),
            ToolParameter(
                name="query", type="string", description="搜索词", required=False
            ),
            ToolParameter(
                name="limit", type="integer", description="条数上限", required=False, default=10
            ),
        ]

    # --- structured helpers（给 Agent / ContextBuilder 用）---

    def create_note(
        self,
        title: str,
        content: str,
        note_type: str = "general",
        tags: list[str] | None = None,
    ) -> str:
        note_id = self._do_create(title, content, note_type, tags or [])
        return note_id

    def list_notes(
        self,
        note_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self._filter_index(note_type=note_type, tags=tags, limit=limit)

    def search_notes(
        self,
        query: str,
        limit: int = 10,
        note_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        q = query.lower()
        results: list[dict[str, Any]] = []
        for meta in self._filter_index(note_type=note_type, tags=tags, limit=10_000):
            try:
                note = self._read_raw(meta["id"])
            except (OSError, ValueError, KeyError):
                continue
            title = note.get("title", "")
            body = note.get("content", "")
            if q in title.lower() or q in body.lower():
                results.append(
                    {
                        "note_id": note["id"],
                        "title": title,
                        "type": note.get("type", "general"),
                        "tags": note.get("tags", []),
                        "content": body,
                        "updated_at": note.get("updated_at", ""),
                    }
                )
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    def summary_dict(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        for meta in self.index.values():
            t = meta.get("type", "general")
            type_counts[t] = type_counts.get(t, 0) + 1
        recent = sorted(
            self.index.values(),
            key=lambda m: m.get("updated_at", m.get("created_at", "")),
            reverse=True,
        )[:5]
        return {
            "total_notes": len(self.index),
            "type_distribution": type_counts,
            "recent_notes": [
                {
                    "id": n["id"],
                    "title": n.get("title", ""),
                    "type": n.get("type"),
                    "updated_at": n.get("updated_at", n.get("created_at")),
                }
                for n in recent
            ],
        }

    def get_note(self, note_id: str) -> dict[str, Any]:
        """读取笔记为结构化字典。"""
        note = self._read_raw(note_id)
        return {
            "note_id": note["id"],
            "title": note.get("title", ""),
            "type": note.get("type", "general"),
            "tags": note.get("tags", []),
            "content": note.get("content", ""),
            "updated_at": note.get("updated_at", ""),
            "created_at": note.get("created_at", ""),
        }

    # --- run handlers ---

    def _create(
        self,
        title: str = "",
        content: str = "",
        note_type: str = "general",
        tags: list[str] | None = None,
        **_: Any,
    ) -> str:
        if not title or not content:
            return "❌ 创建笔记需要提供 title 和 content"
        try:
            note_id = self._do_create(title, content, note_type, tags or [])
        except ValueError as e:
            return f"❌ {e}"
        return f"✅ 笔记创建成功\nID: {note_id}\n标题: {title}\n类型: {note_type}"

    def _read(self, note_id: str = "", **_: Any) -> str:
        if not note_id:
            return "❌ 读取笔记需要提供 note_id"
        try:
            note = self._read_raw(note_id)
        except (OSError, ValueError, KeyError) as e:
            return f"❌ {e}"
        return self._format_note(note)

    def _update(
        self,
        note_id: str = "",
        title: str | None = None,
        content: str | None = None,
        note_type: str | None = None,
        tags: list[str] | None = None,
        **_: Any,
    ) -> str:
        if not note_id:
            return "❌ 更新笔记需要提供 note_id"
        try:
            note = self._read_raw(note_id)
            if title is not None:
                note["title"] = title
            if content is not None:
                note["content"] = content
            if note_type is not None:
                note["type"] = note_type if note_type in _NOTE_TYPES else "general"
            if tags is not None:
                note["tags"] = tags if isinstance(tags, list) else []
            note["updated_at"] = _now_iso()
            self._write_file(note)
            self.index[note_id] = self._index_entry(note)
            self._save_index()
        except (OSError, ValueError, KeyError) as e:
            return f"❌ {e}"
        return f"✅ 笔记已更新: {note['title']}"

    def _delete(self, note_id: str = "", **_: Any) -> str:
        if not note_id:
            return "❌ 删除笔记需要提供 note_id"
        if note_id not in self.index:
            return f"❌ 笔记不存在: {note_id}"
        path = Path(self.index[note_id]["file_path"])
        title = self.index[note_id].get("title", note_id)
        if path.exists():
            path.unlink()
        del self.index[note_id]
        self._save_index()
        return f"✅ 笔记已删除: {title}"

    def _list(
        self,
        note_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        **_: Any,
    ) -> str:
        notes = self._filter_index(note_type=note_type, tags=tags, limit=limit)
        if not notes:
            return "📝 暂无笔记"
        lines = [f"📝 笔记列表（共 {len(notes)} 条）", ""]
        for n in notes:
            lines.append(f"• [{n.get('type')}] {n.get('title')}")
            lines.append(f"  ID: {n['id']}")
            if n.get("tags"):
                lines.append(f"  标签: {', '.join(n['tags'])}")
            lines.append(f"  更新: {n.get('updated_at', n.get('created_at', ''))}")
            lines.append("")
        return "\n".join(lines)

    def _search(
        self,
        query: str = "",
        limit: int = 10,
        note_type: str | None = None,
        tags: list[str] | None = None,
        **_: Any,
    ) -> str:
        if not query:
            return "❌ 搜索需要提供 query"
        matched = self.search_notes(query, limit=limit, note_type=note_type, tags=tags)
        if not matched:
            return f"📝 未找到匹配 '{query}' 的笔记"
        lines = [f"🔍 搜索结果（共 {len(matched)} 条）", ""]
        for n in matched:
            preview = n["content"][:100] + ("..." if len(n["content"]) > 100 else "")
            lines.append(f"[{n['type']}] {n['title']}")
            lines.append(f"ID: {n['note_id']}")
            lines.append(f"内容: {preview}")
            lines.append("")
        return "\n".join(lines)

    def _summary(self, **_: Any) -> str:
        data = self.summary_dict()
        lines = [
            "📊 笔记摘要",
            "",
            f"总笔记数: {data['total_notes']}",
            "",
            "按类型统计:",
        ]
        for t, c in sorted(data["type_distribution"].items()):
            lines.append(f"  • {t}: {c}")
        if data["recent_notes"]:
            lines.append("")
            lines.append("最近更新:")
            for n in data["recent_notes"]:
                lines.append(f"  • [{n['type']}] {n['title']} ({n['updated_at']})")
        return "\n".join(lines)

    # --- persistence ---

    def _do_create(
        self, title: str, content: str, note_type: str, tags: list[str]
    ) -> str:
        if len(self.index) >= self.max_notes:
            raise ValueError(f"笔记数量已达上限 ({self.max_notes})")
        ntype = note_type if note_type in _NOTE_TYPES else "general"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        note_id = f"note_{stamp}_{len(self.index)}"
        now = _now_iso()
        note = {
            "id": note_id,
            "title": title,
            "content": content,
            "type": ntype,
            "tags": list(tags),
            "created_at": now,
            "updated_at": now,
        }
        path = self.workspace / f"{note_id}.md"
        note["file_path"] = str(path)
        self._write_file(note)
        self.index[note_id] = self._index_entry(note)
        self._save_index()
        return note_id

    def _index_entry(self, note: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": note["id"],
            "title": note["title"],
            "type": note["type"],
            "tags": note.get("tags", []),
            "created_at": note["created_at"],
            "updated_at": note["updated_at"],
            "file_path": note.get("file_path") or str(self.workspace / f"{note['id']}.md"),
        }

    def _filter_index(
        self,
        note_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for meta in self.index.values():
            if note_type and meta.get("type") != note_type:
                continue
            if tags:
                if not set(meta.get("tags", [])).intersection(tags):
                    continue
            results.append(meta)
        results.sort(
            key=lambda m: m.get("updated_at", m.get("created_at", "")),
            reverse=True,
        )
        return results[:limit]

    def _read_raw(self, note_id: str) -> dict[str, Any]:
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")
        path = Path(self.index[note_id]["file_path"])
        if not path.exists():
            raise FileNotFoundError(f"笔记文件缺失: {path}")
        text = path.read_text(encoding="utf-8")
        note = self._parse_markdown(text)
        note["file_path"] = str(path)
        return note

    def _write_file(self, note: dict[str, Any]) -> None:
        path = Path(note.get("file_path") or (self.workspace / f"{note['id']}.md"))
        path.write_text(self._build_markdown(note), encoding="utf-8")
        note["file_path"] = str(path)

    def _build_markdown(self, note: dict[str, Any]) -> str:
        tags = note.get("tags") or []
        header = (
            f"---\n"
            f"id: {note['id']}\n"
            f"title: {note['title']}\n"
            f"type: {note['type']}\n"
            f"tags: {json.dumps(tags, ensure_ascii=False)}\n"
            f"created_at: {note['created_at']}\n"
            f"updated_at: {note['updated_at']}\n"
            f"---\n\n"
            f"# {note['title']}\n\n"
            f"{note['content']}"
        )
        return header

    def _parse_markdown(self, raw: str) -> dict[str, Any]:
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", raw, re.DOTALL)
        if not m:
            raise ValueError("无效的笔记格式：缺少 YAML 前置元数据")
        meta: dict[str, Any] = {}
        for line in m.group(1).split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if key == "tags":
                try:
                    meta[key] = json.loads(value)
                except json.JSONDecodeError:
                    meta[key] = []
            else:
                meta[key] = value
        body = raw[m.end() :].strip()
        lines = body.split("\n")
        if lines and lines[0].startswith("# "):
            body = "\n".join(lines[1:]).strip()
        meta["content"] = body
        return meta

    def _format_note(self, note: dict[str, Any]) -> str:
        lines = [
            "📝 笔记详情",
            "",
            f"ID: {note['id']}",
            f"标题: {note['title']}",
            f"类型: {note.get('type', 'general')}",
        ]
        if note.get("tags"):
            lines.append(f"标签: {', '.join(note['tags'])}")
        lines.append(f"创建时间: {note.get('created_at', '')}")
        lines.append(f"更新时间: {note.get('updated_at', '')}")
        lines.append("")
        lines.append("内容:")
        lines.append(note.get("content", ""))
        return "\n".join(lines)

    def _load_index(self) -> None:
        if not self.index_file.exists():
            self.index = {}
            self._save_index()
            return
        data = json.loads(self.index_file.read_text(encoding="utf-8"))
        # 兼容官方 list 格式与章节 dict 格式
        if isinstance(data, dict) and "notes" in data and isinstance(data["notes"], list):
            self.index = {n["id"]: n for n in data["notes"] if "id" in n}
            for nid, meta in self.index.items():
                meta.setdefault("file_path", str(self.workspace / f"{nid}.md"))
                meta.setdefault("updated_at", meta.get("created_at", ""))
        elif isinstance(data, dict):
            self.index = {k: v for k, v in data.items() if isinstance(v, dict) and "id" in v}
        else:
            self.index = {}

    def _save_index(self) -> None:
        self.index_file.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
