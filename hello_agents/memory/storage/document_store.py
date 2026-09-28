"""SQLite 文档存储（情景/感知持久化）。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class SQLiteDocumentStore:
    def __init__(self, database_path: str = "./memory_data/memory.db"):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()
        print(f"[OK] SQLite 文档存储初始化完成: {self.path}")

    def _init(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                memory_type TEXT,
                content TEXT,
                importance REAL,
                timestamp TEXT,
                metadata TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type, user_id)"
        )
        self._conn.commit()
        print("[OK] SQLite 数据库表和索引创建完成")

    def upsert(
        self,
        memory_id: str,
        user_id: str,
        memory_type: str,
        content: str,
        importance: float,
        timestamp: datetime,
        metadata: dict[str, Any],
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, user_id, memory_type, content, importance, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                user_id,
                memory_type,
                content,
                importance,
                timestamp.isoformat(),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def list_by_type(
        self, memory_type: str, user_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if user_id:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE memory_type=? AND user_id=?",
                (memory_type, user_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE memory_type=?",
                (memory_type,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def delete(self, memory_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def clear_type(self, memory_type: str, user_id: Optional[str] = None) -> int:
        if user_id:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE memory_type=? AND user_id=?",
                (memory_type, user_id),
            )
        else:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE memory_type=?", (memory_type,)
            )
        self._conn.commit()
        return cur.rowcount

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"],
            "user_id": r["user_id"],
            "memory_type": r["memory_type"],
            "content": r["content"],
            "importance": r["importance"],
            "timestamp": r["timestamp"],
            "metadata": json.loads(r["metadata"] or "{}"),
        }
