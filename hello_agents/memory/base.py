"""记忆基础数据结构（8.2）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4


@dataclass
class MemoryItem:
    content: str
    memory_type: str = "working"
    importance: float = 0.5
    id: str = field(default_factory=lambda: uuid4().hex)
    user_id: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    working_memory_capacity: int = 50
    working_memory_ttl: int = 60  # 分钟
    database_path: str = "./memory_data/memory.db"


class BaseMemory(ABC):
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        storage_backend: Any = None,
    ):
        self.config = config or MemoryConfig()
        self.storage_backend = storage_backend

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str: ...

    @abstractmethod
    def retrieve(
        self, query: str, limit: int = 5, **kwargs: Any
    ) -> list[MemoryItem]: ...

    def all_items(self) -> list[MemoryItem]:
        return []

    def remove(self, memory_id: str) -> bool:
        return False

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        **metadata: Any,
    ) -> bool:
        return False

    def clear(self) -> int:
        return 0
