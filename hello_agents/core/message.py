"""消息系统（7.3.1）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """统一消息格式；对外 to_dict() 兼容 OpenAI API。"""

    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, content: str, role: MessageRole, **kwargs: Any):
        # 支持 Message("hi", "user") 位置参数（章节用法）
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
