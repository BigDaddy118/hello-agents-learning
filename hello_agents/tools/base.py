"""工具基类（7.5.1）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str: ...

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]: ...

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        required = [p.name for p in self.get_parameters() if p.required]
        return all(p in parameters for p in required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.model_dump() for p in self.get_parameters()],
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"
