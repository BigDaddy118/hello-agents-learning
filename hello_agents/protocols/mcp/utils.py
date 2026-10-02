"""MCP 上下文辅助。"""

from __future__ import annotations

import json
from typing import Any


def create_context(
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    resources: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "messages": messages or [],
        "tools": tools or [],
        "resources": resources or [],
        "metadata": metadata or {},
    }


def parse_context(context: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON context: {e}") from e
    if not isinstance(context, dict):
        raise ValueError("Context must be a dictionary or JSON string")
    for field in ("messages", "tools", "resources"):
        context.setdefault(field, [])
    context.setdefault("metadata", {})
    return context
