"""ToolAwareSimpleAgent — SimpleAgent + 工具调用监听（第十四章）。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .simple_agent import SimpleAgent

logger = logging.getLogger(__name__)


class ToolAwareSimpleAgent(SimpleAgent):
    """在 SimpleAgent 上增加 tool_call_listener，便于 SSE/日志追踪工具调用。"""

    def __init__(
        self,
        *args: Any,
        tool_call_listener: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tool_call_listener = tool_call_listener

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        parsed_parameters: dict[str, Any] | str = parameters
        try:
            parsed_parameters = self._parse_tool_parameters(tool_name, parameters)
        except Exception:  # noqa: BLE001
            parsed_parameters = parameters

        result = super()._execute_tool_call(tool_name, parameters)

        if self._tool_call_listener:
            try:
                self._tool_call_listener(
                    {
                        "agent_name": self.name,
                        "tool_name": tool_name,
                        "raw_parameters": parameters,
                        "parsed_parameters": parsed_parameters,
                        "result": result,
                    }
                )
            except Exception:  # pragma: no cover
                logger.exception("Tool call listener failed")

        return result
