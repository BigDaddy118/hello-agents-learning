"""协议基类（概念性，各协议独立实现）。"""

from __future__ import annotations

from enum import Enum


class ProtocolType(Enum):
    MCP = "mcp"
    A2A = "a2a"
    ANP = "anp"


class Protocol:
    def __init__(self, protocol_type: ProtocolType, version: str = "1.0.0"):
        self._protocol_type = protocol_type
        self._version = version

    @property
    def protocol_name(self) -> str:
        return self._protocol_type.value

    @property
    def version(self) -> str:
        return self._version

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(protocol={self.protocol_name}, version={self.version})"
