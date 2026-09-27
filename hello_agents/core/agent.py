"""Agent 抽象基类（7.3.3）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .config import Config
from .llm import HelloAgentsLLM
from .message import Message


class Agent(ABC):
    """所有智能体的统一接口。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent。"""

    def add_message(self, message: Message) -> None:
        self._history.append(message)
        max_len = self.config.max_history_length
        if max_len and len(self._history) > max_len:
            self._history = self._history[-max_len:]

    def clear_history(self) -> None:
        self._history.clear()

    def get_history(self) -> list[Message]:
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"


def _demo() -> None:
    from .config import Config
    from .message import Message

    m = Message("hello", "user")
    assert m.to_dict() == {"role": "user", "content": "hello"}
    assert "user" in str(m)

    cfg = Config.from_env()
    assert cfg.temperature >= 0
    assert "debug" in cfg.to_dict()

    try:
        Agent("x", llm=None)  # type: ignore[arg-type]
        raise AssertionError("Agent should be abstract")
    except TypeError:
        pass

    class EchoAgent(Agent):
        def run(self, input_text: str, **kwargs) -> str:
            self.add_message(Message(input_text, "user"))
            out = f"echo:{input_text}"
            self.add_message(Message(out, "assistant"))
            return out

    class _FakeLLM:
        provider = "fake"

    agent = EchoAgent("echo", llm=_FakeLLM(), config=Config(max_history_length=2))  # type: ignore[arg-type]
    assert agent.run("a") == "echo:a"
    agent.run("b")
    agent.run("c")
    assert len(agent.get_history()) == 2  # 被 max_history_length 截断
    print(f"✅ 7.3 ok: {agent}")


if __name__ == "__main__":
    _demo()
