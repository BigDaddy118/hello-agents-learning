"""ReflectionAgent（7.4.3）。"""

from __future__ import annotations

from typing import Any, Optional

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
""",
}


class ReflectionAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.prompts = custom_prompts or DEFAULT_PROMPTS
        self._records: list[dict[str, str]] = []

    def run(self, input_text: str, **kwargs: Any) -> str:
        self._records = []
        initial = self._ask(self.prompts["initial"].format(task=input_text), **kwargs)
        self._records.append({"type": "execution", "content": initial})

        for _ in range(self.max_iterations):
            last = self._last_execution()
            feedback = self._ask(
                self.prompts["reflect"].format(task=input_text, content=last), **kwargs
            )
            self._records.append({"type": "reflection", "content": feedback})
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                break
            refined = self._ask(
                self.prompts["refine"].format(
                    task=input_text, last_attempt=last, feedback=feedback
                ),
                **kwargs,
            )
            self._records.append({"type": "execution", "content": refined})

        final = self._last_execution()
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final, "assistant"))
        return final

    def _ask(self, prompt: str, **kwargs: Any) -> str:
        return self.llm.invoke([{"role": "user", "content": prompt}], **kwargs) or ""

    def _last_execution(self) -> str:
        for rec in reversed(self._records):
            if rec["type"] == "execution":
                return rec["content"]
        return ""
