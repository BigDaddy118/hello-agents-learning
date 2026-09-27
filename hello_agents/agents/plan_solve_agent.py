"""PlanAndSolveAgent（7.4.4）。"""

from __future__ import annotations

import ast
from typing import Any, Optional

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class PlanAndSolveAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        custom_prompts = custom_prompts or {}
        self.planner_prompt = custom_prompts.get("planner") or DEFAULT_PLANNER_PROMPT
        self.executor_prompt = custom_prompts.get("executor") or DEFAULT_EXECUTOR_PROMPT

    def run(self, input_text: str, **kwargs: Any) -> str:
        plan = self._plan(input_text, **kwargs)
        if not plan:
            answer = "无法生成有效的行动计划，任务终止。"
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(answer, "assistant"))
            return answer
        answer = self._execute(input_text, plan, **kwargs)
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(answer, "assistant"))
        return answer

    def _plan(self, question: str, **kwargs: Any) -> list[str]:
        text = (
            self.llm.invoke(
                [
                    {
                        "role": "user",
                        "content": self.planner_prompt.format(question=question),
                    }
                ],
                **kwargs,
            )
            or ""
        )
        try:
            plan_str = text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except Exception:
            # 兜底：尝试整段 literal_eval
            try:
                plan = ast.literal_eval(text.strip())
                return plan if isinstance(plan, list) else []
            except Exception:
                return []

    def _execute(self, question: str, plan: list[str], **kwargs: Any) -> str:
        history = ""
        final = ""
        for i, step in enumerate(plan, 1):
            prompt = self.executor_prompt.format(
                question=question,
                plan=plan,
                history=history or "无",
                current_step=step,
            )
            final = (
                self.llm.invoke([{"role": "user", "content": prompt}], **kwargs) or ""
            )
            history += f"步骤 {i}: {step}\n结果: {final}\n\n"
        return final
