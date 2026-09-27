"""ReActAgent（7.4.2）。"""

from __future__ import annotations

import re
from typing import Optional

from ..core.agent import Agent
from ..core.config import Config
from ..core.llm import HelloAgentsLLM
from ..core.message import Message
from ..tools.registry import ToolRegistry

DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

**Thought:** 分析当前问题，思考需要什么信息或采取什么行动。
**Action:** 选择一个行动，格式必须是以下之一：
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动："""


class ReActAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: list[str] = []
        self.prompt_template = custom_prompt or DEFAULT_REACT_PROMPT

    def run(self, input_text: str, **kwargs) -> str:
        self.current_history = []
        for step in range(1, self.max_steps + 1):
            print(f"\n--- 第 {step} 步 ---")
            prompt = self.prompt_template.format(
                tools=self.tool_registry.get_tools_description(),
                question=input_text,
                history="\n".join(self.current_history),
            )
            response = self.llm.invoke(
                [{"role": "user", "content": prompt}], **kwargs
            )
            if not response:
                break
            thought, action = self._parse_output(response)
            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                break
            if action.startswith("Finish"):
                answer = self._parse_action_input(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(answer, "assistant"))
                return answer
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: 无效的Action格式，请检查。")
                continue
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 观察: {observation}")
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(answer, "assistant"))
        return answer

    def _parse_output(self, text: str) -> tuple[Optional[str], Optional[str]]:
        thought_m = re.search(r"\*?\*?\s*Thought:\*?\*?\s*(.*)", text, re.I)
        action_m = re.search(r"\*?\*?\s*Action:\*?\*?\s*(.*)", text, re.I)
        thought = thought_m.group(1).strip().strip("*").strip() if thought_m else None
        action = action_m.group(1).strip().strip("*").strip() if action_m else None
        # 兜底：整段里直接抓 tool[...] / Finish[...]
        if not action:
            finish = re.search(r"Finish\[(.*?)\]", text, re.S)
            if finish:
                action = f"Finish[{finish.group(1)}]"
            else:
                tool = re.search(r"(\w+)\[([^\]]*)\]", text)
                if tool and tool.group(1).lower() != "thought":
                    action = f"{tool.group(1)}[{tool.group(2)}]"
        return thought, action

    def _parse_action(self, action_text: str) -> tuple[Optional[str], Optional[str]]:
        cleaned = action_text.strip().strip("`").strip()
        m = re.search(r"(\w+)\[(.*)\]", cleaned)
        return (m.group(1), m.group(2)) if m else (None, None)

    def _parse_action_input(self, action_text: str) -> str:
        m = re.search(r"\w+\[(.*)\]", action_text.strip().strip("`"))
        return m.group(1) if m else ""
