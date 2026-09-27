import ast

from dotenv import load_dotenv

from llm_client import HelloAgentsLLM

# 加载 .env 文件中的环境变量
_ = load_dotenv()

PLANNER_PROMPT_TEMPLATE = (
    "你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。\n"
    "请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。\n"
    "你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。\n"
    "\n"
    "问题: {question}\n"
    "\n"
    "请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:\n"
    "```python\n"
    '["步骤1", "步骤2", "步骤3", ...]\n'
    "```\n"
)


class Planner:
    """把用户问题分解成有序步骤的规划器。"""

    def __init__(self, llm_client: HelloAgentsLLM) -> None:
        self.llm_client: HelloAgentsLLM = llm_client

    def plan(self, question: str) -> list[str]:
        """根据用户问题生成一个行动计划。"""
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            return _parse_plan(plan_str)
        except (ValueError, SyntaxError, IndexError, TypeError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []


def _parse_plan(plan_str: str) -> list[str]:
    body = ast.parse(plan_str, mode="eval").body
    if not isinstance(body, ast.List):
        raise TypeError("计划不是 Python 列表")
    steps: list[str] = []
    for element in body.elts:
        if not isinstance(element, ast.Constant):
            raise TypeError("计划中的步骤必须是字符串")
        value = element.value
        if not isinstance(value, str):
            raise TypeError("计划中的步骤必须是字符串")
        steps.append(value)
    return steps


EXECUTOR_PROMPT_TEMPLATE = (
    "你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。\n"
    "你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。\n"
    "请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。\n"
    "\n"
    "# 原始问题:\n"
    "{question}\n"
    "\n"
    "# 完整计划:\n"
    "{plan}\n"
    "\n"
    "# 历史步骤与结果:\n"
    "{history}\n"
    "\n"
    "# 当前步骤:\n"
    "{current_step}\n"
    "\n"
    "请仅输出针对“当前步骤”的回答:\n"
)


class Executor:
    """按计划逐步执行，并把每一步的结果传给下一步。"""

    def __init__(self, llm_client: HelloAgentsLLM) -> None:
        self.llm_client: HelloAgentsLLM = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """根据计划，逐步执行并解决问题。"""
        history = ""
        response_text = ""

        print("\n--- 正在执行计划 ---")
        for index, step in enumerate(plan):
            step_number = index + 1
            print(f"\n-> 正在执行步骤 {step_number}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages) or ""
            history += f"步骤 {step_number}: {step}\n结果: {response_text}\n\n"
            print(f"✅ 步骤 {step_number} 已完成，结果: {response_text}")

        return response_text


class PlanAndSolveAgent:
    """先规划，再执行。"""

    def __init__(self, llm_client: HelloAgentsLLM) -> None:
        self.llm_client: HelloAgentsLLM = llm_client
        self.planner: Planner = Planner(self.llm_client)
        self.executor: Executor = Executor(self.llm_client)

    def run(self, question: str) -> str | None:
        """运行智能体的完整流程:先规划，后执行。"""
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 ---\n无法生成有效的行动计划。")
            return None

        final_answer = self.executor.execute(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        return final_answer


if __name__ == "__main__":
    try:
        llm_client = HelloAgentsLLM()
        agent = PlanAndSolveAgent(llm_client)
        question = (
            "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。"
            "周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        )
        _ = agent.run(question)
    except ValueError as e:
        print(e)
