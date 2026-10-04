"""12.3.3 方式1：GAIAEvaluationTool 一键评估。"""

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.tools import GAIAEvaluationTool

load_dotenv(encoding="utf-8")

GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.

If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.

If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.

If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

llm = HelloAgentsLLM()
agent = SimpleAgent(
    name="TestAgent",
    llm=llm,
    system_prompt=GAIA_SYSTEM_PROMPT,
)

gaia_tool = GAIAEvaluationTool()
results = gaia_tool.run(
    agent=agent,
    level=1,
    max_samples=5,
    export_results=True,
    generate_report=True,
)

if "error" in results:
    print("失败:", results["error"])
else:
    print(f"精确匹配率: {results['exact_match_rate']:.2%}")
    print(f"部分匹配率: {results['partial_match_rate']:.2%}")
    print(f"正确数: {results['exact_matches']}/{results['total_samples']}")
