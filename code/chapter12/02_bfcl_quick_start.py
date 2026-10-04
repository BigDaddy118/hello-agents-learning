"""12.2.3 方式1：BFCLEvaluationTool 一键评估。"""

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.tools import BFCLEvaluationTool

load_dotenv(encoding="utf-8")

llm = HelloAgentsLLM()
agent = SimpleAgent(name="TestAgent", llm=llm)
bfcl_tool = BFCLEvaluationTool()

results = bfcl_tool.run(
    agent=agent,
    category="simple_python",
    max_samples=5,
    run_official_eval=False,  # 先不跑官方 CLI；需要时改 True
)

print(f"准确率: {results['overall_accuracy']:.2%}")
print(f"正确数: {results['correct_samples']}/{results['total_samples']}")
