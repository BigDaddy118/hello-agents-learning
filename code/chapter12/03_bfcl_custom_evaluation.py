"""12.2.3 方式3：BFCLDataset + BFCLEvaluator 自定义评估。"""

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.evaluation import BFCLDataset, BFCLEvaluator

load_dotenv(encoding="utf-8")

llm = HelloAgentsLLM()
agent = SimpleAgent(name="TestAgent", llm=llm)

dataset = BFCLDataset(
    bfcl_data_dir="temp_gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
    category="simple_python",
)
dataset.load()

evaluator = BFCLEvaluator(dataset=dataset, category="simple_python", evaluation_mode="ast")
results = evaluator.evaluate(agent, max_samples=5)

print(f"准确率: {results['overall_accuracy']:.2%}")
print(f"正确数: {results['correct_samples']}/{results['total_samples']}")

evaluator.export_to_bfcl_format(
    results, output_path="./evaluation_results/bfcl_custom_result.json"
)
