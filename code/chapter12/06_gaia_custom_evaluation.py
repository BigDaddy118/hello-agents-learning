"""12.3.3 方式2：GAIADataset + GAIAEvaluator。"""

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.evaluation import GAIADataset, GAIAEvaluator

load_dotenv(encoding="utf-8")

GAIA_SYSTEM_PROMPT = (
    "You are a general AI assistant. Report your thoughts, and finish with: "
    "FINAL ANSWER: [YOUR FINAL ANSWER]."
)

llm = HelloAgentsLLM()
agent = SimpleAgent(name="TestAgent", llm=llm, system_prompt=GAIA_SYSTEM_PROMPT)

dataset = GAIADataset(level=1, split="validation")
items = dataset.load()
print(f"加载了 {len(items)} 个样本")

evaluator = GAIAEvaluator(dataset=dataset, level=1)
results = evaluator.evaluate(agent, max_samples=5)

print(f"精确匹配率: {results['exact_match_rate']:.2%}")
print(f"正确数: {results['exact_matches']}/{results['total_samples']}")

evaluator.export_to_gaia_format(
    results, "evaluation_results/gaia_custom_result.jsonl", include_reasoning=True
)
