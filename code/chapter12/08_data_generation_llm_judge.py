"""12.4：LLM Judge 评估（对已有生成题打四维分）。"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM
from hello_agents.tools import LLMJudgeTool

load_dotenv(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data_generation" / "generated_data" / "aime_generated_sample.json"
OUT = Path("evaluation_results") / "llm_judge"

# 小样：只评前 2 题，省 token
llm = HelloAgentsLLM()
tool = LLMJudgeTool(llm=llm)
raw = tool.run(
    {
        "generated_data_path": str(SAMPLE),
        "max_samples": 2,
        "output_dir": str(OUT),
    }
)
result = json.loads(raw)
m = result["metrics"]
print(f"平均总分: {m['average_total_score']:.2f}/5.0")
print(f"通过率: {m['pass_rate']:.2%}")
print(f"报告: {result['report_file']}")
