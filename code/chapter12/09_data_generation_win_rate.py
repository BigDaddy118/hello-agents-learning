"""12.4：Win Rate 评估（生成题 vs AIME 2025 真题）。"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from hello_agents import HelloAgentsLLM
from hello_agents.tools import WinRateTool

load_dotenv(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data_generation" / "generated_data" / "aime_generated_sample.json"
OUT = Path("evaluation_results") / "win_rate"

llm = HelloAgentsLLM()
tool = WinRateTool(llm=llm)
raw = tool.run(
    {
        "generated_data_path": str(SAMPLE),
        "reference_year": 2025,
        "num_comparisons": 2,
        "output_dir": str(OUT),
    }
)
result = json.loads(raw)
m = result["metrics"]
print(f"Win Rate: {m['win_rate']:.2%}")
print(f"Tie Rate: {m['tie_rate']:.2%}")
print(f"Loss Rate: {m['loss_rate']:.2%}")
print(f"报告: {result['report_file']}")
