"""Win Rate 一键评估工具（12.4）。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.evaluation.benchmarks.data_generation.dataset import AIDataset
from hello_agents.evaluation.benchmarks.data_generation.win_rate import (
    WinRateEvaluator,
)

from ..base import Tool, ToolParameter


class WinRateTool(Tool):
    """生成题 vs 真题成对对比，算 Win/Loss/Tie。"""

    def __init__(self, llm: HelloAgentsLLM | None = None):
        super().__init__(
            name="win_rate_evaluation",
            description="通过成对对比计算生成数据相对于真题的胜率",
        )
        self.llm = llm

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="generated_data_path",
                type="string",
                description="生成数据 JSON 路径",
                required=True,
            ),
            ToolParameter(
                name="reference_data_path",
                type="string",
                description="本地参考数据 JSON（可选）",
                required=False,
            ),
            ToolParameter(
                name="reference_year",
                type="integer",
                description="AIME 真题年份，如 2025",
                required=False,
                default=2025,
            ),
            ToolParameter(
                name="num_comparisons",
                type="integer",
                description="对比次数",
                required=False,
            ),
            ToolParameter(
                name="output_dir",
                type="string",
                description="输出目录",
                required=False,
                default="evaluation_results/win_rate",
            ),
            ToolParameter(
                name="judge_model",
                type="string",
                description="评委模型名（仅展示用）",
                required=False,
                default="gpt-4o",
            ),
        ]

    def run(self, parameters: dict[str, Any] | None = None, **kwargs: Any) -> str:
        params = {**(parameters or {}), **kwargs}
        generated_data_path = params["generated_data_path"]
        reference_data_path = params.get("reference_data_path")
        reference_year = params.get("reference_year", 2025)
        num_comparisons = params.get("num_comparisons")
        output_dir = params.get("output_dir", "evaluation_results/win_rate")
        judge_model = (
            getattr(self.llm, "model", None) or params.get("judge_model") or "unknown"
        )

        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 60)
        print("Win Rate evaluation")
        print("=" * 60)

        gen_problems = AIDataset(
            dataset_type="generated", data_path=generated_data_path
        ).load()

        if reference_data_path:
            ref_problems = AIDataset(
                dataset_type="generated", data_path=reference_data_path
            ).load()
        else:
            ref_problems = AIDataset(
                dataset_type="real", year=int(reference_year)
            ).load()

        evaluator = WinRateEvaluator(llm=self.llm, judge_model=judge_model)
        results = evaluator.evaluate_win_rate(
            gen_problems,
            ref_problems,
            num_comparisons=int(num_comparisons) if num_comparisons else None,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"win_rate_results_{timestamp}.json")
        report_file = os.path.join(output_dir, f"win_rate_report_{timestamp}.md")
        evaluator.export_results(results, result_file)
        self._generate_report(results, report_file)

        print(f"[ok] results: {result_file}")
        print(f"[ok] report: {report_file}")
        return json.dumps(
            {
                "status": "success",
                "metrics": results["metrics"],
                "result_file": result_file,
                "report_file": report_file,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _generate_report(self, results: dict[str, Any], output_path: str) -> None:
        m = results["metrics"]
        wr = m["win_rate"]
        if 0.45 <= wr <= 0.55:
            analysis = "接近真题水平（理想区间）"
        elif wr > 0.55:
            analysis = "高于真题（可能评估偏差）"
        else:
            analysis = "低于真题，需改进生成"
        report = f"""# Win Rate评估报告

## 基本信息

- **评估日期**: {results['evaluation_date']}
- **评委模型**: {results['judge_model']}
- **对比次数**: {m['total_comparisons']}

## 结果

| 指标 | 次数 | 比例 |
|------|------|------|
| Win | {m['wins']} | {m['win_rate']:.2%} |
| Loss | {m['losses']} | {m['loss_rate']:.2%} |
| Tie | {m['ties']} | {m['tie_rate']:.2%} |

分析: {analysis}
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
