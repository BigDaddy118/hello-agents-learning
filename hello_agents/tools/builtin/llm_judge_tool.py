"""LLM Judge 一键评估工具（12.4）。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.evaluation.benchmarks.data_generation.dataset import AIDataset
from hello_agents.evaluation.benchmarks.data_generation.llm_judge import (
    LLMJudgeEvaluator,
)

from ..base import Tool, ToolParameter


class LLMJudgeTool(Tool):
    """用 LLM 从正确性/清晰度/难度/完整性四维打分。"""

    def __init__(self, llm: HelloAgentsLLM | None = None):
        super().__init__(
            name="llm_judge_evaluation",
            description="使用LLM作为评委评估数据生成质量",
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
            ),
            ToolParameter(
                name="max_samples",
                type="integer",
                description="最大评估样本数",
                required=False,
            ),
            ToolParameter(
                name="output_dir",
                type="string",
                description="输出目录",
                required=False,
                default="evaluation_results/llm_judge",
            ),
            ToolParameter(
                name="judge_model",
                type="string",
                description="评委模型名（仅展示用；实际用传入 llm）",
                required=False,
                default="gpt-4o",
            ),
        ]

    def run(self, parameters: dict[str, Any] | None = None, **kwargs: Any) -> str:
        params = {**(parameters or {}), **kwargs}
        generated_data_path = params["generated_data_path"]
        reference_data_path = params.get("reference_data_path")
        reference_year = params.get("reference_year")
        max_samples = params.get("max_samples")
        output_dir = params.get("output_dir", "evaluation_results/llm_judge")
        judge_model = (
            getattr(self.llm, "model", None) or params.get("judge_model") or "unknown"
        )

        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "=" * 60)
        print("LLM Judge evaluation")
        print("=" * 60)

        gen_dataset = AIDataset(
            dataset_type="generated", data_path=generated_data_path
        )
        gen_problems = gen_dataset.load()
        if max_samples:
            gen_problems = gen_problems[: int(max_samples)]
            print(f"   max_samples: {max_samples}")

        ref_problems = None
        if reference_data_path:
            ref_problems = AIDataset(
                dataset_type="generated", data_path=reference_data_path
            ).load()
        elif reference_year:
            ref_problems = AIDataset(
                dataset_type="real", year=int(reference_year)
            ).load()

        evaluator = LLMJudgeEvaluator(llm=self.llm, judge_model=judge_model)
        results = evaluator.evaluate_batch(gen_problems, ref_problems)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"llm_judge_results_{timestamp}.json")
        report_file = os.path.join(output_dir, f"llm_judge_report_{timestamp}.md")
        evaluator.export_results(results, result_file)
        self._generate_report(results, report_file)

        print(f"[ok] results: {result_file}")
        print(f"[ok] report: {report_file}")
        return json.dumps(
            {
                "status": "success",
                "metrics": results["metrics"],
                "num_problems": results["num_problems"],
                "result_file": result_file,
                "report_file": report_file,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _generate_report(self, results: dict[str, Any], output_path: str) -> None:
        metrics = results["metrics"]
        dims = metrics["dimension_averages"]
        report = f"""# LLM Judge评估报告

## 基本信息

- **评估日期**: {results['evaluation_date']}
- **评委模型**: {results['judge_model']}
- **评估数量**: {results['num_problems']} 个题目

## 评估结果

### 总体评分

- **平均总分**: {metrics['average_total_score']:.2f}/5.0
- **通过率**: {metrics['pass_rate']:.2%} (>=3.5)
- **优秀率**: {metrics['excellent_rate']:.2%} (>=4.5)

### 各维度评分

| 维度 | 平均分 |
|------|--------|
| 正确性 | {dims['correctness']:.2f}/5.0 |
| 清晰度 | {dims['clarity']:.2f}/5.0 |
| 难度匹配 | {dims['difficulty_match']:.2f}/5.0 |
| 完整性 | {dims['completeness']:.2f}/5.0 |
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
