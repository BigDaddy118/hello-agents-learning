"""GAIA 一键评估工具（12.3）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hello_agents.evaluation.benchmarks.gaia.dataset import GAIADataset
from hello_agents.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator
from hello_agents.evaluation.benchmarks.gaia.metrics import GAIAMetrics

from ..base import Tool, ToolParameter


class GAIAEvaluationTool(Tool):
    """GAIA 一键评估：加载数据 → 跑 Agent → 导出 JSONL → 报告。"""

    def __init__(self, local_data_path: str | None = None):
        super().__init__(
            name="gaia_evaluation",
            description=(
                "评估智能体的通用AI助手能力（GAIA）。"
                "支持 Level 1/2/3，自动导出排行榜格式与报告。"
            ),
        )
        self.local_data_path = local_data_path
        self.dataset: GAIADataset | None = None
        self.evaluator: GAIAEvaluator | None = None
        self.metrics_calculator = GAIAMetrics()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="agent", type="object", description="智能体实例", required=True
            ),
            ToolParameter(
                name="level",
                type="integer",
                description="难度 1/2/3，省略=全部",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="max_samples",
                type="integer",
                description="最大样本数，省略=全部",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="local_data_dir",
                type="string",
                description="本地 GAIA 数据目录",
                required=False,
                default=None,
            ),
        ]

    def run(self, parameters: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """支持 run(agent=..., level=1, max_samples=5) 与 Tool 字典写法。"""
        params = {**(parameters or {}), **kwargs}
        agent = params.get("agent")
        if agent is None:
            return json.dumps(
                {"status": "error", "message": "缺少参数 agent"},
                ensure_ascii=False,
            )

        level = params.get("level")
        if level is not None:
            level = int(level)
        max_samples = params.get("max_samples")
        if max_samples is not None:
            max_samples = int(max_samples)
        local_data_dir = params.get("local_data_dir") or self.local_data_path
        export_results = bool(params.get("export_results", True))
        generate_report = bool(params.get("generate_report", True))

        print("\n" + "=" * 60)
        print("GAIA一键评估")
        print("=" * 60)
        print(f"\n配置:\n   智能体: {getattr(agent, 'name', 'Unknown')}")
        print(f"   难度级别: {level or '全部'}")
        print(f"   样本数量: {max_samples or '全部'}")

        try:
            print("\n" + "=" * 60)
            print("步骤1: 运行HelloAgents评估")
            print("=" * 60)
            results = self._run_evaluation(agent, level, max_samples, local_data_dir)

            if export_results:
                print("\n" + "=" * 60)
                print("步骤2: 导出GAIA格式结果")
                print("=" * 60)
                self._export_results(results)

            if generate_report:
                print("\n" + "=" * 60)
                print("步骤3: 生成评估报告")
                print("=" * 60)
                self.generate_report(results)

            print("\n" + "=" * 60)
            print("最终结果")
            print("=" * 60)
            print(f"   精确匹配率: {results['exact_match_rate']:.2%}")
            print(f"   部分匹配率: {results['partial_match_rate']:.2%}")
            print(
                f"   正确数: {results['exact_matches']}/{results['total_samples']}"
            )
            return results
        except Exception as e:
            print(f"\n[x] 评估失败: {e}")
            return {
                "error": str(e),
                "benchmark": "GAIA",
                "agent_name": getattr(agent, "name", "Unknown"),
                "exact_match_rate": 0.0,
                "partial_match_rate": 0.0,
                "exact_matches": 0,
                "total_samples": 0,
            }

    def _run_evaluation(
        self,
        agent: Any,
        level: int | None,
        max_samples: int | None,
        local_data_dir: str | None,
    ) -> dict[str, Any]:
        self.dataset = GAIADataset(level=level, local_data_dir=local_data_dir)
        if not self.dataset.load():
            raise ValueError(
                "数据集加载失败或为空（检查 HF_TOKEN 与 GAIA 访问权限）"
            )
        self.evaluator = GAIAEvaluator(
            dataset=self.dataset, level=level, local_data_dir=local_data_dir
        )
        return self.evaluator.evaluate(agent, max_samples)

    def _export_results(self, results: dict[str, Any]) -> None:
        assert self.evaluator is not None
        output_dir = Path("./evaluation_results/gaia_official")
        output_dir.mkdir(parents=True, exist_ok=True)
        level = results.get("level_filter")
        level_str = f"_level{level}" if level else "_all"
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"gaia{level_str}_result_{ts}.jsonl"
        self.evaluator.export_to_gaia_format(
            results, output_file, include_reasoning=True
        )
        self._generate_submission_guide(results, output_dir, output_file)

    def _generate_submission_guide(
        self, results: dict[str, Any], output_dir: Path, result_file: Path
    ) -> None:
        agent_name = results.get("agent_name", "Unknown")
        level = results.get("level_filter")
        guide = f"""# GAIA评估结果提交指南

## 评估结果摘要

- **模型名称**: {agent_name}
- **评估级别**: {level or "全部"}
- **总样本数**: {results.get("total_samples", 0)}
- **精确匹配数**: {results.get("exact_matches", 0)}
- **精确匹配率**: {results.get("exact_match_rate", 0):.2%}

## 提交文件

结果文件: `{result_file.name}`

## 如何提交

1. 打开 https://huggingface.co/spaces/gaia-benchmark/leaderboard
2. 填写 Model Name: `{agent_name}`
3. 上传 `{result_file.name}`（JSONL）
4. Submit

## 结果格式

```json
{{"task_id": "xxx", "model_answer": "答案", "reasoning_trace": "推理过程"}}
```

生成时间: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC
"""
        guide_file = (
            output_dir
            / f"SUBMISSION_GUIDE_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.md"
        )
        guide_file.write_text(guide, encoding="utf-8")
        print(f"提交说明已生成: {guide_file}")

    def generate_report(
        self, results: dict[str, Any], output_file: str | Path | None = None
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        exact_rate = results.get("exact_match_rate", 0.0)
        partial_rate = results.get("partial_match_rate", 0.0)
        report = f"""# GAIA评估报告

**生成时间**: {ts} UTC

## 评估概览

- **智能体**: {results.get("agent_name", "Unknown")}
- **难度级别**: {results.get("level_filter") or "全部"}
- **总样本数**: {results.get("total_samples", 0)}
- **精确匹配数**: {results.get("exact_matches", 0)}
- **部分匹配数**: {results.get("partial_matches", 0)}
- **精确匹配率**: {exact_rate:.2%}
- **部分匹配率**: {partial_rate:.2%}

## 详细指标

### 分级准确率

"""
        for level_name, metrics in (results.get("level_metrics") or {}).items():
            report += (
                f"- **{level_name.replace('Level_', 'Level ')}**: "
                f"{metrics.get('exact_match_rate', 0):.2%} 精确 / "
                f"{metrics.get('partial_match_rate', 0):.2%} 部分 "
                f"({metrics.get('exact_matches', 0)}/{metrics.get('total', 0)})\n"
            )

        report += "\n## 样本详情（前10个）\n\n"
        report += "| 任务ID | 级别 | 预测答案 | 正确答案 | 精确匹配 | 部分匹配 |\n"
        report += "|--------|------|----------|----------|----------|----------|\n"
        for detail in (results.get("detailed_results") or [])[:10]:
            exact = "[ok]" if detail.get("exact_match") else "[x]"
            partial = "[ok]" if detail.get("partial_match") else "[x]"
            report += (
                f"| {detail.get('task_id', '')} | {detail.get('level', '')} | "
                f"{str(detail.get('predicted', ''))[:50]} | "
                f"{str(detail.get('expected', ''))[:50]} | {exact} | {partial} |\n"
            )

        bar = int(exact_rate * 50)
        bar_p = int(partial_rate * 50)
        report += (
            "\n## 准确率可视化\n\n```\n"
            f"精确匹配: {'█' * bar}{'░' * (50 - bar)} {exact_rate:.2%}\n"
            f"部分匹配: {'█' * bar_p}{'░' * (50 - bar_p)} {partial_rate:.2%}\n"
            "```\n\n## 建议\n\n"
        )
        if exact_rate >= 0.9:
            report += "- 表现优秀。\n"
        elif exact_rate >= 0.7:
            report += "- 表现良好，可优化提示词与推理策略。\n"
        elif exact_rate >= 0.5:
            report += "- 表现一般，检查工具使用与多步推理。\n"
        else:
            report += "- 建议从 Level 1 小样本开始调试。\n"

        if output_file is None:
            out_dir = Path("./evaluation_reports")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"gaia_report_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.md"
        else:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"报告已生成: {out_path}")
        return report
