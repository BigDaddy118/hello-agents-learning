"""BFCL 一键评估工具（12.2）。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..base import Tool, ToolParameter


class BFCLEvaluationTool(Tool):
    """BFCL 一键评估：加载数据 → 跑 Agent → 导出 → 可选官方 bfcl evaluate → 报告。"""

    def __init__(
        self,
        bfcl_data_dir: str | None = None,
        project_root: str | None = None,
    ):
        super().__init__(
            name="bfcl_evaluation",
            description=(
                "BFCL一键评估工具。评估智能体的工具调用能力。"
                "自动完成数据加载、评估、结果导出和报告生成。"
            ),
        )
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.bfcl_data_dir = (
            Path(bfcl_data_dir)
            if bfcl_data_dir
            else self.project_root
            / "temp_gorilla"
            / "berkeley-function-call-leaderboard"
            / "bfcl_eval"
            / "data"
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="agent",
                type="object",
                description="要评估的智能体实例",
                required=True,
            ),
            ToolParameter(
                name="category",
                type="string",
                description="评估类别，如 simple_python / multiple / parallel / irrelevance",
                required=False,
                default="simple_python",
            ),
            ToolParameter(
                name="max_samples",
                type="integer",
                description="样本数；0 表示全部",
                required=False,
                default=5,
            ),
            ToolParameter(
                name="run_official_eval",
                type="boolean",
                description="是否调用 bfcl evaluate",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="model_name",
                type="string",
                description="官方评估用的模型名",
                required=False,
                default="Qwen/Qwen3-8B",
            ),
        ]

    def run(self, parameters: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """支持章节写法 run(agent=..., category=...) 与 Tool 字典写法。"""
        params = {**(parameters or {}), **kwargs}
        agent = params.get("agent")
        if agent is None:
            return json.dumps(
                {"status": "error", "message": "缺少参数 agent"},
                ensure_ascii=False,
            )

        category = params.get("category", "simple_python")
        max_samples = int(params.get("max_samples", 5) or 0)
        run_official_eval = bool(params.get("run_official_eval", True))
        model_name = params.get("model_name") or "Qwen/Qwen3-8B"

        from hello_agents.evaluation import BFCLDataset, BFCLEvaluator

        print("\n" + "=" * 60)
        print("BFCL一键评估")
        print("=" * 60)
        print(f"\n配置:\n   评估类别: {category}")
        print(f"   样本数量: {max_samples if max_samples > 0 else '全部'}")
        print(f"   智能体: {getattr(agent, 'name', 'Unknown')}")

        if not self.bfcl_data_dir.exists():
            msg = (
                f"BFCL数据目录不存在: {self.bfcl_data_dir}\n"
                "请先: git clone --depth 1 https://github.com/ShishirPatil/gorilla.git temp_gorilla"
            )
            print(f"\n[x] {msg}")
            err = {
                "error": msg,
                "overall_accuracy": 0.0,
                "correct_samples": 0,
                "total_samples": 0,
                "category_metrics": {},
                "detailed_results": [],
            }
            return err if kwargs or not isinstance(parameters, dict) else json.dumps(err, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("步骤1: 运行HelloAgents评估")
        print("=" * 60)

        dataset = BFCLDataset(bfcl_data_dir=str(self.bfcl_data_dir), category=category)
        evaluator = BFCLEvaluator(dataset=dataset, category=category)
        results = evaluator.evaluate(
            agent, max_samples=max_samples if max_samples > 0 else None
        )

        print("\n 评估结果:")
        print(f"   准确率: {results['overall_accuracy']:.2%}")
        print(
            f"   正确数: {results['correct_samples']}/{results['total_samples']}"
        )

        print("\n" + "=" * 60)
        print("步骤2: 导出BFCL格式结果")
        print("=" * 60)

        output_dir = self.project_root / "evaluation_results" / "bfcl_official"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"BFCL_v4_{category}_result.json"
        evaluator.export_to_bfcl_format(results, output_file)

        if run_official_eval:
            self._run_official_evaluation(output_file, model_name, category)

        print("\n" + "=" * 60)
        print("步骤3: 生成评估报告")
        print("=" * 60)
        results["agent_name"] = getattr(agent, "name", "Unknown")
        results["category"] = category
        self.generate_report(results)
        return results

    def _run_official_evaluation(
        self, source_file: Path, model_name: str, category: str
    ) -> None:
        print("\n" + "=" * 60)
        print("步骤3: 运行BFCL官方评估")
        print("=" * 60)

        safe_model_name = model_name.replace("/", "_")
        result_dir = self.project_root / "result" / safe_model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        target_file = result_dir / f"BFCL_v4_{category}_result.json"
        shutil.copy(source_file, target_file)
        print(f"\n[ok] 结果文件已复制到:\n   {target_file}")

        try:
            os.environ["PYTHONUTF8"] = "1"
            cmd = [
                "bfcl",
                "evaluate",
                "--model",
                model_name,
                "--test-category",
                category,
                "--partial-eval",
            ]
            print(f"\n 运行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                print("\n[x] BFCL评估失败:")
                if result.stderr:
                    print(result.stderr)
            else:
                self._show_official_results(model_name, category)
        except FileNotFoundError:
            print("\n[x] 未找到bfcl命令\n   请先安装: pip install bfcl-eval")
        except Exception as e:
            print(f"\n[x] 运行BFCL评估时出错: {e}")

    def _show_official_results(self, model_name: str, category: str) -> None:
        print("\n" + "=" * 60)
        print("BFCL官方评估结果")
        print("=" * 60)
        csv_file = self.project_root / "score" / "data_non_live.csv"
        if csv_file.exists():
            print("\n 评估结果汇总:")
            print(csv_file.read_text(encoding="utf-8"))

        safe_model_name = model_name.replace("/", "_")
        score_file = (
            self.project_root
            / "score"
            / safe_model_name
            / "non_live"
            / f"BFCL_v4_{category}_score.json"
        )
        if score_file.exists():
            print(f"\n 详细评分文件:\n   {score_file}")
            first_line = score_file.read_text(encoding="utf-8").splitlines()[0]
            summary = json.loads(first_line)
            print("\n 最终结果:")
            print(f"   准确率: {summary['accuracy']:.2%}")
            print(
                f"   正确数: {summary['correct_count']}/{summary['total_count']}"
            )

    def generate_report(
        self, results: dict[str, Any], output_file: str | None = None
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        accuracy = results.get("overall_accuracy", 0.0)
        report = f"""# BFCL评估报告

**生成时间**: {timestamp}

## 评估概览

- **智能体**: {results.get("agent_name", "Unknown")}
- **评估类别**: {results.get("category", "Unknown")}
- **总体准确率**: {accuracy:.2%}
- **正确样本数**: {results.get("correct_samples", 0)}/{results.get("total_samples", 0)}

## 详细指标

"""
        cat_metrics = results.get("category_metrics") or {}
        if cat_metrics:
            report += "### 分类准确率\n\n"
            for category, metrics in cat_metrics.items():
                report += (
                    f"- **{category}**: {metrics.get('accuracy', 0.0):.2%} "
                    f"({metrics.get('correct', 0)}/{metrics.get('total', 0)})\n"
                )
            report += "\n"

        details = results.get("detailed_results") or []
        if details:
            report += "## 样本详情\n\n"
            report += "| 样本ID | 问题 | 预测结果 | 正确答案 | 是否正确 |\n"
            report += "|--------|------|----------|----------|----------|\n"
            for detail in details[:10]:
                q = str(detail.get("question", "N/A"))[:60]
                pred = str(detail.get("predicted", "N/A"))[:40]
                exp = str(detail.get("expected", "N/A"))[:40]
                ok = "[ok]" if detail.get("success") else "[x]"
                report += (
                    f"| {detail.get('sample_id', 'N/A')} | {q} | "
                    f"{pred} | {exp} | {ok} |\n"
                )
            if len(details) > 10:
                report += f"\n*显示前10个样本，共{len(details)}个样本*\n"
            report += "\n"

        bar_len = int(accuracy * 50)
        report += (
            "## 准确率可视化\n\n```\n"
            f"准确率: {'█' * bar_len}{'░' * (50 - bar_len)} {accuracy:.2%}\n"
            "```\n\n## 建议\n\n"
        )
        if accuracy >= 0.9:
            report += "- [ok] 表现优秀！\n"
        elif accuracy >= 0.7:
            report += "- [!] 表现良好，可检查错误样本优化提示词。\n"
        else:
            report += "- [x] 需改进工具调用逻辑 / 系统提示词。\n"

        if output_file is None:
            out_dir = self.project_root / "evaluation_reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"bfcl_report_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.md"
        else:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(report, encoding="utf-8")
        print(f"\n 报告已生成: {out_path}")
        return report
