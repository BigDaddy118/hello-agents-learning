"""11.5 模型评估与分析

小样本快跑：对比基座 / SFT / GRPO，并做错误类型粗分。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hello_agents.tools import RLTrainingTool


def evaluate_one(
    tool: RLTrainingTool,
    name: str,
    path: str,
    max_samples: int,
    return_details: bool = False,
) -> dict:
    print("\n" + "=" * 60)
    print(f"Evaluating {name}: {path}")
    print("=" * 60)
    raw = tool.run(
        {
            "action": "evaluate",
            "model_path": path,
            "max_samples": max_samples,
            "max_new_tokens": 64,
            "metrics": [
                "accuracy",
                "average_length",
                "average_steps",
                "format_correctness",
            ],
            "return_details": return_details,
            "use_lora": True,
        }
    )
    data = json.loads(raw)
    if data.get("status") == "error":
        print(raw)
        raise SystemExit(1)
    data["name"] = name
    return data


def print_comparison(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{'Model':<16} {'Accuracy':<12} {'AvgLen':<10} {'AvgSteps':<10} {'Format':<10}")
    print("=" * 70)
    for r in rows:
        print(
            f"{r['name']:<16} "
            f"{r['accuracy']:<12.2%} "
            f"{r.get('average_length', 0):<10.1f} "
            f"{r.get('average_steps', 0):<10.2f} "
            f"{r.get('format_correctness', 0):<10.2%}"
        )
    print("=" * 70)


def print_error_analysis(data: dict) -> None:
    errors = data.get("errors") or []
    print(f"\nTotal errors: {len(errors)}")
    types = data.get("error_types") or {}
    if errors:
        print("Error type distribution:")
        for k, v in types.items():
            print(f"  {k}: {v} ({v / len(errors) * 100:.1f}%)")

    details = data.get("details") or []
    groups = {
        "Easy (1-2 steps)": [],
        "Medium (3-4 steps)": [],
        "Hard (5+ steps)": [],
    }
    for s in details:
        steps = int(s.get("ground_truth_steps") or 0)
        if steps <= 2:
            groups["Easy (1-2 steps)"].append(s["correct"])
        elif steps <= 4:
            groups["Medium (3-4 steps)"].append(s["correct"])
        else:
            groups["Hard (5+ steps)"].append(s["correct"])
    print("\nAccuracy at different difficulty levels:")
    for name, vals in groups.items():
        if vals:
            print(f"  {name}: {sum(vals) / len(vals):.2%} ({len(vals)} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="11.5 模型评估")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--skip-base", action="store_true", help="跳过基座（更快）")
    args = parser.parse_args()

    tool = RLTrainingTool()
    models = []
    if not args.skip_base:
        models.append(("Pretrained", "Qwen/Qwen3-0.6B"))
    models.append(("SFT", "./models/sft_model"))
    models.append(("GRPO", "./models/grpo_model"))

    for _, path in models:
        if path.startswith("./") and not Path(path).exists():
            raise SystemExit(f"缺少模型: {path}")

    rows = []
    for i, (name, path) in enumerate(models):
        # 只对最后一个模型做错误明细，省时间
        detail = i == len(models) - 1
        rows.append(evaluate_one(tool, name, path, args.max_samples, detail))

    print_comparison(rows)
    print_error_analysis(rows[-1])
    print("\n11.5.4 改进方向（按结果自取）: 算力弱→多训 SFT；错算多→加强数值；难题差→加长推理数据。")


if __name__ == "__main__":
    main()
