"""11.4.3 GRPO 训练过程分析

1) 相对奖励手工推演
2) 离线监控：看日志里 Reward / KL / Loss
3) 可选：短训一截，观察指标（--train）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def demo_relative_rewards() -> None:
    """教程里的相对奖励例子。"""
    question = "What is 48 + 24?"
    answers = [
        "48 + 24 = 72. Final Answer: 72",
        "48 + 24 = 72. Final Answer: 72",
        "48 + 24 = 70. Final Answer: 70",
        "Let me think... 72. Final Answer: 72",
    ]
    rewards = [1.0, 1.0, 0.0, 0.8]
    avg = sum(rewards) / len(rewards)
    relative = [r - avg for r in rewards]

    print("=" * 60)
    print("11.4.3 (1) 相对奖励推演")
    print("=" * 60)
    print(f"question: {question}")
    print(f"rewards:          {rewards}")
    print(f"group avg:        {avg:.2f}")
    print(f"relative rewards: {[round(x, 2) for x in relative]}")
    print("→ 高于组均值的答案被鼓励，错误答案相对惩罚最大")
    print()
    print("KL 系数 kl_coef(β) 建议 0.05–0.1：")
    print("  太小 → 偏离 SFT；太大 → 几乎学不动")


def demo_reward_components() -> None:
    from hello_agents.rl import (
        create_accuracy_reward,
        create_combined_reward,
        create_length_penalty_reward,
        create_step_reward,
    )

    print("=" * 60)
    print("11.4.3 奖励组件对比")
    print("=" * 60)
    base = create_accuracy_reward()
    length_fn = create_length_penalty_reward(base, max_length=200, penalty_weight=0.001)
    step_fn = create_step_reward(base, step_bonus=0.1)
    combined = create_combined_reward(
        [
            {"type": "accuracy", "weight": 1.0},
            {"type": "length_penalty", "weight": 0.5, "target_length": 200},
            {"type": "step", "weight": 0.3, "step_bonus": 0.1},
        ]
    )
    samples = [
        ("Final Answer: 72", "72", "短正确"),
        ("Step1\nStep2\nFinal Answer: 72", "72", "有步骤"),
        (("blah " * 80) + "Final Answer: 72", "72", "冗长正确"),
        ("Final Answer: 70", "72", "错误"),
    ]
    for text, gt, desc in samples:
        kwargs = {"ground_truth": [gt]}
        print(f"\n[{desc}] len={len(text)}")
        print(f"  accuracy: {base([text], **kwargs)[0]:.3f}")
        print(f"  length:   {length_fn([text], **kwargs)[0]:.3f}")
        print(f"  step:     {step_fn([text], **kwargs)[0]:.3f}")
        print(f"  combined: {combined([text], **kwargs)[0]:.3f}")


def short_monitored_train(sft_dir: str) -> dict:
    """短训，靠控制台日志观察 Reward / KL（离线监控）。"""
    from hello_agents.tools import RLTrainingTool

    if not Path(sft_dir).exists():
        raise SystemExit(f"找不到 SFT 模型: {sft_dir}")

    print("=" * 60)
    print("11.4.3 (3) 离线监控短训")
    print("关注日志: Reward / KL / Loss；TensorBoard: --logdir=./models/grpo_tb")
    print("=" * 60)

    tool = RLTrainingTool()
    result_str = tool.run(
        {
            "action": "train",
            "algorithm": "grpo",
            "model_name": sft_dir,
            "output_dir": "./models/grpo_tb",
            "max_samples": 8,
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-5,
            "num_generations": 2,
            "max_new_tokens": 64,
            "kl_coef": 0.05,
            "use_lora": True,
            "reward_type": "accuracy",
            "use_tensorboard": True,
            "use_wandb": False,
        }
    )
    result = json.loads(result_str)
    if result.get("status") == "error":
        print(result_str)
        raise SystemExit(1)
    print(f"\n✓ 短训完成 → {result['output_dir']}")
    print("查看曲线: tensorboard --logdir=./models/grpo_tb")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="11.4.3 GRPO 过程分析")
    parser.add_argument(
        "--train",
        action="store_true",
        help="额外跑一小段带 TensorBoard 的 GRPO",
    )
    parser.add_argument("--sft-dir", default="./models/sft_model")
    args = parser.parse_args()

    demo_relative_rewards()
    demo_reward_components()
    if args.train:
        short_monitored_train(args.sft_dir)


if __name__ == "__main__":
    main()
