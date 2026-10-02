"""11.4.2 GRPO 训练实战

在 SFT 模型上做 GRPO。默认用 ./models/sft_model（11.3.3 产物）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hello_agents.tools import RLTrainingTool


def basic_grpo(sft_dir: str = "./models/sft_model") -> dict:
    """教程基础 GRPO（小参数快跑）。"""
    if not Path(sft_dir).exists():
        raise SystemExit(f"找不到 SFT 模型: {sft_dir}，请先跑 11.3.3")

    rl_tool = RLTrainingTool()
    result_str = rl_tool.run(
        {
            "action": "train",
            "algorithm": "grpo",
            "model_name": sft_dir,
            "output_dir": "./models/grpo_model",
            "max_samples": 8,
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-5,
            "num_generations": 2,
            "max_new_tokens": 64,
            "temperature": 0.8,
            "kl_coef": 0.05,
            "clip_range": 0.2,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16,
            "reward_type": "accuracy",
            "use_tensorboard": False,
        }
    )
    result = json.loads(result_str)
    if result.get("status") == "error":
        print(result_str)
        raise SystemExit(1)

    print("\n✓ Training completed!")
    print(f"  - Model save path: {result['output_dir']}")
    print(f"  - Training samples: {result['dataset_size']}")
    print(f"  - Training epochs: {result['num_epochs']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="11.4.2 GRPO 训练实战")
    parser.add_argument(
        "--sft-dir",
        default="./models/sft_model",
        help="SFT 起点（默认 ./models/sft_model）",
    )
    args = parser.parse_args()
    basic_grpo(args.sft_dir)


if __name__ == "__main__":
    main()
