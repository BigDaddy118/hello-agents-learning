"""11.3.3 SFT 训练实战

对应教程：准备数据 → 配置 LoRA → 设训练参数 → 开训 → 保存模型。
默认跑教程「基础训练」配置（100 样本 / 3 epoch）；传 --full 跑全量数据示例。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 仓库根目录加入 path，便于未 pip install -e 时也能 import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hello_agents.tools import RLTrainingTool


def basic_sft() -> dict:
    """教程基础训练示例。"""
    rl_tool = RLTrainingTool()
    result_str = rl_tool.run(
        {
            "action": "train",
            "algorithm": "sft",
            "model_name": "Qwen/Qwen3-0.6B",
            "output_dir": "./models/sft_model",
            "max_samples": 100,
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16,
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


def full_sft() -> dict:
    """教程完整训练示例（全量 GSM8K，时间更长）。"""
    rl_tool = RLTrainingTool()
    result_str = rl_tool.run(
        {
            "action": "train",
            "algorithm": "sft",
            "model_name": "Qwen/Qwen3-0.6B",
            "output_dir": "./models/sft_full",
            "max_samples": None,
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "use_lora": True,
            "lora_rank": 16,
            "lora_alpha": 32,
            "use_tensorboard": False,
        }
    )
    result = json.loads(result_str)
    if result.get("status") == "error":
        print(result_str)
        raise SystemExit(1)

    print("\n✓ Full SFT completed!")
    print(f"  - Model save path: {result['output_dir']}")
    print(f"  - Training samples: {result['dataset_size']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="11.3.3 SFT 训练实战")
    parser.add_argument(
        "--full",
        action="store_true",
        help="跑全量数据集版本（更慢）",
    )
    args = parser.parse_args()
    (full_sft if args.full else basic_sft)()


if __name__ == "__main__":
    main()
