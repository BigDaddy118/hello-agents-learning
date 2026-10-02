"""11.1.5 快速上手示例：SFT → GRPO → 评估。"""

import json
from pathlib import Path

from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()
sft_dir = "./models/quick_test_sft"
grpo_dir = "./models/quick_test_grpo"

# 1. SFT（已有 checkpoint 则跳过）
if Path(sft_dir).exists() and any(Path(sft_dir).iterdir()):
    sft_result = {"output_dir": sft_dir, "status": "skipped"}
    print(f"\n✓ 复用已有 SFT: {sft_dir}")
else:
    sft_result_str = rl_tool.run({
        "action": "train",
        "algorithm": "sft",
        "model_name": "Qwen/Qwen3-0.6B",
        "output_dir": sft_dir,
        "max_samples": 10,
        "num_epochs": 1,
        "batch_size": 2,
        "use_lora": True,
        "use_tensorboard": False,
    })
    sft_result = json.loads(sft_result_str)
    if sft_result.get("status") == "error":
        print(sft_result_str)
        raise SystemExit(1)
    print(f"\n✓ SFT训练完成,模型保存在: {sft_result['output_dir']}")

# 2. GRPO
grpo_result_str = rl_tool.run({
    "action": "train",
    "algorithm": "grpo",
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": grpo_dir,
    "max_samples": 2,
    "num_epochs": 1,
    "batch_size": 2,
    "use_lora": True,
    "use_tensorboard": False,
})
grpo_result = json.loads(grpo_result_str)
if grpo_result.get("status") == "error":
    print(grpo_result_str)
    raise SystemExit(1)
print(f"\n✓ GRPO训练完成,模型保存在: {grpo_result['output_dir']}")

# 3. 评估
eval_result_str = rl_tool.run({
    "action": "evaluate",
    "model_path": grpo_dir,
    "max_samples": 2,
    "use_lora": True,
})
eval_result = json.loads(eval_result_str)
if eval_result.get("status") == "error":
    print(eval_result_str)
    raise SystemExit(1)
print(f"\n✓ 评估完成:")
print(f"  - 准确率: {eval_result['accuracy']}")
print(f"  - 平均奖励: {eval_result['average_reward']}")
print(f"  - 测试样本数: {eval_result['num_samples']}")

print("\n" + "=" * 50)
print("完成第一个 Agentic RL 快速训练")
print("=" * 50)
print(f"  SFT:  {sft_result['output_dir']}")
print(f"  GRPO: {grpo_result['output_dir']}")
