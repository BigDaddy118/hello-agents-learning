"""11.6 完整训练流程实战

数据准备 → SFT → SFT 评估 → GRPO → GRPO 评估 → 保存结果。
默认用 config.json 的小参数快跑；要更大配置自己改 config。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hello_agents.tools import RLTrainingTool


class AgenticRLPipeline:
    """Agentic RL 端到端流水线。"""

    def __init__(self, config_path: str | Path):
        self.rl_tool = RLTrainingTool()
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.results: dict = {}
        self.results_path = self.config_path.parent / "training_results.json"

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {message}")

    def stage1_prepare_data(self) -> dict:
        self.log("=" * 50)
        self.log("Stage 1: Data Preparation")
        self.log("=" * 50)
        raw = self.rl_tool.run(
            {
                "action": "load_dataset",
                "format": "sft",
                "max_samples": self.config["data"]["max_samples"],
            }
        )
        info = json.loads(raw)
        if info.get("status") == "error":
            raise RuntimeError(info)
        self.log(f"✓ Dataset loaded: {info['dataset_size']} samples, cols={info['sample_keys']}")
        self.results["data"] = info
        return info

    def stage2_sft_training(self) -> str:
        self.log("\n" + "=" * 50)
        self.log("Stage 2: SFT Training")
        self.log("=" * 50)
        sft = self.config["sft"]
        mon = self.config.get("monitoring", {})
        raw = self.rl_tool.run(
            {
                "action": "train",
                "algorithm": "sft",
                "model_name": self.config["model"]["base_model"],
                "output_dir": sft["output_dir"],
                "max_samples": self.config["data"]["max_samples"],
                "num_epochs": sft["num_epochs"],
                "batch_size": sft["batch_size"],
                "learning_rate": sft.get("learning_rate", 5e-5),
                "use_lora": True,
                "lora_rank": sft.get("lora_rank", 8),
                "lora_alpha": sft.get("lora_alpha", 16),
                "use_wandb": mon.get("use_wandb", False),
                "use_tensorboard": mon.get("use_tensorboard", False),
                "wandb_project": mon.get("wandb_project"),
            }
        )
        data = json.loads(raw)
        if data.get("status") == "error":
            raise RuntimeError(data)
        self.log(f"✓ SFT done → {data['output_dir']}")
        self.results["sft_training"] = data
        return data["output_dir"]

    def stage3_sft_evaluation(self, model_path: str) -> dict:
        self.log("\n" + "=" * 50)
        self.log("Stage 3: SFT Evaluation")
        self.log("=" * 50)
        ev = self.config["eval"]
        raw = self.rl_tool.run(
            {
                "action": "evaluate",
                "model_path": model_path,
                "max_samples": ev["max_samples"],
                "max_new_tokens": ev.get("max_new_tokens", 64),
                "use_lora": True,
                "metrics": ["accuracy", "average_length", "format_correctness"],
            }
        )
        data = json.loads(raw)
        if data.get("status") == "error":
            raise RuntimeError(data)
        self.log(f"✓ SFT eval accuracy={data['accuracy']:.2%}")
        self.results["sft_evaluation"] = data
        thr = float(ev.get("sft_accuracy_threshold", 0.0))
        if data["accuracy"] < thr:
            self.log(f"⚠ SFT accuracy {data['accuracy']:.2%} < threshold {thr:.2%} (继续 GRPO)")
        return data

    def stage4_grpo_training(self, sft_model_path: str) -> str:
        self.log("\n" + "=" * 50)
        self.log("Stage 4: GRPO Training")
        self.log("=" * 50)
        grpo = self.config["grpo"]
        mon = self.config.get("monitoring", {})
        raw = self.rl_tool.run(
            {
                "action": "train",
                "algorithm": "grpo",
                "model_name": sft_model_path,
                "output_dir": grpo["output_dir"],
                "max_samples": self.config["data"]["max_samples"],
                "num_epochs": grpo["num_epochs"],
                "batch_size": grpo["batch_size"],
                "learning_rate": grpo.get("learning_rate", 1e-5),
                "num_generations": grpo.get("num_generations", 2),
                "max_new_tokens": grpo.get("max_new_tokens", 64),
                "kl_coef": grpo.get("kl_coef", 0.05),
                "use_lora": True,
                "lora_rank": grpo.get("lora_rank", 8),
                "lora_alpha": grpo.get("lora_alpha", 16),
                "reward_type": "accuracy",
                "use_wandb": mon.get("use_wandb", False),
                "use_tensorboard": mon.get("use_tensorboard", False),
                "wandb_project": mon.get("wandb_project"),
            }
        )
        data = json.loads(raw)
        if data.get("status") == "error":
            raise RuntimeError(data)
        self.log(f"✓ GRPO done → {data['output_dir']}")
        self.results["grpo_training"] = data
        return data["output_dir"]

    def stage5_grpo_evaluation(self, model_path: str) -> dict:
        self.log("\n" + "=" * 50)
        self.log("Stage 5: GRPO Evaluation")
        self.log("=" * 50)
        ev = self.config["eval"]
        raw = self.rl_tool.run(
            {
                "action": "evaluate",
                "model_path": model_path,
                "max_samples": ev["max_samples"],
                "max_new_tokens": ev.get("max_new_tokens", 64),
                "use_lora": True,
                "metrics": ["accuracy", "average_length", "format_correctness"],
                "return_details": True,
            }
        )
        data = json.loads(raw)
        if data.get("status") == "error":
            raise RuntimeError(data)
        self.log(f"✓ GRPO eval accuracy={data['accuracy']:.2%}")
        self.results["grpo_evaluation"] = data
        return data

    def stage6_save_results(self) -> None:
        self.log("\n" + "=" * 50)
        self.log("Stage 6: Save Results")
        self.log("=" * 50)
        self.results_path.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.log(f"✓ Results → {self.results_path}")

    def run(self, reuse_sft: bool = False) -> None:
        self.stage1_prepare_data()

        sft_dir = self.config["sft"]["output_dir"]
        if reuse_sft and Path(sft_dir).exists() and any(Path(sft_dir).iterdir()):
            self.log(f"✓ Reuse SFT: {sft_dir}")
            self.results["sft_training"] = {"output_dir": sft_dir, "status": "skipped"}
            sft_path = sft_dir
        else:
            sft_path = self.stage2_sft_training()

        self.stage3_sft_evaluation(sft_path)
        grpo_path = self.stage4_grpo_training(sft_path)
        self.stage5_grpo_evaluation(grpo_path)
        self.stage6_save_results()

        self.log("\n" + "=" * 50)
        self.log("✓ Training pipeline completed!")
        self.log(f"  SFT:  {sft_path}")
        self.log(f"  GRPO: {grpo_path}")
        self.log("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="11.6 完整训练流程")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="配置文件路径",
    )
    parser.add_argument(
        "--reuse-sft",
        action="store_true",
        help="已有 SFT 目录则跳过 SFT 训练",
    )
    args = parser.parse_args()
    AgenticRLPipeline(args.config).run(reuse_sft=args.reuse_sft)


if __name__ == "__main__":
    main()
