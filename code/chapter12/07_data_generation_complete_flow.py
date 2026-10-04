"""12.4：完整流程入口（生成 → LLM Judge → Win Rate）。

用法（在 code/chapter12 下）:
  python 07_data_generation_complete_flow.py 3 2.0

参数: 生成数量, 每题延迟秒数
只评估已有样本见 data_generation/step2_evaluate_only.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

# 脚本内用相对 import `from aime_generator import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent / "data_generation"))

from run_complete_evaluation import run_complete_evaluation


def main() -> None:
    num_problems = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    delay_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    # 工作目录切到 chapter12，保证 data_generation/... 相对路径正确
    import os

    os.chdir(Path(__file__).resolve().parent)
    run_complete_evaluation(num_problems=num_problems, delay_seconds=delay_seconds)


if __name__ == "__main__":
    main()
