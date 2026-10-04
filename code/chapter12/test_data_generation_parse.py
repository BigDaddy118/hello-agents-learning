"""12.4 自检：LaTeX JSON 解析 + LLM Judge metrics（无需 LLM/HF）。"""

from __future__ import annotations

import json
from pathlib import Path

from hello_agents.evaluation.benchmarks.data_generation.dataset import AIDataset
from hello_agents.evaluation.benchmarks.data_generation.llm_judge import (
    LLMJudgeEvaluator,
)
from hello_agents.evaluation.benchmarks.data_generation.win_rate import (
    WinRateEvaluator,
)


def main() -> None:
    # LaTeX 转义修复（与 AIMEGenerator / WinRate 同源逻辑）
    bad = r'{"problem": "Find $\frac{a}{b}$", "answer": 1}'
    import re

    fixed = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", bad)
    assert json.loads(fixed)["answer"] == 1

    sample = (
        Path(__file__).resolve().parent
        / "data_generation"
        / "generated_data"
        / "aime_generated_sample.json"
    )
    problems = AIDataset(dataset_type="generated", data_path=str(sample)).load()
    assert len(problems) >= 2
    assert problems[0]["problem"]

    judge = LLMJudgeEvaluator.__new__(LLMJudgeEvaluator)
    judge.EVALUATION_DIMENSIONS = LLMJudgeEvaluator.EVALUATION_DIMENSIONS
    scores = judge._parse_evaluation_response(
        '```json\n{"correctness":5,"clarity":4,"difficulty_match":4,"completeness":5,"comments":"ok"}\n```'
    )
    assert scores["correctness"] == 5.0

    fake = [
        {
            "problem_id": "a",
            "scores": {
                "correctness": 5,
                "clarity": 4,
                "difficulty_match": 4,
                "completeness": 5,
            },
            "total_score": 4.5,
        }
    ]
    metrics = judge._compute_metrics(fake)
    assert metrics["average_total_score"] == 4.5
    assert metrics["pass_rate"] == 1.0

    wr = WinRateEvaluator.__new__(WinRateEvaluator)
    winner, reason = wr._parse_comparison_response(
        '```json\n{"winner":"Problem A","reason":"clearer"}\n```',
        "Problem A",
        "Problem B",
    )
    assert winner == "Problem A"
    assert "clearer" in reason

    print("test_data_generation_parse: ok")


if __name__ == "__main__":
    main()
