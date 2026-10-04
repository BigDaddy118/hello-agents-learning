"""BFCL AST 匹配自检（无需 LLM / 数据集）。"""

from hello_agents.evaluation.benchmarks.bfcl.evaluator import BFCLEvaluator


def main() -> None:
    ev = BFCLEvaluator.__new__(BFCLEvaluator)

    # 参数顺序不同应匹配
    pred = [{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}]
    exp = [{"get_weather": {"city": ["Beijing"], "unit": ["celsius"]}}]
    ok, score = ev._evaluate_bfcl_v4_format(pred, exp)
    assert ok and score == 1.0, (ok, score)

    # 函数名错误应失败
    bad = [{"name": "get_temperature", "arguments": {"city": "Beijing"}}]
    ok2, score2 = ev._evaluate_bfcl_v4_format(bad, exp)
    assert not ok2 and score2 == 0.0, (ok2, score2)

    # 可接受多值
    exp_multi = [{"get_weather": {"city": ["Beijing", "北京"], "unit": ["celsius"]}}]
    ok3, _ = ev._evaluate_bfcl_v4_format(pred, exp_multi)
    assert ok3

    q = ev._normalize_question(
        [[{"role": "user", "content": "Weather in Beijing?"}]]
    )
    assert q == "Weather in Beijing?"

    print("test_bfcl_ast: ok")


if __name__ == "__main__":
    main()
