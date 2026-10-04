"""GAIA 准精确匹配自检（无需 HF / LLM）。"""

from hello_agents.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator


def main() -> None:
    ev = GAIAEvaluator.__new__(GAIAEvaluator)

    assert ev._normalize_answer("$1,234.56") == "1234.56"
    assert ev._normalize_answer("The United States") == "united states"
    assert ev._normalize_answer("Paris, London, Berlin") == "berlin,london,paris"

    assert ev._extract_answer("thinking...\nFINAL ANSWER: 42\n") == "42"
    assert ev._extract_answer("FINAL ANSWER: [hello]") == "hello"

    assert ev._check_exact_match("42", "42")
    assert ev._check_exact_match("$1,000", "1000")
    assert not ev._check_exact_match("41", "42")
    assert ev._check_partial_match("about 42 apples", "42")

    print("test_gaia_match: ok")


if __name__ == "__main__":
    main()
