"""7.4 / 7.5 最小自检（计算器本地；Agent 需 .env）。"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from hello_agents import (  # noqa: E402
    CalculatorTool,
    HelloAgentsLLM,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
    ToolChain,
    ToolRegistry,
    calculate,
    run_parallel_tools_sync,
)


def test_tools() -> None:
    assert calculate("2 + 3 * 4") == "14"
    assert calculate("sqrt(16)") == "4.0"

    reg = ToolRegistry()
    reg.register_tool(CalculatorTool())
    assert reg.execute_tool("calculator", "10 / 2") == "5.0"

    chain = ToolChain("calc2", "两步计算")
    chain.add_step("calculator", "{input}", "a")
    chain.add_step("calculator", "1 + 1", "b")
    assert chain.execute(reg, "3*3") == "2"

    results = run_parallel_tools_sync(
        reg,
        [
            {"tool_name": "calculator", "input_data": "1+1"},
            {"tool_name": "calculator", "input_data": "2*3"},
        ],
    )
    assert [r["result"] for r in results] == ["2", "6"]
    print("✅ tools ok")


def test_agents() -> None:
    llm = HelloAgentsLLM()
    reg = ToolRegistry()
    reg.register_tool(CalculatorTool())

    basic = SimpleAgent(name="basic", llm=llm, system_prompt="简短回答。")
    r1 = basic.run("用两个字回答：天空通常是什么颜色？")
    assert r1.strip()
    print(f"Simple: {r1[:60]}")

    with_tools = SimpleAgent(
        name="tool",
        llm=llm,
        system_prompt="需要计算时用工具。",
        tool_registry=reg,
    )
    r2 = with_tools.run("计算 15*8+32，只给数字")
    assert r2.strip()
    print(f"Simple+tool: {r2[:80]}")

    react = ReActAgent(name="react", llm=llm, tool_registry=reg, max_steps=6)
    r3 = react.run("用计算器算 6*7，给出数字答案")
    assert r3.strip() and "无法在限定步数" not in r3
    print(f"ReAct: {r3[:80]}")

    reflect = ReflectionAgent(name="reflect", llm=llm, max_iterations=1)
    r4 = reflect.run("用一句话解释什么是 Agent")
    assert r4.strip()
    print(f"Reflection: {r4[:80]}")

    plan = PlanAndSolveAgent(name="plan", llm=llm)
    r5 = plan.run(
        "周一卖15个苹果，周二是周一两倍，周三比周二少5个，三天共卖多少？"
    )
    assert r5.strip()
    print(f"PlanSolve: {r5[:80]}")
    print("✅ agents ok")


if __name__ == "__main__":
    test_tools()
    test_agents()
