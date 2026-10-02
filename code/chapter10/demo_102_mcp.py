"""10.2 MCP — 内置计算器服务器自检（无需 LLM / npx）。"""

from __future__ import annotations

from hello_agents import MCPTool
from hello_agents.tools.registry import ToolRegistry


def main() -> None:
    tool = MCPTool(auto_expand=False)
    listed = tool.run({"action": "list_tools"})
    print(listed)
    assert "add" in listed

    add_result = tool.run(
        {
            "action": "call_tool",
            "tool_name": "add",
            "arguments": {"a": 10, "b": 20},
        }
    )
    print(add_result)
    assert "30" in add_result

    # auto_expand：展开为独立 Tool 并注册（与 SimpleAgent.add_tool 同路径）
    expand = MCPTool(auto_expand=True)
    registry = ToolRegistry()
    for t in expand.get_expanded_tools():
        registry.register_tool(t)
    names = registry.list_tools()
    print("expanded:", names)
    assert "add" in names
    assert "multiply" in names

    wrapped = registry.get_tool("add")
    assert wrapped is not None
    direct = wrapped.run({"a": 3, "b": 4})
    print(direct)
    assert "7" in direct
    print("OK")


if __name__ == "__main__":
    main()
