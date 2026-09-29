"""9.5 TerminalTool 自检：白名单、沙箱、cd、读文件。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from hello_agents.tools.builtin.terminal_tool import TerminalTool


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="terminal_tool_"))
    try:
        (root / "README.md").write_text("# demo\nhello terminal\n", encoding="utf-8")
        src = root / "src"
        src.mkdir()
        (src / "main.py").write_text("print('hi')\n# TODO: polish\n", encoding="utf-8")

        term = TerminalTool(workspace=str(root), timeout=10)

        # 拒绝危险命令
        denied = term.run({"command": "rm -rf /"})
        assert "不允许的命令" in denied

        # 拒绝命令链
        chained = term.run({"command": "echo hi && echo bye"})
        assert "不允许" in chained

        # echo / 列目录（Windows 用 dir）
        echo = term.run({"command": "echo hello-agents"})
        assert "hello-agents" in echo or "成功" in echo

        listing = term.run({"command": "dir"})
        assert "README.md" in listing or "readme.md" in listing.lower()

        # cd 沙箱
        cd_ok = term.run({"command": "cd src"})
        assert "切换到目录" in cd_ok
        assert term.get_current_dir().endswith("src")

        escaped = term.run({"command": "cd ..\\..\\.."})
        # 从 src 上三级通常会出 workspace；至少不能成功逃出
        if "切换到目录" in escaped:
            assert str(term.workspace) in term.get_current_dir() or term.get_current_dir().startswith(
                str(term.workspace)
            )
        term.reset_dir()

        # 工作区外绝对路径
        outside = term.run({"command": "type C:\\Windows\\System32\\drivers\\etc\\hosts"})
        # 若系统无 type/路径检测，至少不应原样泄露；期望拒绝
        assert "不允许" in outside or "返回码" in outside or "失败" in outside or "stderr" in outside

        # 读工作区内文件
        readme = term.run({"command": "type README.md"})
        if "不允许" in readme or "返回码" in readme:
            readme = term.run({"command": "cat README.md"})
        assert "hello terminal" in readme or "demo" in readme

        print(term.run({"command": "dir"}))
        print("\n[OK] 9.5 TerminalTool self-check passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
