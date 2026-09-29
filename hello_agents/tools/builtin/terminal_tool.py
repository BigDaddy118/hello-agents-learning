"""TerminalTool — 即时文件系统访问（9.5）：白名单命令 + 工作目录沙箱。"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ..base import Tool, ToolParameter

# 章节只读白名单；type 为 Windows 读文件等价物
_ALLOWED = frozenset(
    {
        "ls",
        "dir",
        "tree",
        "cat",
        "type",
        "head",
        "tail",
        "less",
        "more",
        "find",
        "grep",
        "egrep",
        "fgrep",
        "findstr",
        "wc",
        "sort",
        "uniq",
        "cut",
        "awk",
        "sed",
        "pwd",
        "cd",
        "file",
        "stat",
        "du",
        "df",
        "echo",
        "which",
        "whereis",
        "where",
    }
)


class TerminalTool(Tool):
    """安全命令行：白名单、沙箱 cwd、超时与输出上限。"""

    ALLOWED_COMMANDS = _ALLOWED

    def __init__(
        self,
        workspace: str = ".",
        timeout: int = 30,
        max_output_size: int = 10 * 1024 * 1024,
        allow_cd: bool = True,
    ):
        super().__init__(
            name="terminal",
            description=(
                "命令行工具：在沙箱内执行只读文件系统/文本命令"
                "（ls/dir、cat/type、grep、head、tail、find 等）"
            ),
        )
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allow_cd = allow_cd
        self.current_dir = self.workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, parameters: dict[str, Any]) -> str:
        command = str(parameters.get("command") or parameters.get("input") or "").strip()
        if not command:
            return "❌ 命令不能为空"

        err = self._validate_pipeline(command)
        if err:
            return err

        try:
            parts = self._split(command)
        except ValueError as e:
            return f"❌ 命令解析失败: {e}"
        if not parts:
            return "❌ 命令不能为空"

        if parts[0] == "cd":
            return self._handle_cd(parts)

        path_err = self._check_path_args(parts)
        if path_err:
            return path_err

        return self._execute_command(command)

    def get_parameters(self) -> list[ToolParameter]:
        sample = ", ".join(sorted(self.ALLOWED_COMMANDS)[:8])
        return [
            ToolParameter(
                name="command",
                type="string",
                description=f"白名单命令，如 ls/dir、cat/type、grep。示例白名单: {sample}...",
                required=True,
            ),
        ]

    def get_current_dir(self) -> str:
        return str(self.current_dir)

    def reset_dir(self) -> None:
        self.current_dir = self.workspace

    @staticmethod
    def _split(command: str) -> list[str]:
        # Windows 下 posix=True 会吞掉反斜杠，路径校验会失效
        return shlex.split(command, posix=(os.name != "nt"))

    def _validate_pipeline(self, command: str) -> str | None:
        # 禁止命令链 / 替换；允许 | 管道，但每段都要在白名单
        for bad in (";", "&&", "||", "`", "\n", "$(", "${"):
            if bad in command:
                return f"❌ 不允许的 shell 语法: {bad!r}"
        for segment in command.split("|"):
            seg = segment.strip()
            if not seg:
                return "❌ 空的管道段"
            try:
                parts = self._split(seg)
            except ValueError as e:
                return f"❌ 命令解析失败: {e}"
            if not parts:
                return "❌ 空的管道段"
            if parts[0] not in self.ALLOWED_COMMANDS:
                allowed = ", ".join(sorted(self.ALLOWED_COMMANDS))
                return f"❌ 不允许的命令: {parts[0]}\n允许的命令: {allowed}"
        return None

    def _check_path_args(self, parts: list[str]) -> str | None:
        for token in parts[1:]:
            if token.startswith("-"):
                continue
            # Windows 开关: /s /b /n 等，勿当路径
            if token.startswith("/") and len(token) <= 4 and token[1:].replace(":", "").isalnum():
                continue
            # 跳过明显非路径参数
            if token in {".", "..", "~", "*.py", "*.*", "*.md", "*.txt", "*.csv", "*.log", "*.json"}:
                continue
            if token.startswith("*"):
                continue
            looks_path = (
                ("/" in token and not token.startswith("/"))
                or "\\" in token
                or token.endswith((".py", ".md", ".txt", ".csv", ".log", ".json"))
                or Path(token).is_absolute()
            )
            if not looks_path:
                continue
            err = self._ensure_in_workspace(token)
            if err:
                return err
        return None

    def _ensure_in_workspace(self, path_str: str) -> str | None:
        raw = Path(path_str)
        target = raw.resolve() if raw.is_absolute() else (self.current_dir / raw).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            return f"❌ 不允许访问工作目录外的路径: {target}"
        return None

    def _handle_cd(self, parts: list[str]) -> str:
        if not self.allow_cd:
            return "❌ cd 命令已禁用"
        if len(parts) < 2:
            return f"当前目录: {self.current_dir}"

        target = parts[1]
        if target == "..":
            new_dir = self.current_dir.parent
        elif target == ".":
            new_dir = self.current_dir
        elif target == "~":
            new_dir = self.workspace
        else:
            new_dir = (self.current_dir / target).resolve()

        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return f"❌ 不允许访问工作目录外的路径: {new_dir}"
        if not new_dir.exists():
            return f"❌ 目录不存在: {new_dir}"
        if not new_dir.is_dir():
            return f"❌ 不是目录: {new_dir}"

        self.current_dir = new_dir
        return f"✅ 切换到目录: {self.current_dir}"

    def _execute_command(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,  # noqa: S602 — 章节 JIT shell；已做白名单/管道校验
                cwd=str(self.current_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return f"❌ 命令执行超时（超过 {self.timeout} 秒）"
        except OSError as e:
            return f"❌ 命令执行失败: {e}"

        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if len(output) > self.max_output_size:
            output = (
                output[: self.max_output_size]
                + f"\n\n⚠️ 输出被截断（超过 {self.max_output_size} 字节）"
            )
        if result.returncode != 0:
            output = f"⚠️ 命令返回码: {result.returncode}\n\n{output}"
        return output if output.strip() else "✅ 命令执行成功（无输出）"
