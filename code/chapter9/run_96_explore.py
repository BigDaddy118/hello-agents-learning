"""用真实 LLM 跑 9.6 CodebaseMaintainer.explore()。需配置 .env。"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from hello_agents import CodebaseMaintainer, HelloAgentsLLM


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    codebase = root / "my_app"
    notes = root / "my_app_notes"

    m = CodebaseMaintainer(
        project_name="my_app",
        codebase_path=str(codebase),
        llm=HelloAgentsLLM(),
        notes_workspace=str(notes),
    )
    print(m.explore())


if __name__ == "__main__":
    main()
