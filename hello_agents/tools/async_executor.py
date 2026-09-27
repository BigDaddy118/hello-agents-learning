"""异步工具执行器（7.5.4）。"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from .registry import ToolRegistry


class AsyncToolExecutor:
    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self.registry.execute_tool, tool_name, input_data
        )

    async def execute_tools_parallel(
        self, tasks: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        coros = [
            self.execute_tool_async(t["tool_name"], t.get("input_data", ""))
            for t in tasks
            if t.get("tool_name")
        ]
        raw = await asyncio.gather(*coros, return_exceptions=True)
        results: list[dict[str, Any]] = []
        for i, (task, item) in enumerate(zip(tasks, raw)):
            if isinstance(item, Exception):
                results.append(
                    {
                        "task_id": i,
                        "tool_name": task["tool_name"],
                        "input_data": task.get("input_data", ""),
                        "result": str(item),
                        "status": "error",
                    }
                )
            else:
                results.append(
                    {
                        "task_id": i,
                        "tool_name": task["tool_name"],
                        "input_data": task.get("input_data", ""),
                        "result": item,
                        "status": "success",
                    }
                )
        return results

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def __enter__(self) -> AsyncToolExecutor:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    async def __aenter__(self) -> AsyncToolExecutor:
        return self

    async def __aexit__(self, *_) -> None:
        self.close()


def run_parallel_tools_sync(
    registry: ToolRegistry, tasks: list[dict[str, str]], max_workers: int = 4
) -> list[dict[str, Any]]:
    async def _run() -> list[dict[str, Any]]:
        async with AsyncToolExecutor(registry, max_workers) as ex:
            return await ex.execute_tools_parallel(tasks)

    return asyncio.run(_run())
