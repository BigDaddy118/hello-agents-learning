"""HelloAgents 框架（第七–八章）。"""

from .agents import (
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
)
from .core.agent import Agent
from .core.config import Config
from .core.exceptions import HelloAgentsException
from .core.llm import HelloAgentsLLM
from .core.message import Message
from .tools import (
    AsyncToolExecutor,
    CalculatorTool,
    MemoryTool,
    RAGTool,
    SearchTool,
    Tool,
    ToolChain,
    ToolChainManager,
    ToolRegistry,
    calculate,
    global_registry,
    run_parallel_tools_sync,
    search,
)

__version__ = "0.2.0"
__all__ = [
    "__version__",
    "Agent",
    "Config",
    "HelloAgentsLLM",
    "HelloAgentsException",
    "Message",
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",
    "Tool",
    "ToolRegistry",
    "global_registry",
    "ToolChain",
    "ToolChainManager",
    "AsyncToolExecutor",
    "CalculatorTool",
    "SearchTool",
    "MemoryTool",
    "RAGTool",
    "calculate",
    "search",
    "run_parallel_tools_sync",
]
