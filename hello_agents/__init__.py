"""HelloAgents 框架（第七–十章）。"""

from .agents import (
    CodebaseMaintainer,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
)
from .context import ContextBuilder, ContextConfig, ContextPacket
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
from .tools.builtin.note_tool import NoteTool as NoteTool
from .tools.builtin.protocol_tools import MCPTool as MCPTool
from .tools.builtin.terminal_tool import TerminalTool as TerminalTool

__version__ = "0.2.0"
__all__ = [
    "Agent",
    "AsyncToolExecutor",
    "CalculatorTool",
    "CodebaseMaintainer",
    "Config",
    "ContextBuilder",
    "ContextConfig",
    "ContextPacket",
    "HelloAgentsException",
    "HelloAgentsLLM",
    "MCPTool",
    "MemoryTool",
    "Message",
    "NoteTool",
    "PlanAndSolveAgent",
    "RAGTool",
    "ReActAgent",
    "ReflectionAgent",
    "SearchTool",
    "SimpleAgent",
    "TerminalTool",
    "Tool",
    "ToolChain",
    "ToolChainManager",
    "ToolRegistry",
    "__version__",
    "calculate",
    "global_registry",
    "run_parallel_tools_sync",
    "search",
]
