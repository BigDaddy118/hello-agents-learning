"""异常体系。"""


class HelloAgentsException(Exception):
    """HelloAgents 基础异常。"""


class LLMException(HelloAgentsException):
    """LLM 相关异常。"""


class AgentException(HelloAgentsException):
    """Agent 相关异常。"""


class ConfigException(HelloAgentsException):
    """配置相关异常。"""


class ToolException(HelloAgentsException):
    """工具相关异常。"""
