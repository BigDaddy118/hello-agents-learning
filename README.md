# Hello-Agents 学习项目

跟着 Datawhale [Hello-Agents](https://hello-agents.datawhale.cc/) 教程做的本地练习。前几章是手写或现成框架实验，第七章起在本仓库里从零搭自己的 `hello_agents` 包。

教程原文：[第七章 构建你的 Agent 框架](https://hello-agents.datawhale.cc/#/./chapter7/%E7%AC%AC%E4%B8%83%E7%AB%A0%20%E6%9E%84%E5%BB%BA%E4%BD%A0%E7%9A%84Agent%E6%A1%86%E6%9E%B6)

## 目录

| 路径 | 内容 |
|------|------|
| `hello_agents/` | 自建框架（第七章 7.1–7.5） |
| `code/chapter4/` | 手写 ReAct、Plan-and-Solve、Memory、Tools、LLM 客户端 |
| `code/chapter6/` | AutoGen、AgentScope、CAMEL、LangGraph 练习 |
| `code/chapter7/` | 框架自检脚本 `demo_74_75.py` |

### `hello_agents` 结构

```
hello_agents/
├── core/          # LLM、Message、Config、Agent 基类
├── agents/        # Simple / ReAct / Reflection / Plan-and-Solve
└── tools/         # 注册表、计算器、搜索、工具链、异步执行
```

设计上只有 Agent 是一等公民，记忆、检索等后续能力都挂在工具上。模型调用走 OpenAI 兼容接口，按环境变量自动识别提供商（DeepSeek、OpenAI、Ollama、vLLM 等）。

## 环境

- Python 3.10+（本地用 3.12）
- 在项目根目录创建虚拟环境并安装本包：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

搜索工具额外依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[search]"
```

第四章、第六章的第三方框架不在本包依赖里，各自目录下有 `requirements.txt` 的按那个装。

### 配置

复制下面内容到项目根目录的 `.env`（该文件已忽略，不会进仓库）：

```env
LLM_API_KEY=your-key
LLM_MODEL_ID=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
# 可选
SERPAPI_API_KEY=
TAVILY_API_KEY=
```

自检环境变量和已装包：

```powershell
.\.venv\Scripts\python.exe _check_env.py
```

## 运行

在项目根目录执行。PowerShell 建议先设 `$env:PYTHONUTF8 = "1"`。

第七章自检（计算器本地即可；四种 Agent 需要可用的 LLM）：

```powershell
.\.venv\Scripts\python.exe code\chapter7\demo_74_75.py
```

最小对话：

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, SimpleAgent

load_dotenv()
agent = SimpleAgent(name="助手", llm=HelloAgentsLLM(), system_prompt="简短回答。")
print(agent.run("用两个字回答：天空通常是什么颜色？"))
```

第四章 ReAct：

```powershell
.\.venv\Scripts\python.exe code\chapter4\ReAct.py
```

第六章 AutoGen 软件团队：

```powershell
.\.venv\Scripts\python.exe code\chapter6\AutoGenDemo\autogen_software_team.py
```

第六章 LangGraph 三步问答：

```powershell
.\.venv\Scripts\python.exe code\chapter6\Langgraph\Dialogue_System.py
```

## 说明

- `.env`、虚拟环境、编辑器本地配置不会提交。
- 第七章的 `FunctionCallAgent`（教程里 0.2.8 之后的内容）还没做。
