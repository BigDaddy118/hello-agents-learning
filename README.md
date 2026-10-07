# Hello-Agents 学习项目

跟着 Datawhale [Hello-Agents](https://hello-agents.datawhale.cc/) 教程做的本地练习。前几章是手写或现成框架实验，第七章起在本仓库里从零搭自己的 `hello_agents` 包。

教程原文：

- [第七章 构建你的 Agent 框架](https://hello-agents.datawhale.cc/#/./chapter7/%E7%AC%AC%E4%B8%83%E7%AB%A0%20%E6%9E%84%E5%BB%BA%E4%BD%A0%E7%9A%84Agent%E6%A1%86%E6%9E%B6)
- [第八章 记忆与检索](https://hello-agents.datawhale.cc/#/./chapter8/%E7%AC%AC%E5%85%AB%E7%AB%A0%20%E8%AE%B0%E5%BF%86%E4%B8%8E%E6%A3%80%E7%B4%A2)
- [第九章 上下文工程](https://hello-agents.datawhale.cc/#/./chapter9/%E7%AC%AC%E4%B9%9D%E7%AB%A0%20%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B)
- [第十章 协议与互操作](https://hello-agents.datawhale.cc/#/./chapter10)
- [第十一章 Agentic RL](https://hello-agents.datawhale.cc/#/./chapter11)
- [第十二章 智能体性能评估](https://hello-agents.datawhale.cc/#/./chapter12/%E7%AC%AC%E5%8D%81%E4%BA%8C%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E6%80%A7%E8%83%BD%E8%AF%84%E4%BC%B0)

## 目录

| 路径 | 内容 |
|------|------|
| `hello_agents/` | 自建框架（第七–十二章：Agent、Memory、RAG、上下文、MCP、RL、评估） |
| `code/chapter4/` | 手写 ReAct、Plan-and-Solve、Memory、Tools、LLM 客户端 |
| `code/chapter6/` | AutoGen、AgentScope、CAMEL、LangGraph 练习 |
| `code/chapter7/` | 框架自检：`demo_74_75.py` |
| `code/chapter8/` | 第八章 Memory/RAG 演示与问答助手 |
| `code/chapter9/` | 第九章 ContextBuilder / NoteTool / TerminalTool / 维护助手演示 |
| `code/chapter10/` | 第十章 MCP 自检 + 自定义天气服务器（`14_weather_*`） |
| `code/chapter11/` | 第十一章 SFT/GRPO 训练与评估流水线 |
| `code/chapter12/` | 第十二章 BFCL / GAIA / 数据生成质量评估 |
| `code/chapter13/helloagents-trip-planner/` | 第十三章智能旅行助手 |
| `code/chapter14/helloagents-deepresearch/` | 第十四章自动化深度研究智能体 |
| `my_app/` | 给 `CodebaseMaintainer` 用的迷你示例代码库 |

### `hello_agents` 结构

```
hello_agents/
├── core/          # LLM、Message、Config、Agent 基类
├── agents/        # Simple / ReAct / Reflection / Plan-and-Solve / CodebaseMaintainer
├── context/       # ContextBuilder（GSSC 流水线）
├── memory/        # 四种记忆 + RAG 管道（本地 TF-IDF 可跑通）
├── protocols/     # 第十章 MCP 客户端/服务器
├── rl/            # 第十一章 SFT/GRPO 训练封装
├── evaluation/    # 第十二章 BFCL / GAIA / 数据生成（LLM Judge、Win Rate）
└── tools/         # 注册表、计算器、搜索、Memory / RAG / Note / Terminal / MCP / RL / BFCL / GAIA / Judge …
```

设计上只有 Agent 是一等公民，记忆、RAG、笔记与终端挂在工具上。模型调用走 OpenAI 兼容接口，按环境变量自动识别提供商（DeepSeek、OpenAI、Ollama、vLLM 等）。Qdrant / Neo4j / MarkItDown / 真实 Embedding 为可选项。

## 环境

- Python 3.10+（本地用 3.12）
- 在项目根目录创建虚拟环境并安装本包：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[search]"   # 搜索
.\.venv\Scripts\python.exe -m pip install -e ".[web]"      # Gradio 问答助手
.\.venv\Scripts\python.exe -m pip install -e ".[memory]"   # Qdrant/Neo4j 等
.\.venv\Scripts\python.exe -m pip install -e ".[protocols]" # 第十章 MCP
.\.venv\Scripts\python.exe -m pip install -e ".[rl]"        # 第十一章 Agentic RL
.\.venv\Scripts\python.exe -m pip install -e ".[evaluation]" # 第十二章评估
# bfcl-eval 与 numpy 版本冲突时另装：pip install "numpy==1.26.4" bfcl-eval
```

第四章、第六章的第三方框架不在本包依赖里，各自目录下有 `requirements.txt` 的按那个装。

### 配置

复制根目录 [`.env.example`](.env.example) 为 `.env`（已 gitignore）。第七章只需 LLM 段；第八章可选填 Qdrant / Neo4j / Embedding（不填则用本地 TF-IDF + JSON）。第九章自检可不配 LLM；`run_96_explore.py` 需要可用的 LLM。

自检环境变量和已装包：

```powershell
.\.venv\Scripts\python.exe _check_env.py
```

## 运行

在项目根目录执行。PowerShell 建议先设 `$env:PYTHONUTF8 = "1"`，并保证能 `import hello_agents`（可装 `-e .`，或临时 `$env:PYTHONPATH = (Get-Location).Path`）。

第七章自检（计算器本地即可；四种 Agent 需要可用的 LLM）：

```powershell
.\.venv\Scripts\python.exe code\chapter7\demo_74_75.py
```

第八章（本地 TF-IDF，无需云端向量库）：

```powershell
$env:EMBED_MODEL_TYPE='tfidf'
.\.venv\Scripts\python.exe code\chapter8\demo_823_memory_ops.py
.\.venv\Scripts\python.exe code\chapter8\demo_825_memory_types.py
.\.venv\Scripts\python.exe code\chapter8\demo_833_rag.py
.\.venv\Scripts\python.exe code\chapter8\demo_84_full.py
```

Gradio 文档问答助手（需先 `pip install gradio`）：

```powershell
.\.venv\Scripts\python.exe code\chapter8\qa_assistant_app.py
```

第九章（上下文工程，自检不依赖 LLM）：

```powershell
.\.venv\Scripts\python.exe code\chapter9\demo_93_context_builder.py
.\.venv\Scripts\python.exe code\chapter9\demo_94_note_tool.py
.\.venv\Scripts\python.exe code\chapter9\demo_95_terminal_tool.py
.\.venv\Scripts\python.exe code\chapter9\demo_96_codebase_maintainer.py
```

用真实 LLM 探索 `my_app/`：

```powershell
.\.venv\Scripts\python.exe code\chapter9\run_96_explore.py
```

第十章 MCP（需 `pip install -e ".[protocols]"`）：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe code\chapter10\demo_102_mcp.py
# 10.5 自定义天气 MCP 服务器
.\.venv\Scripts\python.exe code\chapter10\14_test_weather_server.py
# 可选：接 LLM（需 .env）
.\.venv\Scripts\python.exe code\chapter10\14_weather_agent.py demo
```

第十一章 Agentic RL（需 `pip install -e ".[rl]"`，建议用已装 torch/trl 的解释器；产出写入 `models/`，已 gitignore）：

```powershell
$env:PYTHONIOENCODING='utf-8'
# 小参数端到端：数据 → SFT → 评估 → GRPO → 评估
C:\Python314\python.exe code\chapter11\06_complete_pipeline.py
# 分节练习
C:\Python314\python.exe code\chapter11\04_sft_training.py
C:\Python314\python.exe code\chapter11\05_grpo_training.py
C:\Python314\python.exe code\chapter11\07_model_evaluation.py --max-samples 4 --skip-base
```

第十二章评估（需 `pip install -e ".[evaluation]"`；BFCL 另装 `bfcl-eval`；细节见 [`code/chapter12/README.md`](code/chapter12/README.md)）：

```powershell
$env:PYTHONUTF8 = "1"
# 12.2 BFCL
.\.venv\Scripts\python.exe code\chapter12\test_bfcl_ast.py
.\.venv\Scripts\python.exe code\chapter12\02_bfcl_quick_start.py
# 12.3 GAIA（需 .env 里 HF_TOKEN，并已申请数据集权限）
.\.venv\Scripts\python.exe code\chapter12\test_gaia_match.py
.\.venv\Scripts\python.exe code\chapter12\05_gaia_quick_start.py
# 12.4 数据生成质量（LLM Judge / Win Rate；样例数据已附带）
.\.venv\Scripts\python.exe code\chapter12\test_data_generation_parse.py
.\.venv\Scripts\python.exe code\chapter12\08_data_generation_llm_judge.py
.\.venv\Scripts\python.exe code\chapter12\09_data_generation_win_rate.py
```

或：

```python
from dotenv import load_dotenv
from hello_agents import CodebaseMaintainer, HelloAgentsLLM

load_dotenv()
m = CodebaseMaintainer("my_app", "./my_app", llm=HelloAgentsLLM())
print(m.explore())
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

- `.env`、虚拟环境、`memory_data/`、`knowledge_base/`、`*_notes/`、`models/`、`data/gaia/`、`evaluation_results/`、训练日志、编辑器本地配置不会提交。
- 第七章的 `FunctionCallAgent`（教程里 0.2.8 之后的内容）还没做；第九章 `CodebaseMaintainer` 走「模式预处理 + ContextBuilder」路线。
- GAIA 为 gated 数据集：先在 HuggingFace 申请访问，再把 `HF_TOKEN` 写入 `.env`。
