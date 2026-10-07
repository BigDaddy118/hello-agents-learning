# HelloAgents 自动化深度研究智能体

把一个开放主题拆成 3–5 个子任务，逐个搜索、总结、记笔记，最后生成带引用的 Markdown 报告。前端用 SSE 实时显示进度。

跟着 [Hello-Agents 第十四章](https://hello-agents.datawhale.cc/#/./chapter14/%E7%AC%AC%E5%8D%81%E5%9B%9B%E7%AB%A0%20%E8%87%AA%E5%8A%A8%E5%8C%96%E6%B7%B1%E5%BA%A6%E7%A0%94%E7%A9%B6%E6%99%BA%E8%83%BD%E4%BD%93) 落地的 Web 应用。上游代码：[datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents/tree/main/code/chapter14/helloagents-deepresearch)。

## 不要把密钥推到 GitHub

`.env` 已在 `.gitignore` 里。仓库里只有 `backend/.env.example`。

不要提交、不要写进 README：

- LLM API Key
- Tavily / Perplexity / SerpApi 等搜索 Key

每人自己复制 example 再填。

## 功能

- 输入研究主题，可选覆盖搜索引擎（DuckDuckGo / Tavily / Perplexity / SearXNG / Advanced）
- 三个 Agent 顺序协作：TODO Planner → Task Summarizer → Report Writer
- SearchTool 检索，NoteTool 把子任务笔记落到本地工作区
- FastAPI `/research/stream` 推送进度；Vue 3 全屏展示任务、日志和最终报告

## 需要准备

- Python 3.10–3.12（本机用 3.12 验证；系统默认 3.14 可能装不上部分依赖）
- Node.js 16+
- LLM Key（DeepSeek / OpenAI 等，OpenAI 兼容接口）
- 搜索：默认 `duckduckgo` 无需 Key；换 Tavily 等再填对应 Key

## 配置

```powershell
cd backend
copy .env.example .env
```

至少填写：

```
SEARCH_API=duckduckgo
LLM_PROVIDER=custom
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=your-llm-key
LLM_BASE_URL=https://api.deepseek.com
CORS_ORIGINS=http://localhost:5173,http://localhost:5174,http://localhost:3000
```

前端可选：复制 `frontend/.env.example` 为 `frontend/.env.local`（默认已指向 `http://localhost:8000`）。

## 运行

### 后端（端口 8000）

```powershell
cd backend
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install fastapi "hello-agents==0.2.9" tavily-python python-dotenv requests openai "uvicorn[standard]" ddgs loguru pydantic huggingface_hub

# Windows 控制台需 UTF-8，否则 hello-agents 打印警告会因 emoji 崩溃
$env:PYTHONUTF8 = "1"
python src/main.py
```

健康检查：<http://localhost:8000/healthz>

官方 `pip install -e .` 在当前 `pyproject.toml` 包布局下会失败，按上面列出的依赖安装即可。

### 前端（端口 5174）

新开终端：

```powershell
cd frontend
npm install
npm run dev
```

浏览器打开 <http://localhost:5174>，输入主题（如 `Datawhale是一个什么样的组织？`），点「开始研究」。完整一轮大约 1–3 分钟。

## 结构

```
backend/                 FastAPI + HelloAgents
  src/agent.py           研究协调器
  src/main.py            /research、/research/stream
  src/services/          规划 / 搜索 / 总结 / 报告 / 笔记
frontend/                Vue 3 + Vite + TypeScript
```

## 说明

本仓库在上游示例上补了 `load_dotenv`（启动时读取 `backend/.env`），并记录了 Windows UTF-8 与 `huggingface_hub` 依赖问题。
