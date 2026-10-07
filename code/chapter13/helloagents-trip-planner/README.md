# HelloAgents 智能旅行助手

跟着 [Hello-Agents 第十三章](https://hello-agents.datawhale.cc/#/./chapter13/%E7%AC%AC%E5%8D%81%E4%B8%89%E7%AB%A0%20%E6%99%BA%E8%83%BD%E6%97%85%E8%A1%8C%E5%8A%A9%E6%89%8B) 跑通的旅行规划 Web 应用：四个 Agent 协作生成行程，高德 MCP 搜景点/天气/酒店，前端地图展示。

源码来自 [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents/tree/main/code/chapter13/helloagents-trip-planner)，本地补了 JS API 2.0 所需的安全密钥配置。

## 不要把密钥推到 GitHub

`.env` 已在 `.gitignore` 里。仓库里只有 `.env.example`（占位符）。

不要提交、不要写进 README：

- 高德 Web 服务 Key、JS API Key、安全密钥
- LLM API Key
- Unsplash Key

每人自己复制 example 再填。

## 功能

- 填城市、日期、偏好，生成多日行程（景点、餐饮、酒店、预算）
- 四个 `SimpleAgent`：景点搜索 → 天气 → 酒店 → 行程规划
- 三个搜类 Agent **共用一个** 高德 MCP 进程（`uvx amap-mcp-server`）
- 结果页地图打点、编辑顺序/删除、导出 PNG/PDF（导出时地图可能被隐藏）

## 需要准备

- Python 3.10–3.12（3.12 已验证）
- Node.js 16+
- LLM Key（DeepSeek / OpenAI 等，OpenAI 兼容接口）
- 高德 [控制台](https://console.amap.com/) 两把 Key：
  - **Web 服务**：后端 MCP（搜索、天气）。IP 白名单本地可空
  - **Web 端 (JS API)**：前端地图。域名白名单本地请空着，不要填 `localhost`
  - JS Key 旁边的 **安全密钥**（2.0 必填）
- Unsplash：可选，不填就没有景点图

## 配置

### 后端 `backend/.env`

```bash
cd backend
cp .env.example .env
```

```
LLM_MODEL_ID=deepseek-chat
LLM_API_KEY=your-llm-key
LLM_BASE_URL=https://api.deepseek.com
AMAP_API_KEY=your_web_service_key
UNSPLASH_ACCESS_KEY=
```

### 前端 `frontend/.env`

```bash
cd frontend
cp .env.example .env
```

```
VITE_API_BASE_URL=http://localhost:8000
VITE_AMAP_WEB_KEY=your_web_service_key
VITE_AMAP_WEB_JS_KEY=your_js_api_key
VITE_AMAP_SECURITY_JS_CODE=your_security_js_code
```

两把高德 Key 不能混用。Web 服务那把给 `AMAP_API_KEY`；JS 那把给 `VITE_AMAP_WEB_JS_KEY`。

## 运行

后端：

```bash
cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python run.py
```

http://localhost:8000/docs

前端（另开终端）：

```bash
cd frontend
npm install
npm run dev
```

http://localhost:5173

第一次点「开始规划」会下载 `amap-mcp-server`，要等十几到几十秒。

## 结构

```
backend/app/
  models/schemas.py          # 13.2 Pydantic 模型
  agents/trip_planner_agent.py  # 13.3 四 Agent + 13.4 共享 MCP
  api/routes/                # FastAPI
  services/                  # LLM / 高德 / Unsplash
frontend/src/
  types/index.ts             # 与后端对齐的 TS 类型
  views/Home.vue             # 表单
  views/Result.vue           # 结果、地图、编辑、导出
```

`POST /api/trip/plan` 是主接口。后端依赖 PyPI 包 `hello-agents[protocols]`（0.2.4–0.2.9），不是教程仓库里手写的 `hello_agents`。

## 协议

教程原文为 [CC BY-NC-SA 4.0](https://github.com/datawhalechina/hello-agents)。
