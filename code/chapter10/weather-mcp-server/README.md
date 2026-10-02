# Weather MCP Server

基于 [wttr.in](https://wttr.in) 的实时天气查询 MCP 服务器（**Streamable HTTP**），用于 HelloAgents / Datawhale 第十章实践与 [Smithery](https://smithery.ai) 发布。

- **GitHub**: https://github.com/BigDaddy118/weather-mcp-server  
- **Smithery**: https://smithery.ai/servers/1009264368/weather-mcp-server  

## 功能

| 工具 | 说明 |
|------|------|
| `get_weather(city)` | 查询城市当前天气（支持中文城市名与英文名） |
| `list_supported_cities()` | 列出内置中文城市映射 |
| `get_server_info()` | 服务器元信息 |

内置中文城市：北京、上海、广州、深圳、杭州、成都、重庆、武汉、西安、南京、天津、苏州。其他城市可直接传英文名。

## 快速开始

```bash
git clone https://github.com/BigDaddy118/weather-mcp-server.git
cd weather-mcp-server
pip install -r requirements.txt
python server.py
```

默认监听：`http://0.0.0.0:8081/mcp`

环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `PORT` | `8081` | 端口 |
| `HOST` | `0.0.0.0` | 绑定地址 |
| `MCP_TRANSPORT` | `streamable-http` | MCP 传输（Smithery 需 HTTP） |

## Docker

```bash
docker build -t weather-mcp-server .
docker run --rm -p 8081:8081 weather-mcp-server
```

## 在 Smithery 使用 / 发布

已发布标识：`1009264368/weather-mcp-server`

本地临时公网（开发用）：

```bash
python server.py
# 另开终端：npx localtunnel --port 8081
# 得到 https://xxxx.loca.lt 后，MCP URL 为 https://xxxx.loca.lt/mcp
```

正式发布示例：

```bash
npx @smithery/cli auth login
npx @smithery/cli mcp publish "https://你的公网域名/mcp" -n 1009264368/weather-mcp-server
```

> Smithery 需要**正在运行的 HTTP MCP 地址**，不能填本 GitHub 仓库 URL。临时隧道关掉后线上会不可用，生产环境请部署到稳定主机。

## 在 HelloAgents 中使用

学习仓库里的 stdio 演示见 `14_weather_mcp_server.py`。本仓库面向 HTTP：

```python
# 连接已部署的 HTTP MCP（视 MCPTool / 客户端是否支持 URL）
# 或本地 stdio：把本目录 server 改为 stdio 传输后再用
# MCPTool(server_command=[sys.executable, "server.py"])
```

## 示例返回

`get_weather("北京")`：

```json
{
  "city": "北京",
  "temperature": 22.0,
  "feels_like": 21.0,
  "humidity": 40,
  "condition": "Sunny",
  "wind_speed": 2.5,
  "visibility": 10.0,
  "timestamp": "2026-10-01 17:00:00"
}
```

## 依赖

- Python >= 3.10  
- `fastmcp>=2.0.0`  
- `requests>=2.31.0`  

## 许可证

MIT
