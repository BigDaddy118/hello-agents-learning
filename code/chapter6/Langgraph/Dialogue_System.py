"""
6.5.2 三步问答助手 — LangGraph：理解 → 搜索 → 回答
搜索优先 Tavily；无 TAVILY_API_KEY 时回退到项目已有的 SerpAPI。
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "code" / "chapter4"))

from tools import search as serpapi_search  # noqa: E402


class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: str


llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.7,
)

_tavily = None
if os.getenv("TAVILY_API_KEY"):
    from tavily import TavilyClient

    _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def understand_query_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词"""
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content
    search_query = user_message

    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
    elif "Search terms:" in response_text:
        search_query = response_text.split("Search terms:")[1].strip()

    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我理解您的需求：{response_text}")],
    }


def _tavily_search(query: str) -> str:
    assert _tavily is not None
    response = _tavily.search(
        query=query,
        search_depth="basic",
        include_answer=True,
        include_raw_content=False,
        max_results=5,
    )
    parts: list[str] = []
    if response.get("answer"):
        parts.append(f"综合答案：\n{response['answer']}")
    if response.get("results"):
        lines = ["相关信息："]
        for i, result in enumerate(response["results"][:3], 1):
            lines.append(
                f"{i}. {result.get('title', '')}\n"
                f"{result.get('content', '')}\n来源：{result.get('url', '')}"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts) or "抱歉，没有找到相关信息。"


def tavily_search_node(state: SearchState) -> dict:
    """步骤2：真实搜索（Tavily 或 SerpAPI）"""
    search_query = state["search_query"]
    try:
        print(f"🔍 正在搜索: {search_query}")
        if _tavily:
            search_results = _tavily_search(search_query)
        elif os.getenv("SERPAPI_API_KEY"):
            search_results = serpapi_search(search_query)
        else:
            raise RuntimeError("未配置 TAVILY_API_KEY 或 SERPAPI_API_KEY")

        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [
                AIMessage(content="✅ 搜索完成！找到了相关信息，正在为您整理答案...")
            ],
        }
    except Exception as e:
        print(f"❌ 搜索时发生错误: {e}")
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [
                AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")
            ],
        }

def generate_answer_node(state: SearchState) -> SearchState:
    """步骤3：基于搜索结果生成最终答案"""
    
    # 检查是否有搜索结果
    if state["step"] == "search_failed":
        # 如果搜索失败，基于LLM知识回答
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""
        
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
        
        return {
            "final_answer": response.content,
            "step": "completed",
            "messages": [AIMessage(content=response.content)]
        }
    
    # 基于搜索结果生成答案
    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""

    response = llm.invoke([SystemMessage(content=answer_prompt)])
    
    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }

# 构建搜索工作流
def create_search_assistant():
    workflow = StateGraph(SearchState)
    
    # 添加三个节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    
    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

async def main():
    """主函数：运行智能搜索助手"""
    if not (_tavily or os.getenv("SERPAPI_API_KEY")):
        print("❌ 错误：请在 .env 配置 TAVILY_API_KEY 或 SERPAPI_API_KEY")
        return

    app = create_search_assistant()
    backend = "Tavily" if _tavily else "SerpAPI"
    print("🔍 智能搜索助手启动！（6.5.2 理解→搜索→回答）")
    print(f"搜索后端: {backend}")
    print("(输入 'quit' 退出)\n")
    
    session_count = 0
    
    while True:
        user_input = input("🤔 您想了解什么: ").strip()
        
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        
        if not user_input:
            continue
        
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        # 初始状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start"
        }
        
        try:
            print("\n" + "="*60)
            
            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")

if __name__ == "__main__":
    asyncio.run(main())