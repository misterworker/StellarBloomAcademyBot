from copy import deepcopy
from typing_extensions import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from agents import ds_chatbot, gpt_chatbot, gpt_lorebot, splitter, get_lore
from helper import create_prompt, State, VectorStoreManager

import asyncio

graph_builder = StateGraph(State)


#* Agent Nodes
async def chatbot(state: State):
    async def get_deepseek_response():
        return await ds_chatbot.ainvoke(state["messages"])

    async def get_gpt_response():
        await asyncio.sleep(9)
        return await gpt_chatbot.ainvoke(state["messages"])
    
    try:
        # Create tasks for each coroutine
        ds_task = asyncio.create_task(get_deepseek_response())
        gpt_task = asyncio.create_task(get_gpt_response())
        done, pending = await asyncio.wait([ds_task, gpt_task], return_when=asyncio.FIRST_COMPLETED, timeout=30)

        for task in done: 
            response = task.result(); break
        for task in pending: task.cancel()

    except asyncio.TimeoutError:
        response.content = "gtg, ttyl."

    if not response.content: response.content = "gtg, ttyl."
    
    assert len(response.tool_calls) <= 1

    return {"messages": [response]}

async def lorebot(state: State):
    try:
        last_message = state["messages"][-1]
        print("Last Message: ", last_message)
        pinecone_vs = VectorStoreManager()
        retrieved_docs = pinecone_vs.retrieve_from_vector_store(last_message, 3)
        retrieved_context = "\n".join([res.page_content for res in retrieved_docs])
        result = await gpt_lorebot.ainvoke([SystemMessage(create_prompt(info=[last_message, retrieved_context], llm_type="lore_validator"))])
        print("rag validator result: ", result)
        if result.is_neccessary:
            return Command(
                goto="chatbot",
                update={"messages": retrieved_context},
            )
        else:
            return "chatbot"
    except Exception as e:
        print("❌RAG error: ", e)

async def splitter_bot(state: State):
    last_message = state["messages"][-1]
    message = await splitter.ainvoke([SystemMessage(create_prompt(info=[last_message], llm_type="splitter"))])

    return {"messages": [message]}

#* Tool related nodes
def route_after_llm(state) -> Literal["splitter", "tools"]:
    tools = state["messages"][-1].tool_calls
    if len(tools) == 0:
        return "splitter"
    return "tools"
    
async def tool_node(state):
    new_messages = []
    tools = tools = {"get_lore": get_lore}
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = await tool.ainvoke(tool_call["args"])
        new_messages.append(
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": result,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": new_messages}

def route_after_tool(state) -> Literal["lore_rag", "chatbot"]:
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage) and last_message.name == "get_lore":
        return "lore_rag"
    
    return "chatbot"

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("lore_rag", lorebot)
graph_builder.add_node("splitter", splitter_bot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("lore_rag", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_after_llm,
)
graph_builder.add_conditional_edges(
    "tools",
    route_after_tool,
)
graph_builder.add_edge("splitter", END)
    