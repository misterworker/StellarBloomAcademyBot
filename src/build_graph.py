from copy import deepcopy
from typing_extensions import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from agents import chatbot_llm, splitter
from helper import create_prompt, State, VectorStoreManager

graph_builder = StateGraph(State)

#* Agent Nodes
async def chatbot(state: State):
    message = await chatbot_llm.ainvoke(state["messages"])
    assert len(message.tool_calls) <= 1

    return {"messages": [message]}

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
    tools = {}
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

def route_after_tool(state) -> Literal["chatbot"]:
    last_message = state["messages"][-1]
    
    return "chatbot"

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("splitter", splitter_bot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_after_llm,
)
graph_builder.add_conditional_edges(
    "tools",
    route_after_tool,
)
graph_builder.add_edge("splitter", END)
    