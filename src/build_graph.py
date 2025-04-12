from copy import deepcopy
from typing_extensions import Literal

from langchain_core.messages import ToolMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from agents import chatbot_llm, RAG_llm, fetch_contributions, suspend_user, get_specifics, draft_email
from helper import create_prompt, State, VectorStoreManager

graph_builder = StateGraph(State)

#* Agent Nodes
async def chatbot(state: State):
    message = await chatbot_llm.ainvoke(state["messages"])
    assert len(message.tool_calls) <= 1
    tool_calls = []
    for tool_call in message.tool_calls:
        if tool_call["name"] == "suspend_user":
            tool_call_copy = deepcopy(tool_call)
            tool_call_copy["args"]["fingerprint"] = state["fingerprint"]
            tool_calls.append(tool_call_copy)
        else:
            tool_calls.append(tool_call)

    message.tool_calls = tool_calls

    return {"messages": [message]}

async def rag(state: State):
    try:
        result = await RAG_llm.ainvoke(state["messages"])
        search_term = result.search_term
        k_records = result.k_records
        pinecone_vs = VectorStoreManager()
        retrieved_docs = pinecone_vs.retrieve_from_vector_store(search_term, k_records)
        retrieved_context = "\n".join([res.page_content for res in retrieved_docs])
        message = await chatbot_llm.ainvoke(
            [SystemMessage(content = create_prompt(info=[retrieved_context], llm_type="RAG_CHATBOT"))] + state["messages"]
        )
        return {"messages": [message]}
    except Exception as e:
        print("❌RAG error: ", e)

#* Tool related nodes
def route_after_llm(state) -> Literal[END, "human_review_node", "tools"]:
    tools = state["messages"][-1].tool_calls
    if len(tools) == 0:
        return END
    elif tools[0].get("name", "") in ["suspend_user"]:
        # Automatically suspend user
        return "tools"
    return "human_review_node"
        
def human_review_node(state) -> Command[Literal["chatbot", "tools"]]:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[-1]

    # this is the value we'll be providing via Command(resume=<human_review>)
    human_review = interrupt(
        {
            "question": "Is this correct?",
            # Surface tool calls for review
            "tool_call": tool_call,
        }
    )
    review_action = human_review["action"]
    # if approved, call the tool
    if review_action:
        return Command(goto="tools")
    tool_message = ToolMessage(
        role="tool",
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        content="Tool execution skipped.",
    )
    return Command(goto="chatbot", update={"messages": [tool_message]})
    
async def tool_node(state):
    new_messages = []
    tools = {"suspend_user": suspend_user, "get_specifics": get_specifics, "draft_email": draft_email, "fetch_contributions": fetch_contributions}
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

def route_after_tool(state) -> Literal["rag", "chatbot"]:
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage) and last_message.name == "get_specifics":
        return "rag"
    
    return "chatbot"

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node(human_review_node)
graph_builder.add_node("rag", rag)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_after_llm,
)
graph_builder.add_conditional_edges(
    "tools",
    route_after_tool,
)
graph_builder.add_edge("rag", END)
    