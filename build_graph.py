from langchain_core.messages import ToolMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from typing import Annotated
from typing_extensions import TypedDict, Literal

from agents import chatbot_llm, RAG_llm, suspend_user, get_specifics
from helper import VectorStoreManager

memory = MemorySaver()

class State(TypedDict):
    """Add attributes that are mutable via nodes, for example if the user type can change from guest to user with the help of
    a node in the graph, we should add user type as an attribute"""

    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    fingerprint: str


graph_builder = StateGraph(State)

from copy import deepcopy

#* Agent Nodes
def chatbot(state: State):
    message = chatbot_llm.invoke(state["messages"])
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


def rag(state: State):
    result = RAG_llm.invoke(state["messages"])
    print("rag result: ", result)
    search_term = result.search_term
    k_records = result.k_records
    pinecone_vs = VectorStoreManager()
    retrieved_docs = pinecone_vs.retrieve_from_vector_store(search_term, k_records)
    retrieved_context = "\n".join([res.page_content for res in retrieved_docs])

    return {"messages": [retrieved_context]}

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
    review_data = human_review.get("data")

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
    
def tool_node(state):
    new_messages = []
    tools = {"suspend_user": suspend_user, "get_specifics": get_specifics}
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        print("result: ", result)
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
    print("route state after tool: ", last_message)

    if isinstance(last_message, ToolMessage) and last_message.name == "get_specifics":
        return "rag"
    
    return "chatbot"

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("rag", rag)
graph_builder.add_node(human_review_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_after_llm,
)
# graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges(
    "tools",
    route_after_tool,
)
graph_builder.add_edge("rag", END)

graph = graph_builder.compile(checkpointer=memory)

try:
    with open("graph_output.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
except Exception:
    # This requires some extra dependencies and is optional
    pass
    