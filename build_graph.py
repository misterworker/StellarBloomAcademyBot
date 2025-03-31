from langchain_core.messages import ToolMessage

from typing import Annotated
from typing_extensions import TypedDict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from agents import chatbot_llm, RAG_llm, ban_user, get_specifics

memory = MemorySaver()

class State(TypedDict):
    """Add attributes that are mutable via nodes, for example if the user type can change from guest to user with the help of
    a node in the graph, we should add user type as an attribute"""

    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    user_type: str
    user_id: str


graph_builder = StateGraph(State)

from copy import deepcopy

def chatbot(state: State):
    message = chatbot_llm.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    tool_calls = []
    for tool_call in message.tool_calls:
        if tool_call["name"] == "ban_user":
            tool_call_copy = deepcopy(tool_call)
            tool_call_copy["args"]["user_id"] = state["user_id"]
            tool_calls.append(tool_call_copy)
        else:
            tool_calls.append(tool_call)

    message.tool_calls = tool_calls

    return {"messages": [message], "user_type": state["user_type"]}

def tool_node(state):
    new_messages = []
    tools = {"ban_user": ban_user, "get_specifics": get_specifics}
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

def route_after_llm(state) -> Literal[END, "human_review_node"]:
    if len(state["messages"][-1].tool_calls) == 0:
        return END
    else:
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
    if review_action == "continue":
        return Command(goto="tools")
    else:
        tool_message = ToolMessage(
            role="tool",
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            content="Tool execution skipped.",
        )
        return Command(goto="chatbot", update={"messages": [tool_message]})

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node(human_review_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_after_llm,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile(checkpointer=memory)

# try:
#     with open("graph_output.png", "wb") as f:
#         f.write(graph.get_graph().draw_mermaid_png())
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
    