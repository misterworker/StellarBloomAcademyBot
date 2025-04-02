# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.types import Command

from build_graph import graph
from helper import create_prompt, rewind

fingerprint = "123"
global_config = {"configurable": {"thread_id": fingerprint}}

def stream_graph_updates(user_input: str, fingerprint: str, num_rewind: int, config: dict):
    state = {
        "messages": [SystemMessage(content=create_prompt([230, 3], "chatbot")), {"role": "user", "content": user_input}],
        "fingerprint": fingerprint,
    }
    if num_rewind != 0:
        rewind(int(num_rewind), config, user_input)
    for event in graph.stream(state, config):
        if "__interrupt__" in event:
            action = input("continue or no: ")
            for resume_event in graph.stream(Command(resume={"action": action}), config):
                try:
                    is_chatbot = resume_event.get("chatbot", False)
                    is_rag = resume_event.get("rag", False)
                except:
                    is_chatbot = False
                    is_rag = False
                if is_chatbot:
                    msg = resume_event["chatbot"]["messages"][-1].content
                    if msg:
                        print("ASSISTANT:", msg, "\n")
                elif is_rag:
                    msg = resume_event["rag"]["messages"][-1]
                    if msg:
                        print("ASSISTANT:", msg, "\n")

        elif "tools" in event:
            # For tools that don't require human review
            pass

        elif "chatbot" in event:
            for value in event.values():
                msg = value["messages"][-1].content
                if msg:
                    print("ASSISTANT:", msg, "\n")

while True:
    num_rewind = 0
    
    config = global_config

    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
        print("-" * 80)

    user_input = input("User ('q' to exit, 'e' to edit previous messages): ")

    if user_input.lower() in ["quit", "q"]:
        break

    elif user_input.lower() in ["edit", "e"]:
        num_rewind = input("Number of messages to rewind (num pls): ")
        user_input = input("User: ")
        
    stream_graph_updates(user_input, fingerprint, num_rewind, config)
