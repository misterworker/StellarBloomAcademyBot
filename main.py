from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.types import Command
from build_graph import graph

user_id = "1"

global_config = {"configurable": {"thread_id": "1"}}

def create_prompt(user_type):
    return f"You are an agent called Ethanbot, Ethan's web portfolio manager. You are speaking to {user_type}"

def stream_graph_updates(user_input: str, user_type: str, user_id: str, config: dict):
    state = {
        "messages": [SystemMessage(content=create_prompt(user_type)), {"role": "user", "content": user_input}],
        "user_type": user_type,
        "user_id": user_id,
    }
    for event in graph.stream(state, config):
        if "__interrupt__" in event:
            
            action = input("continue or no: ")
            for resume_event in graph.stream(Command(resume={"action": action}), config):
                print("RESUME_EVENT: ", resume_event)
                try:
                    is_chatbot = resume_event.get("chatbot", False)
                except:
                    is_chatbot = False
                if is_chatbot:
                    msg = resume_event["chatbot"]["messages"][-1].content
                    if msg:
                        print("ASSISTANT:", msg)

        else:
            for value in event.values():
                # print("Value: ", value)
                msg = value["messages"][-1].content
                if msg:
                    print("ASSISTANT:", msg)

while True:
    user_input = input("User ('q' to exit, 'e' to edit previous message): ")
    config = global_config

    if user_input.lower() in ["quit", "q"]:
        break

    elif user_input.lower() in ["edit", "e"]:
        user_input = input("User: ")

        for state in graph.get_state_history(config):
            if "__start__" in state.next:
                config = state.config
                break


    user_type = input("User Type: ").lower()
    if user_type == "":
        user_type = "viewer"


    stream_graph_updates(user_input, user_type, user_id, config)
