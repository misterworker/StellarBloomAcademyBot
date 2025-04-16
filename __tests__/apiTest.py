import requests, json

API_URL = "http://127.0.0.1:8000"

def chat_with_bot(user_id: str, user_input: str, name: str, bot_name: str):
    """Start a conversation with the chatbot."""
    payload = {
        "user_id": user_id,
        "user_input": user_input,
        "name": name,
        "bot_name": bot_name,
    }

    response = requests.post(f"{API_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        response = data["response"]
        other_name = data["other_name"]
        if other_name == "interrupt":
            print("Human Review Time!")
            return True
        else:
            print("ASSISTANT CHAT:", response)
            return False
    else:
        print("Error:", response.status_code, response.text)
        return False


def resume_conversation(user_id: str, action: bool):
    """Resume the conversation if paused for user input — with streaming."""
    payload = {
        "action": action,
        "user_id": user_id,
    }

    with requests.post(f"{API_URL}/resume", json=payload, stream=True) as response:
        if response.status_code == 200:
            print("ASSISTANT RESUME:", end=" ", flush=True)
            tool_name = None
            tool_msg = None

            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8").replace("data: ", "")
                    try:
                        data = json.loads(decoded)

                        if data.get("other_name") is not None:
                            tool_name = data["other_name"]
                            tool_msg = data.get("other_msg")

                        if "response" in data and data["response"]:
                            print(data["response"], end="", flush=True)
                    except Exception as e:
                        print(f"\n[Stream Parse Error] {e} — line: {decoded}")

            if tool_name:
                print("\nTool Called!")
                if tool_name == "provide_feedback":
                    print("Email Body: ", tool_msg)
            print()  # newline after stream
        else:
            print("Error:", response.status_code, response.text)


def wipe_thread(user_id: str):
    payload = {
        "user_id": user_id,
    }
    response = requests.post(f"{API_URL}/wipe", json=payload)
    if response.status_code == 200:
        data = response.json()
        print("ASSISTANT WIPE:", data.get("response", "No message from assistant"))
    else:
        print("Error:", response.status_code, response.text)

def interact_with_chatbot():
    """Interact with the chatbot through the FastAPI API."""
    user_id = "abc"
    user_input = input("You (w to wipe thread): ")
    if user_input == "w":
        wipe_thread(user_id)
        return

    conversation_active = chat_with_bot(user_id, user_input, "Ethan", "Orion")

    # If the conversation requires user input to continue, ask the user
    while conversation_active:
        user_decision = input("Do you want to continue? (yes/no): ").strip().lower()

        if user_decision == "yes":
            resume_conversation(user_id, True)
            break
        else:
            resume_conversation(user_id, False)
            break

while True:
    interact_with_chatbot()
