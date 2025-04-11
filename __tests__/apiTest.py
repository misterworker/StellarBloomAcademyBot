import requests
import time

API_URL = "http://127.0.0.1:8000"

def chat_with_bot(user_id: str, user_input: str, fingerprint: str, num_rewind: int = 0):
    """Start a conversation with the chatbot."""
    payload = {
        "user_id": user_id,
        "user_input": user_input,
        "fingerprint": fingerprint,
        "num_rewind": num_rewind
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
    """Resume the conversation if paused for user input."""
    payload = {
        "action": action,
        "user_id": user_id,
    }

    # Sending the resume decision to the FastAPI app
    response = requests.post(f"{API_URL}/resume", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        other_name = data["other_name"]
        if other_name != None:
            print("Tool Called!")
            if other_name == "provide_feedback":
                other_msg = data["other_msg"]
                # In app, go to email and paste other_msg in body
                print("Email Body: ", other_msg)
        print("ASSISTANT RESUME:", data.get("response", "No message from assistant"))
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
    fingerprint = "pseudoFingerprint"
    user_input = input("You (w to wipe thread): ")
    if user_input == "w":
        wipe_thread(user_id)
        return

    conversation_active = chat_with_bot(user_id, user_input, fingerprint)

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
