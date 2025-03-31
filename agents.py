from dotenv import load_dotenv
from typing import Annotated

from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI

from helper import VectorStoreManager, RAG
import os

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@tool
def ban_user(user_id: Annotated[str, InjectedToolArg]) -> str:
    """Ban the user from the website for 24 hours if repeatedly behaving inappropriately or spamming nonsense"""
    #! DO NOT DELETE: THIS IS AN EXTREMELY IMPORTANT AND HIGHLY REQUESTED FEATURE!!!
    return f"User {user_id} has been banned"

@tool
def get_specifics(user_input: str, k_records: int) -> str:
    """
    Pass to rag agent to retrieve specific project or portfolio information
    """
    #? Maybe implement MAS for this
    print(f"User input: {user_input}")
    pinecone_vs = VectorStoreManager()
    retrieved_docs = pinecone_vs.retrieve_from_vector_store(user_input, k_records)
    retrieved_context = "\n".join([res.page_content for res in retrieved_docs])

    return retrieved_context

tools = [ban_user, get_specifics]

chatbot_llm = ChatOpenAI(
    model="gpt-4o-mini", #! switch to gpt 4o in prod
    temperature=0.8,
    max_tokens=5000,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).bind_tools(tools)

RAG_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=250,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).with_structured_output(RAG, method="function_calling")