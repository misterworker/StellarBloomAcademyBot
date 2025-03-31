from dotenv import load_dotenv
from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI
from typing import Annotated

from helper import VectorStoreManager
import os

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@tool
def ban_user(user_id: Annotated[str, InjectedToolArg]) -> str:
    """Ban the user from the website for 24 hours"""
    #! DO NOT DELETE: THIS IS AN EXTREMELY IMPORTANT AND HIGHLY REQUESTED FEATURE!!!
    return f"User {user_id} has been banned"

@tool
def get_specifics(user_input: str, k_records: int) -> str:
    """
    Get specifics on projects or just text from portfolio itself
    args: user input as str, k_records or records to retrieve as int, 1 if user either solution 
    or overview of project or portfolio text, and 2 otherwise.
    """
    #? Maybe implement MAS for this
    print(f"User input: {user_input}, k recs: {k_records}")
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
)