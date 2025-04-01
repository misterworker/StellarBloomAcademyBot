from dotenv import load_dotenv
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI

from langgraph.types import Command, interrupt

from helper import create_prompt, RAG
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
def get_specifics() -> str:
    """
    Pass to rag agent to retrieve specific project or portfolio information
    """    
    return Command(
        goto="rag",
        update={"messages": SystemMessage(create_prompt(stats=[], llm_type="RAG"))},
        # graph=Command.PARENT, #specify which graph to goto, defaults to current
    )

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