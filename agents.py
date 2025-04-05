from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI

from langgraph.types import Command, interrupt

from helper import create_prompt, RAG
import os

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#TODO: Feedback to Ethan
#TODO: Parse website content as pdf/docx and provide as downloadable content
#TODO: Update prompt to prevent more than 1 tool call
@tool
def suspend_user(fingerprint: Annotated[str, InjectedToolArg]) -> str:
    """Temporarily suspend user that sends any inappropriate, unsafe or spam messages."""
    # Only banned during runtime at the moment
    return f"blud got suspended"

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

tools = [suspend_user, get_specifics]

chatbot_llm = ChatOpenAI(
    model="gpt-4o-mini", #! switch to gpt 4o in prod
    temperature=0.8,
    max_tokens=5000,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).bind_tools(tools)

RAG_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=250,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).with_structured_output(RAG, method="function_calling")