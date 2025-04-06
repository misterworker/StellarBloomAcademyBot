from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI

from langgraph.types import Command, interrupt

from helper import create_prompt, RAG
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        update={"messages": SystemMessage(create_prompt(info=[], llm_type="RAG"))},
        # graph=Command.PARENT, #specify which graph to goto, defaults to current
    )

@tool
def provide_feedback(feedback: str) -> None:
    """
    Pass feedback to Ethan. This feedback is the actual body of an email.
    """
    print("Feedback Passed: ", feedback)

    return None

tools = [suspend_user, get_specifics, provide_feedback]

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