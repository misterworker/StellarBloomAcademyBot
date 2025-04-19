from dotenv import load_dotenv
from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from langgraph.types import Command, interrupt

from config import GPT_TYPE
from helper import create_prompt, ValidateLore, Identify

import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

@tool
async def get_lore() -> str:
    """
    Pass to rag agent to retrieve story, lore or ambiguous references to entities
    """    
    return Command(
        goto="lore_rag",
        # update={"messages": SystemMessage(create_prompt(info=[], llm_type="RAG"))},
        # graph=Command.PARENT, #specify which graph to goto, defaults to current
    )

tools = [get_lore]

gpt_chatbot = ChatOpenAI(
    model=GPT_TYPE,
    temperature=0.9,
    max_tokens=None,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).bind_tools(tools)

ds_chatbot = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.9,
    max_tokens=None,
    timeout=30,
    max_retries=1,
    api_key=DEEPSEEK_API_KEY,
).bind_tools(tools)

gpt_lorebot = ChatOpenAI(
    model=GPT_TYPE,
    temperature=0,
    max_tokens=250,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).with_structured_output(ValidateLore, method="function_calling")

splitter = ChatOpenAI(
    model=GPT_TYPE,
    temperature=0.1,
    max_tokens=5000,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)

identity_validator = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=250,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).with_structured_output(Identify, method="function_calling")
