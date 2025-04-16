from dotenv import load_dotenv
from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI

from langgraph.types import Command, interrupt

from config import GPT_TYPE
from helper import create_prompt, Identify

import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tools = []

chatbot_llm = ChatOpenAI(
    model=GPT_TYPE,
    temperature=0.9,
    max_tokens=5000,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).bind_tools(tools)

identity_validator = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=250,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).with_structured_output(Identify, method="function_calling")

splitter = ChatOpenAI(
    model=GPT_TYPE,
    temperature=0.1,
    max_tokens=5000,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
)
