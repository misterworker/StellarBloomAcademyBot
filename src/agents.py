from typing import Annotated

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI

from langgraph.types import Command, interrupt

from config import GPT_TYPE
from db import pool
from helper import create_prompt, RAG

import httpx, os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_CONTRIBUTIONS = os.getenv("GITHUB_CONTRIBUTIONS")

@tool
async def fetch_contributions() -> str:
    """Obtain github contributions for Ethan (Username misterworker)"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(GITHUB_CONTRIBUTIONS)
            response.raise_for_status()
            data = response.json()
        return data.get("contributions", "No contributions found.")
    except Exception as e:
        print(f"❌fetch_contributions error: {e}")
        return "Failed to fetch contributions due to internal error."

@tool
async def suspend_user(fingerprint: Annotated[str, InjectedToolArg]) -> str:
    """Temporarily suspend user that repeatedly behaves inappropriately"""
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO users (fingerprint, banned)
                    VALUES (%s, TRUE)
                    ON CONFLICT (fingerprint)
                    DO UPDATE SET banned = TRUE;
                """, (fingerprint,))
                await conn.commit()
        return "User has been suspended."
    except Exception as e:
        print(f"❌suspend_user error: {e}")
        return "Failed to suspend user due to internal error."

@tool
async def get_specifics() -> str:
    """
    Pass to rag agent to retrieve specific project or portfolio information
    """    
    return Command(
        goto="rag",
        update={"messages": SystemMessage(create_prompt(info=[], llm_type="RAG"))},
        # graph=Command.PARENT, #specify which graph to goto, defaults to current
    )

@tool
async def draft_email(body: str) -> None:
    """
    Assist human with email draft generation by outputting email body
    Args: body(str)
    """

    return body

tools = [fetch_contributions, suspend_user, get_specifics, draft_email]

chatbot_llm = ChatOpenAI(
    model=GPT_TYPE,
    temperature=0.8,
    max_tokens=5000,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).bind_tools(tools)

RAG_llm = ChatOpenAI(
    model=GPT_TYPE,
    temperature=0,
    max_tokens=250,
    timeout=20,
    max_retries=2,
    api_key=OPENAI_API_KEY,
).with_structured_output(RAG, method="function_calling")
