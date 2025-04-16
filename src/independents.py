from agents import identity_validator
from helper import create_prompt
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages

async def validate_identity(input: str):
    is_valid = await identity_validator.ainvoke([SystemMessage(content=create_prompt(info=[input], llm_type="identity_validator"))])
    return is_valid
