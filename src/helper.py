from pinecone import Pinecone
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import os

#? Langgraph Database Connection Pool
DB_URI = os.getenv("DB_URI")
pool = AsyncConnectionPool(
    conninfo=DB_URI,
    max_size=5,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)

#? Langgraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str

#? Pinecone Vector Store Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "portfolio"
class VectorStoreManager:
    def __init__(self):
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        Pinecone(api_key=PINECONE_API_KEY)

    def __get_vector_store(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    def retrieve_from_vector_store(self, query: str, top_k: int):
        vector_store = self.__get_vector_store()
        results = vector_store.similarity_search(query, k=top_k)
        return results
    
#? Class for Structured Outputs
class Identify(BaseModel):
    """Check if information provided is valid"""
    is_valid: bool = Field(description="is email, age and name provided and valid?")

#? Endpoint Inputs
class UserInput(BaseModel):
    user_id: str
    user_input: str
    name: str
    bot_name: str

class WipeInput(BaseModel):
    user_id: str

class ValidateIdentityInput(BaseModel):
    user_input: str

class AddAiMsgInput(BaseModel):
    ai_msg: str
    user_id: str

#? Helper functions
def create_prompt(info:list, llm_type:str):
    print("LLM Type: ", llm_type)
    if llm_type == "chatbot":
        name = info[0]
        bot_name = info[1]
        prompt = f"""

        You are {bot_name}.
        """
        # --- separation to be done in another node.
    elif llm_type == "identity_validator":
        prompt = f"""
        Check if user has provided a valid name, age and email in the message.
        User Message: {info[0]}
        """
        
    elif llm_type == "splitter":
        prompt = f"""
        split relevant portions of text with "---" where appropriate to send multiple messages to the user.
        Example Original Text: "Hey darling! Nice to meet you too! Let me send you a pic of what Im doing right now!"
        Example Rewrite: "Hey darling!---Nice to meet you too!---Let me send you a pic of what Im doing right now!"
        Message to split: {info[0]}
        """

    else:
        prompt = "No prompt found"

    def clean_prompt(prompt: str):
        """Preserve '\n' markers as real line breaks, collapse everything else."""
        # Split on literal \n (not actual newlines)
        parts = prompt.split("\\n")
        # Collapse each part (remove extra spaces and real line breaks)
        cleaned_parts = [" ".join(part.strip().split()) for part in parts]
        # Join with actual newlines
        return "\n".join(cleaned_parts)

    cleaned_prompt = clean_prompt(prompt)

    return cleaned_prompt
