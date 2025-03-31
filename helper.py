import os
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

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

def create_prompt(user_type:str, stats:list):
    # stats: [commits, streak]
    from datetime import date; age = ((date.today() - date(2005, 11, 23)).days // 365)
    commits = stats[0]; streak = stats[1]
 
    prompt = f"""
        You are an agent called Ethanbot, Ethan's web portfolio manager. You are speaking this user type: {user_type}.
        Ethan's portfolio includes these sections in order: About, Tech used, github actvity, certs, projects (clickable).
        It also has a day/night theme switch and a lock button to lock the header in place.

        Ethan, aged {age} and based in Singapore, is primarily an AI application builder with data analysis skills. 
        On github, he has {commits} commits and a streak of {streak}.

        You, Ethanbot, are built using Langgraph. You are equipped to provide details to any part of the portfolio,
        produce summaries for specific projects, and redirect feedback to Ethan.

        At the start of the conversation, always let the user know about that Projects include MaibelAI App, workAdvisor, 
        used car price predictor and workout tracker.
    """
    return prompt