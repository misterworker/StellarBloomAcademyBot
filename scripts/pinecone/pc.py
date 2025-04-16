import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)

index_name = "stellarbloom"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
print("Existing Indexes: ", existing_indexes)

# if index_name in existing_indexes:
#     pc.delete_index(
#         name=index_name
#     )
# pc.delete_collection # Delete entire Portfolio collection

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait until the index is ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)
stellarbloom = "C:/Users/ethan/Documents/Work/Stellar Bloom Academy/bot/data/stellarbloom"
overview_proj = ["celestial_guides", "origin", "philosophy_of_light", "starbourne_journey", "the_bloom_rite"]
# main_portfolio = "C:/Users/ethan/Documents/General/Portfolio-Bot/data/"

FILE_PATHS = [
    f"{stellarbloom}/{proj}.txt" for proj in overview_proj
]

print("File Paths: ", FILE_PATHS[:3])

# Initialize Embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create or Use Existing Index
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

from langchain_core.documents import Document
from uuid import uuid4
documents = []

# Add Files to Pinecone
for file_path in FILE_PATHS:
    with open(file_path, "r") as file:
        text = file.read()
        document = Document(
            page_content=text,
            # metadata={"source": "tweet"}
            )
        documents.append(document)
        print(f"Added: {file_path}")

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)