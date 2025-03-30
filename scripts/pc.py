import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)

index_name = "portfolio"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
print("Existing Indexes: ", existing_indexes)

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
overview = "C:/Users/ethan/Documents/General/Portfolio-Bot/data/overview"
solution = "C:/Users/ethan/Documents/General/Portfolio-Bot/data/solution"
overview_proj = ["ov_maibel_app", "ov_mlops", "ov_workAdvisor", "ov_workoutTracker"]
solution_proj = ["sl_maibel_app", "sl_mlops", "sl_workAdvisor", "sl_workoutTracker"]
main_portfolio = "C:/Users/ethan/Documents/General/Portfolio-Bot/data/"

FILE_PATHS = [
    f"{overview}/{proj}.txt" for proj in overview_proj
]
FILE_PATHS.append(main_portfolio)
FILE_PATHS.extend([f"{solution}/{proj}.txt" for proj in solution_proj])

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