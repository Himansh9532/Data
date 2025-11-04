from dotenv import load_dotenv
import os
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np

# Load environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyCSKNnQWMCFCWNDKG3KrNQuek8UTIy_D9o"

# Set up embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
sample_embedding = embeddings.embed_query("Hello, world!")
embedding_dim = len(sample_embedding)

# Create FAISS index and vector store
index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Add a document
doc = Document(page_content="Hello, world!")
vector_store.add_documents([doc])

# Print stored docs
print("\n Stored documents:")
for doc_id, doc in vector_store.docstore._dict.items():
    print(f"ID: {doc_id} | Content: {doc.page_content}")

# Print stored vectors
print("\n Stored FAISS vector (first 10 values):")
vec = vector_store.index.reconstruct(0)
print(np.round(vec[:10], 4))

# Perform similarity search
query = "Hello there!"
results_with_score = vector_store.similarity_search_with_score(query)
print("\n Similarity Search Result:")
for doc, score in results_with_score:
    print(f"Match: {doc.page_content} | Score: {score}")
