from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = PyPDFLoader("MACHINE LEARNING(R17A0534).pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")
print(chunks[0].page_content[:500])  
