from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Load the PDF
loader = PyPDFLoader("MACHINE LEARNING(R17A0534).pdf")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embedding_model = OpenAIEmbeddings()  # or use HuggingFaceEmbeddings
vectorstore = FAISS.from_documents(docs, embedding_model)


vectorstore.save_local("ml_pdf_index")
