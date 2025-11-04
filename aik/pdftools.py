from langchain_community.document_loaders import PyPDFLoader

file_path = "MACHINE LEARNING(R17A0534).pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
docs[0]
print(docs[0])