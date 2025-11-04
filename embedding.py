import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = "AIzaSyCSKNnQWMCFCWNDKG3KrNQuek8UTIy_D9o"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
result = embeddings.embed_query("Hello, world!")
print(result)
print(len(result))