from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyCSKNnQWMCFCWNDKG3KrNQuek8UTIy_D9o"

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash") 
chain = prompt_template | llm
a = input("Enter topic: ")
response = chain.invoke({"topic": a})

print(response.content)