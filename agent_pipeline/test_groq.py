from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# openai/gpt-oss-20b is the open source replacement
# for the deprecated llama-3.1-8b-instant
# It is free on Groq's developer tier — not a paid OpenAI model
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b"
)

response = llm.invoke("Say hello and confirm you are working.")
print(response.content)
