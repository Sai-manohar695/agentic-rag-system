import os
from dotenv import load_dotenv
load_dotenv()

# Test Groq
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user",
               "content": "Say: Groq connected successfully"}]
)
print(response.choices[0].message.content)

# Test Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
indexes = pc.list_indexes()
print(f"Pinecone connected — indexes: {indexes}")