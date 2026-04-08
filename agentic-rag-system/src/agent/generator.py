import os
from groq import Groq
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Get the GROQ API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment or .env file.")

# Initialize Groq client with the API key
client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query, context_chunks):
    context_text = "\n\n".join([f"Source {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""
    You are an expert assistant. Answer the user's query using ONLY the provided context.
    If the answer is not contained in the context, say "I cannot answer this based on the provided documents."
    
    Context:
    {context_text}
    
    Query: {query}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Updated to a supported model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content