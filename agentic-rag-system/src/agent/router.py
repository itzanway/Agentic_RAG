import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_relevance(query, context_chunks):
    """
    Evaluates if the context contains the answer. If not, generates a better search query.
    """
    context_text = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""
    You are a strict grading assistant.
    User Query: {query}
    
    Retrieved Context:
    {context_text}
    
    Does the retrieved context contain enough information to accurately answer the query?
    Respond in the following format exactly:
    RELEVANT: [YES or NO]
    NEW_QUERY: [If NO, write a modified, better search query to find the missing information. If YES, leave blank]
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Updated to a supported model
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    output = response.choices[0].message.content.strip().split('\n')
    is_relevant = "YES" in output[0].upper()
    new_query = output[1].split(":")[1].strip() if len(output) > 1 and ":" in output[1] else ""

    return {"is_relevant": is_relevant, "new_query": new_query}