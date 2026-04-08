import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        model="gpt-4o-mini", # Use a fast, cheap model for routing
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    output = response.choices[0].message.content.strip().split('\n')
    is_relevant = output[0].split(":")[1].strip()
    new_query = output[1].split(":")[1].strip() if len(output) > 1 else ""

    return {"is_relevant": is_relevant == "YES", "new_query": new_query}