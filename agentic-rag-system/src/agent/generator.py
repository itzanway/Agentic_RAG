from openai import OpenAI

client = OpenAI() # Assumes api key is in env

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
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return response.choices[0].message.content