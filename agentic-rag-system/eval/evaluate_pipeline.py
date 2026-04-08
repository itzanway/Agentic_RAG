import os
import json
from groq import Groq
from dotenv import load_dotenv

# Import your pipeline modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingestion import sliding_window_chunker
from src.retrieval import DenseRetriever, SparseRetriever, Reranker
from src.agent import generate_answer

load_dotenv()

# UPDATE 1: Swap OpenAI for Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_with_llm(query, context, final_answer):
    """
    Uses an LLM as a judge to score the RAG outputs from 1 to 5.
    """
    context_text = "\n\n".join(context)
    
    prompt = f"""
    You are an impartial grading expert evaluating a Retrieval-Augmented Generation (RAG) system.
    
    User Query: {query}
    Retrieved Context: {context_text}
    System Answer: {final_answer}
    
    Evaluate the System Answer based on the following two metrics:
    
    1. Faithfulness (1-5): Is the System Answer strictly based on the Retrieved Context? (1 = Completely hallucinated, 5 = Perfectly grounded in context).
    2. Answer Relevance (1-5): Does the System Answer directly and fully address the User Query? (1 = Completely irrelevant, 5 = Perfect, direct answer).
    
    Respond ONLY with a valid JSON object in this exact format:
    {{"faithfulness": score, "relevance": score, "reasoning": "Brief explanation"}}
    """

    # UPDATE 2: Use a free, fast open-source model like Llama 3
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Updated to a supported model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={ "type": "json_object" }
    )
    return json.loads(response.choices[0].message.content)

def main():
    print("--- Booting up Evaluation Pipeline ---")
    
    # 1. Setup mock data & index it (In production, load your real Vector DB)
    mock_text = [
        "The company revenue in Q3 2023 was $4.5 million, up 12% from Q2.",
        "Server architecture relies on AWS EC2 instances and an RDS PostgreSQL database.",
        "Employee policy states that vacation days do not roll over into the next calendar year."
    ]
    chunks = sliding_window_chunker(mock_text, chunk_size=20)
    
    dense_db = DenseRetriever()
    sparse_db = SparseRetriever()
    reranker = Reranker()
    
    dense_db.ingest(chunks)
    sparse_db.ingest(chunks)

    # 2. Define a "Golden Dataset" (Test Cases)
    test_cases = [
        {"query": "How much did revenue grow in Q3 2023?"},
        {"query": "Where is the database hosted?"},
        {"query": "Can I save my vacation days for next year?"}
    ]

    total_faithfulness = 0
    total_relevance = 0

    print("\n--- Running Evaluations ---")
    
    # 3. Run the pipeline for each test case and score it
    for i, test in enumerate(test_cases):
        query = test["query"]
        print(f"\nTest {i+1}: '{query}'")
        
        # Run Retrieval
        dense_results = dense_db.search(query, top_k=5)
        sparse_results = sparse_db.search(query, top_k=5)
        best_chunks = reranker.fuse_and_rerank(query, dense_results, sparse_results, top_k=2)
        
        # Run Generation
        final_answer = generate_answer(query, best_chunks)
        
        # Run Evaluation
        eval_result = evaluate_with_llm(query, best_chunks, final_answer)
        
        print(f"System Answer: {final_answer}")
        print(f"Scores -> Faithfulness: {eval_result['faithfulness']}/5 | Relevance: {eval_result['relevance']}/5")
        print(f"Judge Reasoning: {eval_result['reasoning']}")
        
        total_faithfulness += eval_result["faithfulness"]
        total_relevance += eval_result["relevance"]

    # 4. Calculate final pipeline metrics
    avg_faithfulness = total_faithfulness / len(test_cases)
    avg_relevance = total_relevance / len(test_cases)

    print("\n================ FINAL METRICS ================")
    print(f"Average Faithfulness:  {avg_faithfulness:.1f} / 5.0")
    print(f"Average Relevance:     {avg_relevance:.1f} / 5.0")
    if avg_faithfulness < 4.0 or avg_relevance < 4.0:
        print("Status: NEEDS IMPROVEMENT (Tweak retrieval parameters or chunk sizes)")
    else:
        print("Status: PRODUCTION READY")

if __name__ == "__main__":
    main()