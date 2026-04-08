import os
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY) right at the start
load_dotenv()

# Notice how clean these imports are now thanks to your __init__.py files!
from src.ingestion import sliding_window_chunker
from src.retrieval import DenseRetriever, SparseRetriever, Reranker
from src.agent import evaluate_relevance, generate_answer

def main():
    # 1. Setup mock data (replace with parsing your PDFs)
    print("--- Phase 1: Ingestion ---")
    mock_text = [
        "The company revenue in Q3 2023 was $4.5 million, up 12% from Q2.",
        "Server architecture relies on AWS EC2 instances and an RDS PostgreSQL database.",
        "Employee policy states that vacation days do not roll over into the next calendar year."
    ]
    
    # Chunk the text
    chunks = sliding_window_chunker(mock_text, chunk_size=20)
    
    # 2. Initialize and populate retrieval systems
    print("\n--- Phase 2: Indexing ---")
    dense_db = DenseRetriever()
    sparse_db = SparseRetriever()
    reranker = Reranker()
    
    # Ingest chunks into both Dense (Vector) and Sparse (BM25) systems
    dense_db.ingest(chunks)
    sparse_db.ingest(chunks)
    
    # 3. The Agentic Query Loop
    user_query = "What is our database hosted on?"
    print(f"\n--- Phase 3: Querying -> '{user_query}' ---")
    
    max_retries = 3
    current_query = user_query
    
    for attempt in range(max_retries):
        print(f"\nAttempt {attempt + 1}: Searching for '{current_query}'")
        
        # Run parallel Hybrid Search
        dense_results = dense_db.search(current_query, top_k=5)
        sparse_results = sparse_db.search(current_query, top_k=5)
        
        # Cross-Encoder Rerank
        best_chunks = reranker.fuse_and_rerank(current_query, dense_results, sparse_results, top_k=2)
        
        # Agentic Routing / Self-Reflection
        print("Agent evaluating context relevance...")
        evaluation = evaluate_relevance(current_query, best_chunks)
        
        if evaluation["is_relevant"]:
            print("Context is relevant! Generating final answer...")
            final_answer = generate_answer(user_query, best_chunks)
            print("\n================ FINAL ANSWER ================")
            print(final_answer)
            return
        else:
            print(f"Context irrelevant. Agent rewriting query to: {evaluation['new_query']}")
            current_query = evaluation["new_query"]
            
    print("\nSystem failed to find relevant information after max retries.")

if __name__ == "__main__":
    main()