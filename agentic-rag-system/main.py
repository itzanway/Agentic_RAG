from src.ingestion.chunker import sliding_window_chunker
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.sparse_search import SparseRetriever
from src.retrieval.reranker import Reranker
from src.agent.router import evaluate_relevance
from src.agent.generator import generate_answer

def main():
    # 1. Setup mock data (replace with parsing your PDFs)
    print("--- Phase 1: Ingestion ---")
    mock_text = [
        "The company revenue in Q3 2023 was $4.5 million, up 12% from Q2.",
        "Server architecture relies on AWS EC2 instances and an RDS PostgreSQL database.",
        "Employee policy states that vacation days do not roll over into the next calendar year."
    ]
    chunks = sliding_window_chunker(mock_text, chunk_size=20)
    
    # 2. Initialize and populate retrieval systems
    print("\n--- Phase 2: Indexing ---")
    dense_db = DenseRetriever()
    sparse_db = SparseRetriever()
    reranker = Reranker()
    
    dense_db.ingest(chunks)
    sparse_db.ingest(chunks)
    
    # 3. The Agentic Query Loop
    user_query = "What is our database hosted on?"
    print(f"\n--- Phase 3: Querying -> '{user_query}' ---")
    
    max_retries = 3
    current_query = user_query
    
    for attempt in range(max_retries):
        print(f"\nAttempt {attempt + 1}: Searching for '{current_query}'")
        
        # Hybrid Search
        dense_results = dense_db.search(current_query, top_k=5)
        sparse_results = sparse_db.search(current_query, top_k=5)
        
        # Rerank
        best_chunks = reranker.fuse_and_rerank(current_query, dense_results, sparse_results, top_k=2)
        
        # Self-Reflection
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