import streamlit as st
import os
from dotenv import load_dotenv

# Import your custom modules
from src.ingestion import sliding_window_chunker
from src.retrieval import DenseRetriever, SparseRetriever, Reranker
from src.agent import evaluate_relevance, generate_answer

load_dotenv()

st.set_page_config(page_title="Agentic RAG", page_icon="🤖")
st.title("Agentic RAG System 🤖")
st.markdown("Ask questions based on the ingested documents. The system will autonomously evaluate and retrieve context.")

# 1. Initialize and cache the Vector DBs so they only load once
@st.cache_resource
def initialize_system():
    # Setup mock data (replace with your PDF parsing logic later)
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
    
    return dense_db, sparse_db, reranker

# Load the backend
dense_db, sparse_db, reranker = initialize_system()

# 2. Setup Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if prompt := st.chat_input("Ask a question about the database or revenue..."):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Agentic RAG Logic Loop
    with st.chat_message("assistant"):
        with st.spinner("Agent is searching and reflecting..."):
            max_retries = 3
            current_query = prompt
            final_answer = "System failed to find relevant information after max retries."
            
            for attempt in range(max_retries):
                # Retrieve and Rerank
                dense_results = dense_db.search(current_query, top_k=5)
                sparse_results = sparse_db.search(current_query, top_k=5)
                best_chunks = reranker.fuse_and_rerank(current_query, dense_results, sparse_results, top_k=2)
                
                # Agentic Evaluation
                evaluation = evaluate_relevance(current_query, best_chunks)
                
                if evaluation["is_relevant"]:
                    final_answer = generate_answer(prompt, best_chunks)
                    break # Break the loop if we found a good answer!
                else:
                    current_query = evaluation["new_query"]
            
            # Display final answer
            st.markdown(final_answer)
            
    # Save assistant answer to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})