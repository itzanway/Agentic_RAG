# 🔍 Agentic RAG System
**Domain-specific QA with Hybrid Search, Cross-Encoder Reranking & Self-Reflective Query Rewriting**

---

## Architecture Overview

```
                         ┌─────────────────────────────────────────────────┐
                         │              INGESTION PIPELINE                  │
                         │                                                   │
  Raw Document ──────▶  │  Parser  ──▶  Chunker  ──▶  Embedder  ──▶  Store │
  (PDF/DOCX/TXT)         │  (table-    (semantic/     (BGE local)   ChromaDB│
                         │   aware)    hierarchical)               + BM25   │
                         └─────────────────────────────────────────────────┘
                                                 │
                         ┌─────────────────────────────────────────────────┐
                         │              QUERY PIPELINE                      │
                         │                                                   │
  Question ──────────▶  │  Dense Search ──┐                                │
                         │  (ChromaDB)     ├──▶ RRF Fusion ──▶ Reranker   │
                         │  Sparse Search ─┘    (weighted)   (cross-enc)   │
                         │  (BM25)                                          │
                         └─────────────────────────────────────────────────┘
                                                 │
                         ┌─────────────────────────────────────────────────┐
                         │              AGENTIC LOOP (Self-RAG/CRAG)        │
                         │                                                   │
                         │  Retrieve ──▶ Evaluate ──▶ Relevant? ──▶ Answer │
                         │                  │                               │
                         │                  └── No ──▶ Rewrite ──▶ Retry  │
                         │                                                   │
                         └─────────────────────────────────────────────────┘
```

## Key Engineering Decisions

### 1. Table-Aware Parsing
Standard PDF parsers (PyPDF2, pdfminer) flatten tables into garbage text.
This system uses `pdfplumber` to:
- Extract table bounding boxes first
- Convert tables to structured Markdown
- Crop table regions before extracting prose text
- Preserve page order across both elements

### 2. Semantic Chunking
Instead of splitting at character count boundaries:
1. Split document into sentences
2. Embed each sentence using a local BGE model
3. Compute cosine similarity between consecutive sentences
4. Split where similarity drops below threshold (topic boundary)
5. Merge groups that are too short

Result: chunks that represent coherent topics, not arbitrary text windows.

### 3. Hybrid Retrieval
```
Dense (BGE embeddings)   →  Conceptual matches, paraphrases
Sparse (BM25)            →  Exact keywords: AAPL, 10-K, Section 7(a)
RRF Fusion               →  Merge ranked lists without tuning scores
Cross-Encoder Reranking  →  Precise relevance: attends to both query + doc
```
RRF score: `Σ α/(k+rank_dense) + (1-α)/(k+rank_sparse)`

### 4. Self-Reflective Agentic Loop
Implements concepts from **Self-RAG** (Asai et al., 2023) and **CRAG** (Yan et al., 2024):
- Evaluate retrieved context relevance *before* generating
- Trigger query rewriting if context is off-target
- Score answer faithfulness *after* generating
- Fallback with disclaimer if all rewrites fail

### 5. Quantitative Evaluation (RAGAS)
Four metrics computed by LLM-as-judge:
- **Context Precision**: `|relevant_retrieved| / |total_retrieved|`
- **Context Recall**: `|info_needed_covered| / |info_needed|`
- **Answer Faithfulness**: fraction of answer claims supported by context
- **Answer Relevancy**: does the answer address the actual question?

## Setup

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
```

### 3. Start the API server
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Start the UI
```bash
streamlit run frontend/app.py
```
Open http://localhost:8501

## Usage

### Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue in Q3?"}'
```

### Run evaluation
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 20, "generate_dataset": true}'
```

## File Structure

```
agentic-rag/
├── config/
│   └── settings.py           # Centralised config from .env
├── src/
│   ├── ingestion/
│   │   ├── parser.py         # PDF/DOCX/TXT parsing with table extraction
│   │   ├── chunker.py        # Semantic + Hierarchical + Fixed chunking
│   │   └── pipeline.py       # Ingestion orchestrator
│   ├── retrieval/
│   │   ├── embedder.py       # HuggingFace sentence-transformers wrapper
│   │   ├── vector_store.py   # ChromaDB dense store
│   │   ├── bm25_store.py     # BM25 sparse index
│   │   └── hybrid_retriever.py  # RRF fusion + cross-encoder reranking
│   ├── agents/
│   │   └── rag_agent.py      # Agentic self-reflection loop
│   ├── evaluation/
│   │   └── evaluator.py      # RAGAS metrics + LLM-as-judge
│   └── api/
│       └── main.py           # FastAPI endpoints
├── frontend/
│   └── app.py                # Streamlit UI
├── data/
│   ├── chroma_db/            # Persisted vector embeddings
│   ├── bm25_index.pkl        # Persisted BM25 index
│   └── eval_dataset.json     # Synthetic evaluation Q&A pairs
├── .env.example
├── requirements.txt
└── README.md
```

## Models Used (All Local, No External API)

| Component | Model | Size |
|-----------|-------|------|
| Embedder | `BAAI/bge-small-en-v1.5` | ~130MB |
| Reranker | `BAAI/bge-reranker-base` | ~280MB |
| LLM | Claude (via Anthropic API) | — |

Swap in any HuggingFace model by changing `.env`.

## Extending This Project

- **Add Cohere reranking**: Replace `CrossEncoder` with `cohere.Client().rerank()`
- **Add Milvus**: Swap `VectorStore` implementation — same interface
- **Add streaming**: FastAPI `StreamingResponse` + Streamlit `st.write_stream`
- **Add multi-modal**: Parse images from PDFs using `pdfplumber` + OCR
- **Add citations**: Track sentence-level sources during generation

## References

- Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023)
- CRAG: Corrective Retrieval Augmented Generation (Yan et al., 2024)
- RAGAS: Automated Evaluation of RAG Pipelines (Es et al., 2023)
- BGE: BAAI General Embedding (Zhang et al., 2023)
- Reciprocal Rank Fusion (Cormack et al., 2009)
