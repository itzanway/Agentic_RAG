from rank_bm25 import BM25Okapi
import numpy as np
import uuid

class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.ids = []

    def ingest(self, chunks):
        print("Tokenizing chunks for sparse (BM25) retrieval...")
        self.chunks = chunks
        self.ids = [str(uuid.uuid4()) for _ in chunks]
        tokenized_corpus = [chunk.lower().split(" ") for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=20):
        if not self.bm25:
            return []
            
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # Only return if there's an actual match
                results.append({"text": self.chunks[idx], "score": scores[idx], "id": self.ids[idx]})
        return results