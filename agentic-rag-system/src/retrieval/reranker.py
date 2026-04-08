from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        # Cross-encoders are highly accurate but computationally heavy, so we only run them on the top few results
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def fuse_and_rerank(self, query, dense_results, sparse_results, top_k=5):
        # 1. Deduplicate results using a dictionary keyed by the text chunk
        unique_docs = {}
        for res in dense_results + sparse_results:
            unique_docs[res["text"]] = res

        doc_texts = list(unique_docs.keys())
        if not doc_texts:
            return []

        # 2. Score query-document pairs using the cross-encoder
        pairs = [[query, text] for text in doc_texts]
        rerank_scores = self.model.predict(pairs)

        # 3. Sort by the new cross-encoder score
        scored_docs = list(zip(doc_texts, rerank_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return the top K texts
        return [doc[0] for doc in scored_docs[:top_k]]