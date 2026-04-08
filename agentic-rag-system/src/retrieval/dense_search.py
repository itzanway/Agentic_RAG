from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class DenseRetriever:
    def __init__(self, collection_name="rag_collection"):
        # Using a fast, local embedding model
        self.encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        # Initialize Qdrant locally in-memory (change path for persistent storage)
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        
        # Setup Qdrant collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )

    def ingest(self, chunks):
        print(f"Embedding {len(chunks)} chunks for dense retrieval...")
        points = []
        for chunk in chunks:
            vector = self.encoder.encode(chunk).tolist()
            point_id = str(uuid.uuid4())
            points.append(PointStruct(id=point_id, vector=vector, payload={"text": chunk}))
        
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query, top_k=20):
        query_vector = self.encoder.encode(query).tolist()
        
        # FIX: Use query_points instead of search for the new Qdrant API
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector, # Note: parameter is now 'query' instead of 'query_vector'
            limit=top_k
        )
        
        # FIX: Loop through response.points instead of results directly
        return [{"text": res.payload["text"], "score": res.score, "id": res.id} for res in response.points]