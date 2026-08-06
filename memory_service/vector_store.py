from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
import uuid

# Connect to locally running Qdrant
client = QdrantClient(host="localhost", port=6333)

# Load sentence transformer model
# all-MiniLM-L6-v2 is small, fast, and works well for 
# short factual sentences like IT incident facts
print("Loading sentence transformer model...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

COLLECTION_NAME = "incident_facts"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

def ensure_collection_exists():
    """
    Create the Qdrant collection if it does not exist yet.
    Called once on startup.
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        print(f"Qdrant collection already exists: {COLLECTION_NAME}")

def embed_fact(fact_type: str, value: str) -> list:
    """
    Convert a fact into a vector embedding.
    We embed the combination of fact_type and value
    so similar facts (same type, similar values) end up close together.
    
    Example: "priority 2-High" and "priority High" will be close
    Example: "priority 2-High" and "state Resolved" will be far apart
    """
    text = f"{fact_type} {value}"
    embedding = encoder.encode(text)
    return embedding.tolist()

def store_embedding(
    fact_id: str,
    entity_hash: str,
    fact_type: str,
    value: str,
    agent_id: str
):
    """
    Store a fact's embedding in Qdrant.
    
    We store entity_hash and fact_type as payload
    so we can filter by them during similarity search.
    The actual sensitive value is NOT stored in Qdrant —
    only the embedding and metadata for search.
    """
    embedding = embed_fact(fact_type, value)
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "fact_id": fact_id,
                    "entity_hash": entity_hash,
                    "fact_type": fact_type,
                    "agent_id": agent_id
                    # value NOT stored — only the hash is in PostgreSQL
                }
            )
        ]
    )

def find_similar_facts(
    entity_hash: str,
    fact_type: str,
    value: str,
    threshold: float = 0.70,
    limit: int = 5
) -> list:
    """
    Find semantically similar facts using query_points.
    Works with qdrant-client 1.7+
    Threshold 0.70 chosen based on diagnostic:
    - Same fact type, different value = ~0.76 similarity
    - Different fact type = ~0.22 similarity
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    embedding = embed_fact(fact_type, value)

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="entity_hash",
                        match=MatchValue(value=entity_hash)
                    ),
                    FieldCondition(
                        key="fact_type",
                        match=MatchValue(value=fact_type)
                    )
                ]
            ),
            limit=limit,
            score_threshold=threshold
        ).points

        return [
            {
                "fact_id": r.payload["fact_id"],
                "agent_id": r.payload["agent_id"],
                "similarity_score": r.score
            }
            for r in results
        ]

    except Exception as e:
        print(f"Qdrant query error: {e}")
        return []