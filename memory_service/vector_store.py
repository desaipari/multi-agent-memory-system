from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
import uuid
import os
import numpy as np

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

COLLECTION_NAME = "incident_facts"
VECTOR_SIZE = 384

# ── Model cache — load once, reuse forever ────────────────────
# This is the critical fix. The model must be loaded once at
# module import time and kept in memory. If it reloads per
# request, each write takes 1-2 seconds just for model loading.
_encoder = None

def get_encoder() -> SentenceTransformer:
    """
    Returns the cached sentence transformer model.
    Loads it once on first call, then returns the cached instance.
    Thread-safe for read-only inference.
    """
    global _encoder
    if _encoder is None:
        print("Loading sentence transformer model (first time only)...")
        _encoder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"  # explicit CPU, avoids CUDA detection overhead
        )
        print("Model loaded and cached.")
    return _encoder

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
    """Convert a fact into a vector embedding."""
    text = f"{fact_type} {value}"
    embedding = get_encoder().encode(text, show_progress_bar=False)
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