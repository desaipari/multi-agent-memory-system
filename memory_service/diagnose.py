"""
Self-contained diagnostic script.
Run while FastAPI is running on port 8000.

Run from memory_service folder:
python diagnose.py
"""

import hashlib
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# ── Setup — define everything directly, no imports from other files ──
MEMORY_URL = "http://localhost:8000"
COLLECTION_NAME = "incident_facts"

# Connect to Qdrant directly
qdrant = QdrantClient(host="localhost", port=6333)

# Load encoder directly
print("Loading sentence transformer...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoder ready.\n")

def hash_value(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()

# ── Step 1: Check PostgreSQL via API ──────────────────────────
print("=== STEP 1: PostgreSQL Facts ===")
try:
    response = requests.get(f"{MEMORY_URL}/memory/all", timeout=5)
    data = response.json()
    facts = data.get("facts", [])
    print(f"Total facts in database: {len(facts)}")
    for f in facts:
        print(f"  entity={f['entity']} | fact_type={f['fact_type']} | "
              f"value={f['value']} | agent={f['agent_id']}")
except Exception as e:
    print(f"Error connecting to FastAPI: {e}")
    print("Make sure uvicorn main:app --reload --port 8000 is running")

# ── Step 2: Check Qdrant points ───────────────────────────────
print("\n=== STEP 2: Qdrant Points ===")
try:
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"Points in Qdrant: {info.points_count}")

    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=20,
        with_payload=True,
        with_vectors=False
    )
    if points:
        for point in points:
            print(f"  fact_id={point.payload.get('fact_id', 'N/A')[:8]}... | "
                  f"entity_hash={point.payload.get('entity_hash', '')[:16]}... | "
                  f"fact_type={point.payload.get('fact_type')} | "
                  f"agent={point.payload.get('agent_id')}")
    else:
        print("  No points found in Qdrant")
except Exception as e:
    print(f"Qdrant error: {e}")
    print("Make sure Qdrant is running on port 6333")

# ── Step 3: Manual similarity search ─────────────────────────
print("\n=== STEP 3: Manual Similarity Search ===")

entity = "INC0000001"
fact_type = "priority"
search_value = "3-Medium"

entity_hash = hash_value(entity)
print(f"Entity: {entity}")
print(f"Entity hash: {entity_hash[:16]}...")
print(f"Searching for fact_type={fact_type}, value={search_value}")

query_embedding = encoder.encode(
    f"{fact_type} {search_value}"
).tolist()

try:
    print("\nSearch with NO score threshold (show all results):")
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
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
        limit=10
    ).points

    if results:
        print(f"Found {len(results)} similar facts:")
        for r in results:
            print(f"  fact_id={r.payload.get('fact_id', 'N/A')[:8]}... | "
                  f"score={r.score:.4f} | "
                  f"agent={r.payload.get('agent_id')}")
        print(f"\nHighest similarity score: {results[0].score:.4f}")
        print(f"Your threshold is 0.70")
        if results[0].score >= 0.70:
            print("Score is ABOVE threshold — contradiction SHOULD be detected")
        else:
            print("Score is BELOW threshold — this is why contradiction was missed")
            print("You need to lower the threshold further")
    else:
        print("No results found at all")
        print("This means either:")
        print("  1. Qdrant collection is empty")
        print("  2. Entity hash does not match stored facts")
        print("  3. fact_type filter is excluding everything")

except Exception as e:
    print(f"Search error: {e}")

# ── Step 4: Hash consistency check ───────────────────────────
print("\n=== STEP 4: Hash Consistency Check ===")
test = "INC0000001"
h = hash_value(test)
print(f"hash('{test}') = {h[:16]}...")
print(f"Full hash: {h}")

# ── Step 5: Test what threshold actually works ────────────────
print("\n=== STEP 5: Threshold Test ===")
print("Testing similarity between priority values:")

pairs = [
    ("priority 2-High", "priority 3-Medium"),
    ("priority 2-High", "priority 2-High"),
    ("priority 2-High", "state Resolved"),
    ("priority 2-High", "priority 1-Critical"),
]

for text_a, text_b in pairs:
    vec_a = encoder.encode(text_a).tolist()
    vec_b = encoder.encode(text_b).tolist()

    import numpy as np
    a = np.array(vec_a)
    b = np.array(vec_b)
    cosine_sim = float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )
    print(f"  '{text_a}' vs '{text_b}': {cosine_sim:.4f}")

print("\nDiagnosis complete.")