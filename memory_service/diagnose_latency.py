"""
Diagnose where the latency is coming from.
Run while FastAPI is running.
"""
import time
import requests
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

BASE = "http://localhost:8000"
DB_URL = os.getenv("DATABASE_URL")

print("=== LATENCY DIAGNOSIS ===\n")

# Test 1: Raw HTTP round trip to FastAPI health
print("Test 1: Raw HTTP to /memory/health (no DB, no model)")
times = []
for _ in range(5):
    start = time.time()
    requests.get(f"{BASE}/memory/health", timeout=10)
    times.append((time.time() - start) * 1000)
print(f"  Mean: {sum(times)/len(times):.1f} ms")
print(f"  If this is > 500ms, the problem is FastAPI/network overhead")
print(f"  If this is < 100ms, FastAPI is fine — problem is DB or model\n")

# Test 2: Raw PostgreSQL query (bypass FastAPI entirely)
print("Test 2: Direct PostgreSQL query (bypass FastAPI)")
try:
    conn = psycopg2.connect(DB_URL)
    times = []
    for _ in range(5):
        start = time.time()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM facts")
        cur.fetchone()
        cur.close()
        times.append((time.time() - start) * 1000)
    conn.close()
    print(f"  Mean: {sum(times)/len(times):.1f} ms")
    print(f"  If this is < 10ms, PostgreSQL is fine")
    print(f"  If this is > 100ms, PostgreSQL connection is the problem\n")
except Exception as e:
    print(f"  Error: {e}\n")

# Test 3: Direct Qdrant query (bypass FastAPI)
print("Test 3: Direct Qdrant health check")
try:
    times = []
    for _ in range(5):
        start = time.time()
        requests.get("http://localhost:6333/", timeout=10)
        times.append((time.time() - start) * 1000)
    print(f"  Mean: {sum(times)/len(times):.1f} ms")
    print(f"  If this is > 500ms, Qdrant is the bottleneck\n")
except Exception as e:
    print(f"  Error: {e}\n")

# Test 4: Sentence transformer encode time
print("Test 4: Sentence transformer encode time")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # First encode (cold)
    start = time.time()
    model.encode("priority 2-High", show_progress_bar=False)
    cold = (time.time() - start) * 1000
    # Second encode (warm)
    start = time.time()
    model.encode("priority 2-High", show_progress_bar=False)
    warm = (time.time() - start) * 1000
    print(f"  Cold encode: {cold:.1f} ms")
    print(f"  Warm encode: {warm:.1f} ms")
    print(f"  If warm > 500ms, model is the bottleneck\n")
except Exception as e:
    print(f"  Error: {e}\n")

print("=== DIAGNOSIS COMPLETE ===")
print("Share these numbers to identify the exact bottleneck.")