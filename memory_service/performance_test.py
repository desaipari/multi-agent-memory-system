"""
Performance test — measures latency under concurrent load.
Run while FastAPI is running on port 8000.

Usage:
    python performance_test.py

Tests:
    1. Sequential writes — baseline latency per write
    2. Concurrent writes — 50 simultaneous writes
    3. Read latency — fetch all facts
    4. Conflict detection latency — write that triggers detection
"""

import requests
import time
import concurrent.futures
import statistics
import uuid

BASE = "http://127.0.0.1:8000"

def write_fact(incident_id: str, agent_id: str,
               fact_type: str, value: str) -> float:
    """Write one fact and return latency in milliseconds."""
    start = time.time()
    try:
        response = requests.post(
            f"{BASE}/memory/write",
            json={
                "entity": incident_id,
                "fact_type": fact_type,
                "value": value,
                "agent_id": agent_id,
                "extraction_type": "direct",
                "source_file": "perf_test"
            },
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        if response.status_code == 200:
            return elapsed
        else:
            print(f"  Error {response.status_code}: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None

print("=" * 60)
print("PERFORMANCE TEST — Multi-Agent Memory Service")
print("=" * 60)

# ── Test 1: Sequential write baseline ─────────────────────────
print("\nTest 1: Sequential writes (10 writes)")
latencies = []
for i in range(10):
    inc_id = f"PERF_{uuid.uuid4().hex[:8]}"
    lat = write_fact(inc_id, "intake_agent", "priority", "2-High")
    if lat:
        latencies.append(lat)

if latencies:
    print(f"  Mean latency:   {statistics.mean(latencies):.1f} ms")
    print(f"  Median latency: {statistics.median(latencies):.1f} ms")
    print(f"  Max latency:    {max(latencies):.1f} ms")
    print(f"  Min latency:    {min(latencies):.1f} ms")

# ── Test 2: Concurrent writes ──────────────────────────────────
print("\nTest 2: Concurrent writes (50 simultaneous)")

def write_concurrent(i):
    inc_id = f"CONC_{uuid.uuid4().hex[:8]}"
    return write_fact(inc_id, "intake_agent", "state", "New")

start_total = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(write_concurrent, i) for i in range(50)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

total_elapsed = (time.time() - start_total) * 1000
concurrent_latencies = [r for r in results if r is not None]
errors = len(results) - len(concurrent_latencies)

print(f"  Total time for 50 writes: {total_elapsed:.1f} ms")
print(f"  Successful writes: {len(concurrent_latencies)}/50")
print(f"  Errors: {errors}")
if concurrent_latencies:
    print(f"  Mean per-write latency: {statistics.mean(concurrent_latencies):.1f} ms")
    print(f"  P95 latency: {sorted(concurrent_latencies)[int(len(concurrent_latencies)*0.95)]:.1f} ms")

# ── Test 3: Read latency ───────────────────────────────────────
print("\nTest 3: Read all facts latency (5 reads)")
read_latencies = []
for _ in range(5):
    start = time.time()
    response = requests.get(f"{BASE}/memory/all", timeout=10)
    elapsed = (time.time() - start) * 1000
    if response.status_code == 200:
        total_facts = response.json().get("total", 0)
        read_latencies.append(elapsed)

if read_latencies:
    print(f"  Facts in database: {total_facts}")
    print(f"  Mean read latency: {statistics.mean(read_latencies):.1f} ms")
    print(f"  Max read latency:  {max(read_latencies):.1f} ms")

# ── Test 4: Conflict detection latency ────────────────────────
print("\nTest 4: Conflict detection latency")
inc_id = f"CONFLICT_{uuid.uuid4().hex[:8]}"

# Write first fact
lat1 = write_fact(inc_id, "intake_agent", "priority", "1-Critical")
print(f"  First write (no conflict expected): {lat1:.1f} ms")

# Write conflicting fact — triggers detection
lat2 = write_fact(inc_id, "billing_agent", "priority", "3-Medium")
print(f"  Second write (conflict detection): {lat2:.1f} ms")
overhead = (lat2 or 0) - (lat1 or 0)
print(f"  Contradiction detection overhead: {overhead:.1f} ms")

print("\n" + "=" * 60)
print("Performance test complete.")
print("Run this test and record results for your evaluation report.")
print("=" * 60)