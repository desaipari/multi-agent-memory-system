"""
Centralized client for all calls to the memory service.
Every agent imports from here instead of writing its own requests.post/get calls.
"""

import os
import requests

MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://127.0.0.1:8000")


def write_memory(entity, fact_type, value, agent_id, extraction_type="direct",
                  confidence=None, source_file=None) -> dict:
    payload = {
        "entity": entity,
        "fact_type": fact_type,
        "value": value,
        "agent_id": agent_id,
        "extraction_type": extraction_type,
        "source_file": source_file
    }
    if confidence is not None:
        payload["confidence"] = confidence

    try:
        response = requests.post(f"{MEMORY_SERVICE_URL}/memory/write", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[memory_client] write_memory failed: {e}")
        return None


def read_memory(entity: str, fact_type: str = None) -> list:
    params = {"entity": entity}
    if fact_type:
        params["fact_type"] = fact_type
    try:
        response = requests.get(f"{MEMORY_SERVICE_URL}/memory/read", params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("facts", [])
    except requests.exceptions.RequestException as e:
        print(f"[memory_client] read_memory failed: {e}")
        return []


def get_conflicts() -> dict:
    try:
        response = requests.get(f"{MEMORY_SERVICE_URL}/memory/conflicts", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[memory_client] get_conflicts failed: {e}")
        return None


def resolve_conflict(
    conflict_id: str,
    winning_fact_id: str,
    reason: str,
    confidence: float = None  # kept as param but not sent
) -> dict:
    """
    Call /memory/resolve to apply a conflict resolution decision.
    Person A's endpoint expects: conflict_id, winning_fact_id,
    resolved_by, reason — does NOT expect confidence field.
    """
    payload = {
        "conflict_id": conflict_id,
        "winning_fact_id": winning_fact_id,
        "resolved_by": "coordinator_agent",
        "reason": reason
        # confidence NOT included — Person A's endpoint ignores it
        # and cleaner to not send unexpected fields
    }
    try:
        response = requests.post(
            f"{MEMORY_SERVICE_URL}/memory/resolve",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[memory_client] resolve_conflict failed: {e}")
        return None