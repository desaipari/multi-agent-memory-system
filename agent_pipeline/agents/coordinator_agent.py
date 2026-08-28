from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_client import get_conflicts, resolve_conflict

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

AGENT_PRIORITY = {
    "intake_agent": 4,
    "delivery_agent": 3,
    "coordinator_agent": 2,
    "billing_agent": 1
}


def process_flagged_conflicts() -> dict:
    """
    Main Coordinator function:
    1. Fetch all flagged conflicts from the memory service
    2. For each, score both sides using confidence + agent trust priority
    3. Call resolve_conflict() with the decision
    4. Return a summary
    """
    print("\n[COORDINATOR] Checking for flagged conflicts...")

    conflicts_data = get_conflicts()
    if not conflicts_data:
        return {"resolved": 0, "errors": 0, "message": "Could not fetch conflicts from memory service"}

    conflicts = conflicts_data.get("conflicts", [])
    flagged = [c for c in conflicts if c.get("status") == "flagged"]

    print(f"[COORDINATOR] Found {len(flagged)} flagged conflicts")

    resolved = 0
    errors = 0
    resolutions = []

    for conflict in flagged:
        entity = conflict["entity"]
        fact_type = conflict["fact_type"]

        print(f"\n[COORDINATOR] Processing conflict: {entity} / {fact_type}")
        print(f"  Option A: {conflict['value_a']} ({conflict['agent_a']}, conf={conflict['confidence_a']})")
        print(f"  Option B: {conflict['value_b']} ({conflict['agent_b']}, conf={conflict['confidence_b']})")

        priority_a = AGENT_PRIORITY.get(conflict["agent_a"], 0)
        priority_b = AGENT_PRIORITY.get(conflict["agent_b"], 0)
        conf_a = conflict.get("confidence_a") or 0
        conf_b = conflict.get("confidence_b") or 0

        # Weighted score: 70% confidence, 30% agent trust ranking
        score_a = conf_a * 0.7 + (priority_a / 4) * 0.3
        score_b = conf_b * 0.7 + (priority_b / 4) * 0.3

        if score_a >= score_b:
            winner_value = conflict["value_a"]
            winner_agent = conflict["agent_a"]
            winning_fact_id = conflict.get("fact_id_a")
            reason = f"{conflict['agent_a']} scored higher (conf={conf_a:.2f}, trust={priority_a})"
        else:
            winner_value = conflict["value_b"]
            winner_agent = conflict["agent_b"]
            winning_fact_id = conflict.get("fact_id_b")
            reason = f"{conflict['agent_b']} scored higher (conf={conf_b:.2f}, trust={priority_b})"

        print(f"  Decision: {winner_value} ({winner_agent}) wins — {reason}")

        if winning_fact_id:
            resolve_result = resolve_conflict(
                conflict_id=conflict.get("conflict_id"),
                winning_fact_id=winning_fact_id,
                reason=reason,
                confidence=max(score_a, score_b)
            )
            if resolve_result:
                resolved += 1
            else:
                errors += 1
                print(f"  Failed to call /memory/resolve")
        else:
            errors += 1
            print(f"  Skipped — conflict data missing fact_id_a/fact_id_b. Confirm field names with Person A.")

        resolutions.append({
            "entity": entity,
            "fact_type": fact_type,
            "winner_value": winner_value,
            "winner_agent": winner_agent,
            "reason": reason
        })

    return {"resolved": resolved, "errors": errors, "resolutions": resolutions}


if __name__ == "__main__":
    print("Testing Coordinator Agent")
    print("=" * 60)

    result = process_flagged_conflicts()
    print(f"\nSummary:")
    print(f"  Resolved: {result['resolved']}")
    print(f"  Errors: {result['errors']}")
    for r in result.get("resolutions", []):
        print(f"  {r['entity']}/{r['fact_type']}: {r['winner_value']} ({r['winner_agent']})")