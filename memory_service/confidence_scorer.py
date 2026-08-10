"""
Confidence scoring module.

Formula:
confidence = (0.35 × source_reliability)
           + (0.30 × corroboration_score)
           + (0.15 × extraction_directness)
           - (0.20 × time_decay_penalty)

All components are normalized 0.0 to 1.0.
Final score is clamped between 0.10 and 0.99.

Threshold for auto-resolution:
  confidence_gap >= 0.30 → auto-resolve (higher confidence wins)
  confidence_gap <  0.30 → flag as contested (human review needed)
"""

import math
from datetime import datetime, timezone


# ── Component weights ────────────────────────────────────────
W_SOURCE_RELIABILITY  = 0.35
W_CORROBORATION       = 0.30
W_EXTRACTION_DIRECT   = 0.15
W_TIME_DECAY          = 0.20

# ── Auto-resolution threshold ────────────────────────────────
AUTO_RESOLVE_THRESHOLD = 0.30

# ── Agent base trust scores ──────────────────────────────────
# These are starting values before any learning happens
# They will be updated by the Agent table as resolutions occur
DEFAULT_AGENT_TRUST = {
    "intake_agent":      0.85,  # direct from ticket system, highest trust
    "delivery_agent":    0.75,  # reads from monitoring, high trust
    "billing_agent":     0.55,  # reads from older/transferred records
    "coordinator_agent": 0.80,  # human-directed, high trust
}


def get_source_reliability(agent_id: str, db_trust_score: float = None) -> float:
    """
    Returns agent reliability score 0.0 to 1.0.
    Uses database trust score if available, otherwise uses defaults.
    Trust score is updated after each conflict resolution.
    """
    if db_trust_score is not None:
        return max(0.10, min(0.99, db_trust_score))
    return DEFAULT_AGENT_TRUST.get(agent_id, 0.50)


def get_corroboration_score(corroboration_count: int) -> float:
    """
    Returns corroboration score based on how many independent
    agents have confirmed this fact.

    Diminishing returns — each additional corroboration
    adds less than the previous one.

    1 source  → 0.20 (only one agent said it)
    2 sources → 0.55 (one independent confirmation)
    3 sources → 0.75 (two independent confirmations)
    4+ sources → 0.90 (strongly corroborated)
    """
    if corroboration_count <= 1:
        return 0.20
    elif corroboration_count == 2:
        return 0.55
    elif corroboration_count == 3:
        return 0.75
    else:
        return 0.90


def get_extraction_directness(extraction_type: str) -> float:
    """
    Returns directness score based on how the fact was extracted.
    direct  → agent stated this explicitly from the source document
    inferred → agent interpreted or derived this
    """
    if extraction_type == "direct":
        return 0.90
    elif extraction_type == "inferred":
        return 0.45
    else:
        return 0.50


def get_time_decay_penalty(timestamp: datetime) -> float:
    """
    Returns decay penalty 0.0 to 1.0 based on how old the fact is.
    Recent facts (< 1 hour) get almost no penalty.
    Old facts (> 30 days) get maximum penalty.

    Penalty is applied as a subtraction so higher = worse.
    """
    now = datetime.now(timezone.utc)

    # Make timestamp timezone aware if it is not
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    age_hours = (now - timestamp).total_seconds() / 3600

    if age_hours < 1:
        return 0.02   # very recent, almost no decay
    elif age_hours < 24:
        return 0.05   # within a day, minimal decay
    elif age_hours < 168:  # 7 days
        return 0.15
    elif age_hours < 720:  # 30 days
        return 0.30
    else:
        return 0.50   # very old, significant decay


def compute_confidence(
    agent_id: str,
    extraction_type: str,
    corroboration_count: int = 1,
    timestamp: datetime = None,
    db_trust_score: float = None
) -> float:
    """
    Main confidence computation function.
    Returns a score between 0.10 and 0.99.

    Called on every write and every corroboration.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    source_reliability = get_source_reliability(agent_id, db_trust_score)
    corroboration = get_corroboration_score(corroboration_count)
    directness = get_extraction_directness(extraction_type)
    decay = get_time_decay_penalty(timestamp)

    raw_score = (
        (W_SOURCE_RELIABILITY * source_reliability)
        + (W_CORROBORATION    * corroboration)
        + (W_EXTRACTION_DIRECT * directness)
        - (W_TIME_DECAY        * decay)
    )

    # Clamp between 0.10 and 0.99
    final_score = max(0.10, min(0.99, raw_score))

    return round(final_score, 4)


def should_auto_resolve(confidence_a: float, confidence_b: float) -> bool:
    """
    Returns True if the confidence gap is large enough to auto-resolve.
    Returns False if gap is too small and human review is needed.
    """
    return abs(confidence_a - confidence_b) >= AUTO_RESOLVE_THRESHOLD


def update_agent_trust_after_resolution(
    winner_agent_id: str,
    loser_agent_id: str,
    db
):
    """
    Update agent trust scores after a conflict is resolved.
    Winner gets a small boost, loser gets a small penalty.
    Uses diminishing returns so scores do not go to extremes.

    Import models here to avoid circular imports.
    """
    from models import Agent

    for agent_id, direction in [
        (winner_agent_id, "up"),
        (loser_agent_id, "down")
    ]:
        agent = db.query(Agent).filter(
            Agent.agent_id == agent_id
        ).first()

        if not agent:
            agent = Agent(agent_id=agent_id)
            db.add(agent)
            db.flush()

        current = agent.trust_score

        if direction == "up":
            # Diminishing returns going up
            boost = (1.0 - current) * 0.10
            agent.trust_score = min(0.99, current + boost)
            agent.correct_writes = (agent.correct_writes or 0) + 1
        else:
            # Diminishing penalty going down
            penalty = current * 0.10
            agent.trust_score = max(0.10, current - penalty)
            agent.overturned_writes = (agent.overturned_writes or 0) + 1

        agent.reliability_score = agent.trust_score

    db.commit()


if __name__ == "__main__":
    # Test confidence formula with different scenarios
    print("=== Confidence Score Tests ===\n")

    scenarios = [
        {
            "name": "Intake Agent — direct statement, fresh",
            "agent_id": "intake_agent",
            "extraction_type": "direct",
            "corroboration_count": 1,
        },
        {
            "name": "Intake Agent — corroborated by delivery",
            "agent_id": "intake_agent",
            "extraction_type": "direct",
            "corroboration_count": 2,
        },
        {
            "name": "Billing Agent — indirect, single source",
            "agent_id": "billing_agent",
            "extraction_type": "inferred",
            "corroboration_count": 1,
        },
        {
            "name": "Delivery Agent — inferred, two sources",
            "agent_id": "delivery_agent",
            "extraction_type": "inferred",
            "corroboration_count": 2,
        },
    ]

    for s in scenarios:
        score = compute_confidence(
            agent_id=s["agent_id"],
            extraction_type=s["extraction_type"],
            corroboration_count=s["corroboration_count"]
        )
        print(f"{s['name']}")
        print(f"  Score: {score}")
        print()

    # Test auto-resolve decision
    print("=== Auto-resolve Threshold Tests ===\n")
    pairs = [
        (0.82, 0.45, "Intake vs Billing — should auto-resolve"),
        (0.65, 0.55, "Close scores — should flag for human review"),
        (0.90, 0.58, "High gap — should auto-resolve"),
    ]
    for a, b, label in pairs:
        decision = should_auto_resolve(a, b)
        gap = abs(a - b)
        print(f"{label}")
        print(f"  Scores: {a} vs {b} | Gap: {gap:.2f} | "
              f"Auto-resolve: {decision}")
        print()