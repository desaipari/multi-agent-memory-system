"""
Confidence scoring with dynamic per-fact-type agent weights.

Core insight from literature (TOKI arXiv:2606.06240, 
fleet-memory arXiv:2606.24535):
Confidence should be a function of the claim type AND the source's
authoritative domain, not just global source reliability.

Formula:
confidence = (0.35 × source_reliability_for_fact_type)
           + (0.30 × corroboration_score)
           + (0.15 × extraction_directness)
           - (0.20 × time_decay_penalty)

The key change from Week 2: source_reliability is now a
2D lookup (agent_id × fact_type) rather than a 1D lookup
(agent_id only).
"""

from datetime import datetime, timezone

# ── Component weights ────────────────────────────────────────
W_SOURCE_RELIABILITY = 0.35
W_CORROBORATION      = 0.30
W_EXTRACTION_DIRECT  = 0.15
W_TIME_DECAY         = 0.20

# ── Auto-resolution threshold ─────────────────────────────────
AUTO_RESOLVE_THRESHOLD = 0.30

# ── Dynamic agent trust matrix ────────────────────────────────
# Rows: agent_id
# Cols: fact_type
# Values: trust score 0.0 to 1.0
#
# Justification per agent per fact type:
#
# intake_agent reads from primary ticket system (ServiceNow ticket)
#   priority    → 0.90  ticket system is the official source of priority
#   state       → 0.85  initial state is directly recorded at ticket open
#   assignment  → 0.80  initial assignment is recorded at ticket open
#   category    → 0.70  often misclassified at intake, corrected later
#   urgency     → 0.65  subjective at intake, may change after triage
#   impact      → 0.70  self-reported at intake, often inaccurate
#   opened_date → 0.95  timestamp is objective and exact
#   resolved_by → 0.50  intake agent does not know who will resolve it
#
# delivery_agent reads from monitoring logs (live system telemetry)
#   state       → 0.88  monitoring systems track live operational state
#   urgency     → 0.82  monitoring knows actual system impact in real time
#   priority    → 0.72  monitoring-derived priority is reasonable estimate
#   impact      → 0.80  monitoring can measure actual system impact
#   assignment  → 0.58  monitoring logs rarely track team assignments
#   category    → 0.65  monitoring knows symptoms but not root cause well
#   opened_date → 0.60  monitoring timestamp may differ from ticket open
#   resolved_by → 0.55  monitoring may not capture resolver identity
#
# billing_agent reads from field reports (transferred/older records)
#   resolved_by → 0.88  field agents know exactly who resolved it
#   state       → 0.80  field agent sees actual current physical state
#   assignment  → 0.52  field report may have stale team assignment
#   priority    → 0.38  field reports often reflect old priority levels
#   category    → 0.62  field agent knows root cause from direct inspection
#   urgency     → 0.45  field report urgency often stale
#   impact      → 0.55  field report impact may be historical
#   opened_date → 0.40  transferred records often have wrong timestamps
#
# coordinator_agent acts on human judgment or orchestration logic
#   all types   → 0.82  human-directed, generally trustworthy but not
#                        authoritative on any specific domain

# ── Skill-Conditional Agent Trust Matrix ──────────────────────
#
# Theoretical basis:
# Conditional trust R(i|k) — trust in agent i for fact type k —
# rather than a single global score per agent.
# Source: "When Should Agent Trust Be Conditional?"
# arXiv:2606.14200 (June 2026)
#
# Dynamic trust updating after conflict resolution:
# Source: "DynaTrust: Dynamic Trust Graphs for Multi-Agent Systems"
# arXiv:2603.15661 (March 2026)
#
# Domain-grounded weight values:
# Derived from ITSM (IT Service Management) source-of-record
# principles under the ITIL framework, which defines which
# system is authoritative for each incident fact type:
#
# Intake Agent reads from the primary ticket system (ServiceNow).
# In ITSM practice, the ticket system is the system of record for
# priority, assignment, and initial categorization. It has high
# authority for administrative facts set at ticket creation but
# low authority for resolution details it cannot know at intake.
#
# Delivery Agent reads from monitoring logs (live telemetry).
# Monitoring systems are purpose-built for tracking operational
# state and urgency. They have high authority for live system
# facts but low authority for administrative assignments.
#
# Billing/Ops Agent reads from field reports and transferred
# records. Field agents have direct knowledge of resolution
# outcomes but read from potentially stale transferred records
# for administrative facts like priority.
#
# Initial weight values are set by domain authority per ITIL.
# These weights evolve dynamically after each conflict resolution
# following DynaTrust's continuous trust update principle.



# Source-conditional agent trust matrix
# Principle: skill-conditional trust R(agent|fact_type)
# References:
#   - "When Should Agent Trust Be Conditional?" arXiv:2606.14200
#   - STRATUS, NeurIPS 2025, arXiv:2506.02009
#   - OpsAgent, arXiv:2510.24145
#
# Weight values derived from ITSM source-of-record principles:
# each weight answers "how authoritative is this source
# for this specific fact type in IT incident management?"
SOURCE_CONDITIONAL_TRUST = {
    "intake_agent": {        # reads from primary ticket system
        "priority":         0.88,  # ticket system = system of record for priority
        "assignment_group": 0.83,  # assignment set at ticket creation
        "category":         0.78,  # initial categorization, often revised
        "opened_date":      0.95,  # objective timestamp, highly reliable
        "state":            0.72,  # opening state only, not current state
        "urgency":          0.62,  # self-reported, often inaccurate
        "impact":           0.68,  # self-reported at intake
        "resolved_by":      0.38,  # unknown at intake time
        "default":          0.70
    },
    "delivery_agent": {      # reads from monitoring logs (live telemetry)
        "state":            0.90,  # monitoring built specifically to track live state
        "urgency":          0.84,  # monitoring measures actual system impact
        "impact":           0.80,  # monitoring can quantify real impact
        "priority":         0.70,  # inferred from severity, not official
        "category":         0.63,  # monitors symptoms not root cause
        "assignment_group": 0.55,  # monitoring rarely tracks team assignments
        "opened_date":      0.58,  # monitoring timestamp differs from ticket open
        "resolved_by":      0.52,  # monitoring may not capture resolver identity
        "default":          0.66
    },
    "billing_agent": {       # reads from field reports / transferred records
        "resolved_by":      0.90,  # field agent has direct first-hand knowledge
        "state":            0.80,  # field agent sees actual post-resolution state
        "category":         0.60,  # field agent knows root cause from inspection
        "assignment_group": 0.50,  # may reflect stale assignment from transfer
        "impact":           0.52,  # historical impact, not current
        "urgency":          0.44,  # urgency often stale in transferred records
        "priority":         0.38,  # priority frequently stale in field reports
        "opened_date":      0.32,  # timestamps often wrong in transferred records
        "default":          0.50
    },
    "coordinator_agent": {   # human-directed orchestration
        "default":          0.80   # moderate authority across all fact types
    }
}

# Fallback for unknown agents
DEFAULT_AGENT_TRUST_SCORE = 0.50


def get_dynamic_source_reliability(
    agent_id: str,
    fact_type: str,
    db_trust_score: float = None
) -> float:
    """
    Returns skill-conditional trust R(agent_id | fact_type).

    Implements the conditional trust framework from
    arXiv:2606.14200 — trust conditioned on the specific
    fact type being claimed, not a global agent score.

    If the agent has an updated trust score from the database
    (learned after conflict resolutions, per DynaTrust
    arXiv:2603.15661), blends it with domain authority:
      70% domain authority (ITIL-grounded initial weight)
      30% learned reliability (updated from resolution history)
    """
    agent_matrix = SOURCE_CONDITIONAL_TRUST.get(agent_id, {})
    domain_authority = agent_matrix.get(
        fact_type,
        agent_matrix.get("default", 0.50)
    )

    if db_trust_score is not None:
        blended = (0.70 * domain_authority) + (0.30 * db_trust_score)
        return max(0.10, min(0.99, blended))

    return domain_authority


def get_corroboration_score(corroboration_count: int) -> float:
    """
    Diminishing returns on corroboration.
    Each additional confirmation adds less than the previous one.
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
    """Direct statement beats inferred interpretation."""
    if extraction_type == "direct":
        return 0.90
    elif extraction_type == "inferred":
        return 0.45
    else:
        return 0.50


def get_time_decay_penalty(timestamp: datetime) -> float:
    """Recent facts decay slower than old ones."""
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    age_hours = (now - timestamp).total_seconds() / 3600

    if age_hours < 1:
        return 0.02
    elif age_hours < 24:
        return 0.05
    elif age_hours < 168:
        return 0.15
    elif age_hours < 720:
        return 0.30
    else:
        return 0.50


def compute_confidence(
    agent_id: str,
    fact_type: str,
    extraction_type: str,
    corroboration_count: int = 1,
    timestamp: datetime = None,
    db_trust_score: float = None
) -> float:
    """
    Main confidence computation.
    Now requires fact_type so domain expertise can be applied.
    Returns score clamped between 0.10 and 0.99.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    source_reliability = get_dynamic_source_reliability(
        agent_id, fact_type, db_trust_score
    )
    corroboration = get_corroboration_score(corroboration_count)
    directness = get_extraction_directness(extraction_type)
    decay = get_time_decay_penalty(timestamp)

    raw_score = (
        (W_SOURCE_RELIABILITY * source_reliability)
        + (W_CORROBORATION * corroboration)
        + (W_EXTRACTION_DIRECT * directness)
        - (W_TIME_DECAY * decay)
    )

    return round(max(0.10, min(0.99, raw_score)), 4)


def should_auto_resolve(
    confidence_a: float,
    confidence_b: float
) -> bool:
    return abs(confidence_a - confidence_b) >= AUTO_RESOLVE_THRESHOLD


def update_agent_trust_after_resolution(
    winner_agent_id: str,
    loser_agent_id: str,
    db
):
    """Update learned trust scores after conflict resolution."""
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

        current = agent.trust_score or 0.50

        if direction == "up":
            boost = (1.0 - current) * 0.10
            agent.trust_score = min(0.99, current + boost)
            agent.correct_writes = (agent.correct_writes or 0) + 1
        else:
            penalty = current * 0.10
            agent.trust_score = max(0.10, current - penalty)
            agent.overturned_writes = (agent.overturned_writes or 0) + 1

        agent.reliability_score = agent.trust_score

    db.commit()


if __name__ == "__main__":
    print("=== Dynamic Confidence Score Tests ===\n")

    test_cases = [
        # (agent, fact_type, extraction_type, corroboration, label)
        ("intake_agent",   "priority",    "direct",   1, "Intake priority — primary source"),
        ("intake_agent",   "resolved_by", "direct",   1, "Intake resolved_by — weak domain"),
        ("billing_agent",  "priority",    "inferred", 1, "Billing priority — stale source"),
        ("billing_agent",  "resolved_by", "direct",   1, "Billing resolved_by — strong domain"),
        ("delivery_agent", "state",       "direct",   1, "Delivery state — monitoring strong"),
        ("intake_agent",   "priority",    "direct",   2, "Intake priority — corroborated"),
    ]

    for agent, fact_type, extraction, corroboration, label in test_cases:
        score = compute_confidence(
            agent_id=agent,
            fact_type=fact_type,
            extraction_type=extraction,
            corroboration_count=corroboration
        )
        domain = get_dynamic_source_reliability(agent, fact_type)
        print(f"{label}")
        print(f"  Domain expertise: {domain:.2f} | Final score: {score:.4f}")
        print()

    print("=== Key Conflict Scenarios ===\n")
    scenarios = [
        ("intake_agent",  "priority", "direct",   "billing_agent",  "priority", "inferred"),
        ("billing_agent", "resolved_by", "direct", "intake_agent",  "resolved_by", "direct"),
        ("delivery_agent","state",    "direct",   "billing_agent",  "state",    "inferred"),
    ]

    for a_agent, a_fact, a_ext, b_agent, b_fact, b_ext in scenarios:
        score_a = compute_confidence(a_agent, a_fact, a_ext)
        score_b = compute_confidence(b_agent, b_fact, b_ext)
        gap = abs(score_a - score_b)
        decision = "AUTO-RESOLVE" if gap >= AUTO_RESOLVE_THRESHOLD else "CONTESTED"
        print(f"{a_agent}/{a_fact} vs {b_agent}/{b_fact}")
        print(f"  Score A: {score_a:.4f} | Score B: {score_b:.4f}")
        print(f"  Gap: {gap:.4f} | Decision: {decision}")
        print()