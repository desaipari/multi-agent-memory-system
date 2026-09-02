"""
Role-based access control for fact retrieval.

Each fact has a readable_by field — a comma-separated list
of agent_ids permitted to read it, or "all" for unrestricted.

Default access rules by fact type:
- financial/sensitive facts: restricted to specific agents
- operational facts: readable by all agents
- resolution details: readable by coordinator and billing only

These defaults can be overridden at write time.
"""

# Default readable_by per fact type
# Controls which agents can read each type of fact
FACT_TYPE_ACCESS = {
    "priority":         "all",
    "state":            "all",
    "assignment_group": "all",
    "category":         "all",
    "urgency":          "all",
    "impact":           "all",
    "opened_date":      "all",
    # Resolution details only visible to coordinator and billing
    # since these contain sensitive closure information
    "resolved_by":      "coordinator_agent,billing_agent,intake_agent",
    # Recommendations only visible to coordinator and delivery
    "recommendation":   "coordinator_agent,delivery_agent",
}


def get_default_access(fact_type: str) -> str:
    """Return default readable_by for a given fact type."""
    return FACT_TYPE_ACCESS.get(fact_type.lower(), "all")


def can_agent_read(agent_id: str, readable_by: str) -> bool:
    """
    Check if an agent is permitted to read a fact.
    
    readable_by is either:
    - "all" → any agent can read
    - comma-separated agent_ids → only listed agents can read
    """
    if not readable_by or readable_by.strip() == "all":
        return True

    permitted = [
        a.strip() for a in readable_by.split(",")
        if a.strip()
    ]
    return agent_id in permitted


def filter_facts_for_agent(
    facts: list,
    requesting_agent_id: str
) -> list:
    """
    Filter a list of fact objects to only those the
    requesting agent is permitted to read.
    Returns filtered list.
    """
    if not requesting_agent_id:
        return facts  # no filter if no agent specified

    return [
        f for f in facts
        if can_agent_read(
            requesting_agent_id,
            f.readable_by or "all"
        )
    ]