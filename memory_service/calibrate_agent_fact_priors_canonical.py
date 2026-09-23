import calibrate_agent_fact_priors as base


AGENT_ALIASES = {
    "intake_agent": "intake_agent",

    "delivery_agent": "delivery_agent",
    "monitoring_agent": "delivery_agent",

    "billing_agent": "billing_agent",
    "field_report_agent": "billing_agent",

    "coordinator_agent": "coordinator_agent",
}


FACT_TYPE_ALIASES = {
    "state": "state",
    "incident_state": "state",
    "incident state": "state",
    "incident-state": "state",

    "priority": "priority",
    "urgency": "urgency",
    "impact": "impact",
    "category": "category",

    "assignment_group": "assignment_group",
    "assignment group": "assignment_group",
    "assignment-group": "assignment_group",

    "opened_date": "opened_date",
    "opened date": "opened_date",
    "opened-date": "opened_date",

    "resolved_by": "resolved_by",
    "resolved by": "resolved_by",
    "resolved-by": "resolved_by",
}


def canonical_agent(value):
    if value is None:
        return None

    key = str(value).strip().lower()

    return AGENT_ALIASES.get(
        key,
        key
    )


def canonical_fact_type(value):
    if value is None:
        return None

    key = str(value).strip().lower()

    return FACT_TYPE_ALIASES.get(
        key,
        key
    )


def canonical_get_agent(observation):
    for key in [
        "agent_id",
        "agent",
        "source_agent",
    ]:
        value = observation.get(key)

        if value:
            return canonical_agent(
                value
            )

    return None


def canonical_get_fact_type(
    observation,
    scenario
):
    value = observation.get(
        "fact_type"
    )

    if value:
        return canonical_fact_type(
            value
        )

    value = scenario.get(
        "fact_type"
    )

    if value:
        return canonical_fact_type(
            value
        )

    ground_truth = scenario.get(
        "ground_truth",
        {}
    )

    if isinstance(
        ground_truth,
        dict
    ):
        value = ground_truth.get(
            "fact_type"
        )

        if value:
            return canonical_fact_type(
                value
            )

    return None


base.get_agent = canonical_get_agent
base.get_fact_type = canonical_get_fact_type


if __name__ == "__main__":
    base.main()