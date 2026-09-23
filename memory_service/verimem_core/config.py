EXPERT_PRIORS = {
    "intake_agent": {
        "priority": 0.88,
        "assignment_group": 0.83,
        "category": 0.78,
        "opened_date": 0.95,
        "state": 0.72,
        "urgency": 0.62,
        "impact": 0.68,
        "resolved_by": 0.38,
        "default": 0.70,
    },

    "delivery_agent": {
        "state": 0.90,
        "urgency": 0.84,
        "impact": 0.80,
        "priority": 0.70,
        "category": 0.63,
        "assignment_group": 0.55,
        "opened_date": 0.58,
        "resolved_by": 0.52,
        "default": 0.66,
    },

    "billing_agent": {
        "resolved_by": 0.90,
        "state": 0.80,
        "category": 0.60,
        "assignment_group": 0.50,
        "impact": 0.52,
        "urgency": 0.44,
        "priority": 0.38,
        "opened_date": 0.32,
        "default": 0.50,
    },

    "coordinator_agent": {
        "default": 0.80,
    },
}


CONFIDENCE_WEIGHTS = {
    "source": 0.35,
    "corroboration": 0.30,
    "directness": 0.15,
    "time_penalty": 0.20,
}


DIRECTNESS_SCORES = {
    "direct": 0.90,
    "inferred": 0.45,
}


VOLATILE_FACTS = {
    "state",
    "priority",
    "urgency",
    "impact",
    "assignment_group",
}


STABLE_FACTS = {
    "category",
    "opened_date",
    "resolved_by",
}


VOLATILE_HORIZON_DAYS = 7.0
STABLE_HORIZON_DAYS = 30.0


PRIOR_STRENGTH = 5.0

# ---------------------------------------------------------
# V2 AUTO-RESOLUTION THRESHOLDS
# ---------------------------------------------------------
#
# Calibrated on fresh CAL seeds 103-132:
# - 81,000 natural scenarios
# - 56,611 natural conflict cases
# - 40,500 targeted stress scenarios
#
# 42 threshold combinations were evaluated.
#
# Predeclared selection rule:
# 1. Auto-resolution accuracy >= 90%
# 2. Ambiguous-case contest rate >= 95%
# 3. Among qualifying pairs, maximize automatic coverage
# 4. Exact ties prefer stricter thresholds
#
# Selected operating point:
# - coverage: 65.4%
# - auto-resolution accuracy: 90.4%
# - selective error: 9.6%
# - ambiguous contest rate: 99.8%
# - newer-wrong safe rate: 88.3%

MIN_WINNER_SCORE = 0.35
MIN_WINNER_MARGIN = 0.15

def get_expert_prior(agent_id, fact_type):
    matrix = EXPERT_PRIORS.get(agent_id)

    if matrix is None:
        return 0.50

    return matrix.get(
        fact_type,
        matrix.get("default", 0.50)
    )


def get_directness_score(extraction_type):
    if extraction_type is None:
        return 0.675

    return DIRECTNESS_SCORES.get(
        extraction_type.lower(),
        0.675
    )