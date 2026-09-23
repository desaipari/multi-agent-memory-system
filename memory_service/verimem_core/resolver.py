import math
import re

from collections import defaultdict
from datetime import timezone

from .config import (
    CONFIDENCE_WEIGHTS,
    PRIOR_STRENGTH,
    STABLE_FACTS,
    VOLATILE_FACTS,
    STABLE_HORIZON_DAYS,
    VOLATILE_HORIZON_DAYS,
    get_directness_score,
)

from .trust import contextual_trust


TIE_EPSILON = 1e-9


def normalize_value(value):
    if value is None:
        return ""

    value = str(value).lower().strip()

    return re.sub(
        r"[\s\-_]+",
        "",
        value
    )


def normalize_datetime(value):
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


def get_reference_time(
    observations
):
    valid_times = []

    for observation in observations:
        observed_at = (
            normalize_datetime(
                observation.observed_at
            )
        )

        if observed_at is not None:
            valid_times.append(
                observed_at
            )

    if not valid_times:
        return None

    return max(valid_times)


def temporal_penalty(
    observed_at,
    reference_time,
    fact_type
):
    observed_at = (
        normalize_datetime(
            observed_at
        )
    )

    reference_time = (
        normalize_datetime(
            reference_time
        )
    )

    if (
        observed_at is None
        or reference_time is None
    ):
        return 0.50

    age_days = (
        reference_time
        - observed_at
    ).total_seconds() / 86400.0

    age_days = max(
        age_days,
        0.0
    )

    if fact_type in VOLATILE_FACTS:
        horizon = (
            VOLATILE_HORIZON_DAYS
        )

    elif fact_type in STABLE_FACTS:
        horizon = (
            STABLE_HORIZON_DAYS
        )

    else:
        horizon = (
            STABLE_HORIZON_DAYS
        )

    return min(
        age_days / horizon,
        1.0
    )


def noisy_or(
    probabilities
):
    probabilities = list(
        probabilities
    )

    if not probabilities:
        return 0.0

    product = 1.0

    for probability in probabilities:
        probability = max(
            0.0,
            min(1.0, probability)
        )

        product *= (
            1.0 - probability
        )

    return 1.0 - product


def log_odds_support(
    probabilities
):
    probabilities = list(
        probabilities
    )

    if not probabilities:
        return 0.0

    log_odds = 0.0

    for probability in probabilities:
        probability = max(
            0.001,
            min(
                0.999,
                probability
            )
        )

        log_odds += math.log(
            probability
            / (1.0 - probability)
        )

    probability = (
        1.0
        / (
            1.0
            + math.exp(-log_odds)
        )
    )

    if probability <= 0.50:
        return 0.0

    return min(
        1.0,
        2.0 * (
            probability - 0.50
        )
    )


def distinct_supporters(
    observations
):
    by_agent = {}

    for observation in observations:
        current = by_agent.get(
            observation.agent_id
        )

        if current is None:
            by_agent[
                observation.agent_id
            ] = observation

            continue

        current_time = (
            normalize_datetime(
                current.observed_at
            )
        )

        new_time = (
            normalize_datetime(
                observation.observed_at
            )
        )

        if (
            current_time is None
            and new_time is not None
        ):
            by_agent[
                observation.agent_id
            ] = observation

        elif (
            current_time is not None
            and new_time is not None
            and new_time > current_time
        ):
            by_agent[
                observation.agent_id
            ] = observation

    return list(
        by_agent.values()
    )


def score_candidate(
    observations,
    provider,
    fact_type,
    reference_time,
    corroboration_method="noisy_or",
    prior_strength=PRIOR_STRENGTH
):
    supporters = (
        distinct_supporters(
            observations
        )
    )

    evidence = []

    for observation in supporters:
        trust = contextual_trust(
            provider=provider,
            agent_id=(
                observation.agent_id
            ),
            fact_type=fact_type,
            extraction_type=(
                observation.extraction_type
            ),
            prior_strength=(
                prior_strength
            ),
        )

        reliability = trust[
            "probability"
        ]

        directness = (
            get_directness_score(
                observation.extraction_type
            )
        )

        time_penalty = (
            temporal_penalty(
                observed_at=(
                    observation.observed_at
                ),
                reference_time=(
                    reference_time
                ),
                fact_type=fact_type,
            )
        )

        evidence.append({
            "observation":
                observation,

            "reliability":
                reliability,

            "directness":
                directness,

            "time_penalty":
                time_penalty,

            "trust_evidence_count":
                trust[
                    "evidence_count"
                ],
        })

    evidence.sort(
        key=lambda item: (
            -item["reliability"],
            item[
                "observation"
            ].agent_id,
        )
    )

    primary = evidence[0]

    source_score = (
        primary["reliability"]
    )

    additional_support = [
        item["reliability"]
        for item in evidence[1:]
    ]

    if (
        corroboration_method
        == "noisy_or"
    ):
        corroboration_score = (
            noisy_or(
                additional_support
            )
        )

    elif (
        corroboration_method
        == "log_odds"
    ):
        corroboration_score = (
            log_odds_support(
                additional_support
            )
        )

    else:
        raise ValueError(
            "corroboration_method "
            "must be 'noisy_or' "
            "or 'log_odds'"
        )

    reliability_sum = sum(
        item["reliability"]
        for item in evidence
    )

    if reliability_sum > 0:
        directness_score = (
            sum(
                item["reliability"]
                * item["directness"]
                for item in evidence
            )
            / reliability_sum
        )

        time_penalty_score = (
            sum(
                item["reliability"]
                * item[
                    "time_penalty"
                ]
                for item in evidence
            )
            / reliability_sum
        )

    else:
        directness_score = (
            sum(
                item["directness"]
                for item in evidence
            )
            / len(evidence)
        )

        time_penalty_score = (
            sum(
                item[
                    "time_penalty"
                ]
                for item in evidence
            )
            / len(evidence)
        )

    score = (
        CONFIDENCE_WEIGHTS[
            "source"
        ] * source_score

        + CONFIDENCE_WEIGHTS[
            "corroboration"
        ] * corroboration_score

        + CONFIDENCE_WEIGHTS[
            "directness"
        ] * directness_score

        - CONFIDENCE_WEIGHTS[
            "time_penalty"
        ] * time_penalty_score
    )

    representative = (
        evidence[0][
            "observation"
        ]
    )

    return {
        "value":
            representative.value,

        "normalized_value":
            normalize_value(
                representative.value
            ),

        "score":
            score,

        "source_score":
            source_score,

        "corroboration_score":
            corroboration_score,

        "directness_score":
            directness_score,

        "time_penalty":
            time_penalty_score,

        "distinct_agent_count":
            len(evidence),

        "supporting_agents": [
            item[
                "observation"
            ].agent_id
            for item in evidence
        ],

        "evidence": [
            {
                "agent_id":
                    item[
                        "observation"
                    ].agent_id,

                "fact_id":
                    item[
                        "observation"
                    ].fact_id,

                "reliability":
                    item[
                        "reliability"
                    ],

                "directness":
                    item[
                        "directness"
                    ],

                "time_penalty":
                    item[
                        "time_penalty"
                    ],

                "trust_evidence_count":
                    item[
                        "trust_evidence_count"
                    ],
            }
            for item in evidence
        ],
    }


def resolve(
    observations,
    provider,
    min_winner_score=0.60,
    min_margin=0.10,
    corroboration_method="noisy_or",
    prior_strength=PRIOR_STRENGTH
):
    if not observations:
        return {
            "decision": "no_evidence",
            "winner": None,
            "candidates": [],
            "margin": None,
        }

    fact_types = {
        observation.fact_type
        for observation in observations
    }

    if len(fact_types) != 1:
        raise ValueError(
            "All observations passed "
            "to resolve() must have "
            "the same fact_type."
        )

    fact_type = next(
        iter(fact_types)
    )

    grouped = defaultdict(list)

    for observation in observations:
        grouped[
            normalize_value(
                observation.value
            )
        ].append(
            observation
        )

    reference_time = (
        get_reference_time(
            observations
        )
    )

    candidates = []

    for candidate_observations in (
        grouped.values()
    ):
        candidates.append(
            score_candidate(
                observations=(
                    candidate_observations
                ),
                provider=provider,
                fact_type=fact_type,
                reference_time=(
                    reference_time
                ),
                corroboration_method=(
                    corroboration_method
                ),
                prior_strength=(
                    prior_strength
                ),
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate["score"],
            candidate[
                "normalized_value"
            ],
        )
    )

    winner = candidates[0]

    if len(candidates) == 1:
        return {
            "decision":
                "no_conflict",

            "winner":
                winner,

            "runner_up":
                None,

            "margin":
                None,

            "candidates":
                candidates,
        }

    runner_up = candidates[1]

    margin = (
        winner["score"]
        - runner_up["score"]
    )

    if abs(margin) <= TIE_EPSILON:
        return {
            "decision":
                "contested",

            "winner":
                winner,

            "runner_up":
                runner_up,

            "margin":
                margin,

            "reason":
                "tie",

            "candidates":
                candidates,
        }

    if (
        winner["score"]
        >= min_winner_score
        and margin >= min_margin
    ):
        decision = (
            "auto_resolve"
        )

    else:
        decision = (
            "contested"
        )

    return {
        "decision":
            decision,

        "winner":
            winner,

        "runner_up":
            runner_up,

        "margin":
            margin,

        "candidates":
            candidates,
    }