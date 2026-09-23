"""
Legacy compatibility helpers for VeriMem.

The final V2 resolution logic lives in:

    verimem_core/config.py
    verimem_core/trust.py
    verimem_core/providers.py
    verimem_core/resolver.py

main.py uses that V2 core directly.

This file is retained only because some older tests or modules may still
import confidence-scoring helper functions from confidence_scorer.py.
"""

from datetime import datetime, timezone

from verimem_core.config import (
    CONFIDENCE_WEIGHTS,
    DIRECTNESS_SCORES,
    MIN_WINNER_SCORE,
    MIN_WINNER_MARGIN,
    get_expert_prior,
    get_directness_score,
)


# ---------------------------------------------------------
# Confidence component weights
# ---------------------------------------------------------

W_SOURCE_RELIABILITY = CONFIDENCE_WEIGHTS["source"]
W_CORROBORATION = CONFIDENCE_WEIGHTS["corroboration"]
W_EXTRACTION_DIRECT = CONFIDENCE_WEIGHTS["directness"]
W_TIME_DECAY = CONFIDENCE_WEIGHTS["time_penalty"]


DEFAULT_AGENT_TRUST_SCORE = 0.50


def get_dynamic_source_reliability(
    agent_id: str,
    fact_type: str,
    db_trust_score: float = None,
) -> float:
    """
    Compatibility helper.

    The V2 resolver obtains contextual R(agent, fact_type) from a
    TrustProvider using the Beta posterior.

    If a contextual reliability value is supplied by the caller,
    this helper returns it directly. Otherwise it falls back to the
    expert prior from config.py.

    No 70/30 blending is performed here.
    """

    if db_trust_score is not None:
        return float(db_trust_score)

    return get_expert_prior(
        agent_id,
        fact_type,
    )


def get_extraction_directness(
    extraction_type: str,
) -> float:
    """
    Return the V2 directness score.

    direct   = 0.90
    inferred = 0.45
    missing/unknown = 0.675
    """

    return get_directness_score(
        extraction_type
    )


def get_corroboration_score(
    corroboration_count: int,
) -> float:
    """
    Legacy compatibility helper only.

    The actual V2 resolver does NOT calculate corroboration from a
    simple count. It uses independent supporting agents and noisy-OR.

    This function is retained only for older unit tests or code paths.
    """

    if corroboration_count <= 1:
        return 0.0

    # Neutral compatibility approximation.
    # Runtime conflict resolution does not use this function.
    additional_supporters = corroboration_count - 1

    return min(
        1.0,
        1.0 - (0.5 ** additional_supporters),
    )


def get_time_decay_penalty(
    timestamp: datetime = None,
) -> float:
    """
    Legacy compatibility helper.

    The actual V2 resolver calculates temporal evidence using
    observed_at and fact-type-specific horizons.

    If observation time is unavailable, V2 uses a neutral penalty 0.5.
    """

    if timestamp is None:
        return 0.50

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(
            tzinfo=timezone.utc
        )

    now = datetime.now(
        timezone.utc
    )

    age_days = max(
        0.0,
        (now - timestamp).total_seconds()
        / 86400.0,
    )

    return min(
        age_days / 30.0,
        1.0,
    )


def compute_confidence(
    agent_id: str,
    fact_type: str,
    extraction_type: str,
    corroboration_count: int = 1,
    timestamp: datetime = None,
    db_trust_score: float = None,
) -> float:
    """
    Compatibility confidence calculation.

    Formula:

        C = 0.35*S
          + 0.30*Cr
          + 0.15*D
          - 0.20*T

    IMPORTANT:
    This function is retained for backward compatibility only.

    The final V2 runtime conflict resolution uses
    verimem_core.resolver.resolve(), which calculates candidate-level
    corroboration, temporal evidence and adaptive contextual trust.
    """

    source_reliability = (
        get_dynamic_source_reliability(
            agent_id=agent_id,
            fact_type=fact_type,
            db_trust_score=db_trust_score,
        )
    )

    corroboration = (
        get_corroboration_score(
            corroboration_count
        )
    )

    directness = (
        get_extraction_directness(
            extraction_type
        )
    )

    decay = (
        get_time_decay_penalty(
            timestamp
        )
    )

    score = (
        W_SOURCE_RELIABILITY
        * source_reliability
        + W_CORROBORATION
        * corroboration
        + W_EXTRACTION_DIRECT
        * directness
        - W_TIME_DECAY
        * decay
    )

    return round(
        score,
        4,
    )


def should_auto_resolve(
    confidence_a: float,
    confidence_b: float,
) -> bool:
    """
    Compatibility check for a two-candidate conflict.

    Final calibrated V2 conditions:

        winner score >= 0.35

    AND

        winner-runner margin >= 0.15

    Threshold calibration used fresh CAL seeds 103-132.

    Selected operating point:
        coverage = 65.4%
        auto-resolution accuracy = 90.4%
        ambiguous contest rate = 99.8%
    """

    winner_score = max(
        confidence_a,
        confidence_b,
    )

    runner_score = min(
        confidence_a,
        confidence_b,
    )

    margin = (
        winner_score
        - runner_score
    )

    return (
        winner_score
        >= MIN_WINNER_SCORE
        and
        margin
        >= MIN_WINNER_MARGIN
    )


if __name__ == "__main__":

    print(
        "=== VeriMem Confidence Compatibility Test ==="
    )

    test_cases = [
        (
            "intake_agent",
            "priority",
            "direct",
        ),
        (
            "billing_agent",
            "priority",
            "inferred",
        ),
        (
            "delivery_agent",
            "state",
            "direct",
        ),
    ]

    for (
        agent,
        fact_type,
        extraction_type,
    ) in test_cases:

        score = compute_confidence(
            agent_id=agent,
            fact_type=fact_type,
            extraction_type=extraction_type,
        )

        print(
            f"{agent}/{fact_type}: "
            f"{score:.4f}"
        )

    print(
        "\nAuto-resolution thresholds:"
    )

    print(
        f"Minimum winner score: "
        f"{MIN_WINNER_SCORE}"
    )

    print(
        f"Minimum margin: "
        f"{MIN_WINNER_MARGIN}"
    )