from datetime import (
    datetime,
    timezone,
)

from verimem_core.types import (
    Observation,
)

from verimem_core.providers import (
    InMemoryTrustProvider,
)

from verimem_core.resolver import (
    resolve,
)

from verimem_core.trust import (
    contextual_trust,
)


def make_observation(
    fact_id,
    agent_id,
    fact_type,
    value,
    observed_at,
    extraction_type="direct"
):
    return Observation(
        fact_id=fact_id,
        agent_id=agent_id,
        fact_type=fact_type,
        value=value,
        extraction_type=(
            extraction_type
        ),
        observed_at=observed_at,
    )


def test_same_value_is_not_conflict():
    provider = (
        InMemoryTrustProvider()
    )

    observations = [
        make_observation(
            "1",
            "intake_agent",
            "priority",
            "High",
            datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc
            ),
        ),

        make_observation(
            "2",
            "delivery_agent",
            "priority",
            "High",
            datetime(
                2026,
                1,
                2,
                tzinfo=timezone.utc
            ),
        ),
    ]

    result = resolve(
        observations,
        provider
    )

    assert (
        result["decision"]
        == "no_conflict"
    )


def test_exact_tie_becomes_contested():
    provider = (
        InMemoryTrustProvider()
    )

    same_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc
    )

    observations = [
        make_observation(
            "1",
            "unknown_agent_a",
            "priority",
            "High",
            same_time,
        ),

        make_observation(
            "2",
            "unknown_agent_b",
            "priority",
            "Medium",
            same_time,
        ),
    ]

    result = resolve(
        observations,
        provider,
        min_winner_score=0.0,
        min_margin=0.0,
    )

    assert (
        result["decision"]
        == "contested"
    )

    assert abs(
        result["margin"]
    ) <= 1e-9


def test_newer_value_does_not_automatically_win():
    provider = (
        InMemoryTrustProvider()
    )

    provider.set_stats(
        "intake_agent",
        "priority",
        successes=90,
        failures=10,
    )

    provider.set_stats(
        "billing_agent",
        "priority",
        successes=20,
        failures=80,
    )

    observations = [
        make_observation(
            "1",
            "intake_agent",
            "priority",
            "High",
            datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc
            ),
        ),

        make_observation(
            "2",
            "billing_agent",
            "priority",
            "Medium",
            datetime(
                2026,
                1,
                2,
                tzinfo=timezone.utc
            ),
        ),
    ]

    result = resolve(
        observations,
        provider,
        min_winner_score=0.0,
        min_margin=0.0,
    )

    assert (
        result[
            "winner"
        ][
            "normalized_value"
        ]
        == "high"
    )


def test_multiple_agents_create_corroboration():
    provider = (
        InMemoryTrustProvider()
    )

    same_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc
    )

    observations = [
        make_observation(
            "1",
            "intake_agent",
            "priority",
            "High",
            same_time,
        ),

        make_observation(
            "2",
            "delivery_agent",
            "priority",
            "High",
            same_time,
        ),

        make_observation(
            "3",
            "billing_agent",
            "priority",
            "Medium",
            same_time,
        ),
    ]

    result = resolve(
        observations,
        provider,
        min_winner_score=0.0,
        min_margin=0.0,
    )

    high_candidate = next(
        candidate
        for candidate
        in result["candidates"]
        if (
            candidate[
                "normalized_value"
            ]
            == "high"
        )
    )

    medium_candidate = next(
        candidate
        for candidate
        in result["candidates"]
        if (
            candidate[
                "normalized_value"
            ]
            == "medium"
        )
    )

    assert (
        high_candidate[
            "distinct_agent_count"
        ]
        == 2
    )

    assert (
        high_candidate[
            "corroboration_score"
        ]
        > 0
    )

    assert (
        medium_candidate[
            "corroboration_score"
        ]
        == 0
    )


def test_single_source_does_not_corroborate_itself():
    provider = (
        InMemoryTrustProvider()
    )

    same_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc
    )

    observations = [
        make_observation(
            "1",
            "intake_agent",
            "priority",
            "High",
            same_time,
        ),

        make_observation(
            "2",
            "billing_agent",
            "priority",
            "Medium",
            same_time,
        ),
    ]

    result = resolve(
        observations,
        provider,
        min_winner_score=0.0,
        min_margin=0.0,
    )

    for candidate in (
        result["candidates"]
    ):
        assert (
            candidate[
                "corroboration_score"
            ]
            == 0
        )


def test_directness_is_separate_from_trust():
    provider = (
        InMemoryTrustProvider()
    )

    same_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc
    )

    observations = [
        make_observation(
            "1",
            "unknown_agent_a",
            "priority",
            "High",
            same_time,
            "direct",
        ),

        make_observation(
            "2",
            "unknown_agent_b",
            "priority",
            "Medium",
            same_time,
            "inferred",
        ),
    ]

    result = resolve(
        observations,
        provider,
        min_winner_score=0.0,
        min_margin=0.0,
    )

    high = next(
        candidate
        for candidate
        in result["candidates"]
        if (
            candidate[
                "normalized_value"
            ]
            == "high"
        )
    )

    medium = next(
        candidate
        for candidate
        in result["candidates"]
        if (
            candidate[
                "normalized_value"
            ]
            == "medium"
        )
    )

    assert (
        high["source_score"]
        == medium["source_score"]
    )

    assert (
        high["directness_score"]
        > medium[
            "directness_score"
        ]
    )

    assert (
        high["score"]
        > medium["score"]
    )


def test_verified_history_changes_contextual_trust():
    provider = (
        InMemoryTrustProvider()
    )

    before = contextual_trust(
        provider=provider,
        agent_id=(
            "billing_agent"
        ),
        fact_type="priority",
    )

    for _ in range(50):
        provider.record_verified_outcome(
            agent_id=(
                "billing_agent"
            ),
            fact_type="priority",
            correct=True,
        )

    after = contextual_trust(
        provider=provider,
        agent_id=(
            "billing_agent"
        ),
        fact_type="priority",
    )

    assert (
        after["probability"]
        > before["probability"]
    )


def test_extraction_type_does_not_create_separate_trust_bucket():
    provider = (
        InMemoryTrustProvider()
    )

    for _ in range(20):
        provider.record_verified_outcome(
            agent_id=(
                "delivery_agent"
            ),
            fact_type="state",
            extraction_type="direct",
            correct=True,
        )

    direct = contextual_trust(
        provider=provider,
        agent_id=(
            "delivery_agent"
        ),
        fact_type="state",
        extraction_type="direct",
    )

    inferred = contextual_trust(
        provider=provider,
        agent_id=(
            "delivery_agent"
        ),
        fact_type="state",
        extraction_type="inferred",
    )

    assert (
        direct["probability"]
        == inferred["probability"]
    )


def test_missing_observed_at_is_handled():
    provider = (
        InMemoryTrustProvider()
    )

    observations = [
        make_observation(
            "1",
            "intake_agent",
            "priority",
            "High",
            None,
        ),

        make_observation(
            "2",
            "billing_agent",
            "priority",
            "Medium",
            None,
        ),
    ]

    result = resolve(
        observations,
        provider,
        min_winner_score=0.0,
        min_margin=0.0,
    )

    assert (
        len(
            result["candidates"]
        )
        == 2
    )

    for candidate in (
        result["candidates"]
    ):
        assert (
            candidate[
                "time_penalty"
            ]
            == 0.50
        )