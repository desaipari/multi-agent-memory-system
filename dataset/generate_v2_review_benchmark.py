import hashlib
import json
import random

from copy import deepcopy
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path


SEED = 73

AGENTS = [
    "intake_agent",
    "delivery_agent",
    "billing_agent",
]

FACT_TYPES = [
    "state",
    "priority",
    "urgency",
    "impact",
    "assignment_group",
    "category",
]


VALUES = {
    "state": [
        "New",
        "Active",
        "Resolved",
        "Closed",
    ],

    "priority": [
        "1-Critical",
        "2-High",
        "3-Medium",
        "4-Low",
    ],

    "urgency": [
        "Critical",
        "High",
        "Medium",
        "Low",
    ],

    "impact": [
        "Critical",
        "High",
        "Medium",
        "Low",
    ],

    "assignment_group": [
        "Service Desk",
        "Network Team",
        "Application Support",
        "Infrastructure",
    ],

    "category": [
        "Network",
        "Software",
        "Hardware",
        "Access",
    ],
}


EXPERT_PRIORS = {
    "intake_agent": {
        "priority": 0.88,
        "assignment_group": 0.83,
        "category": 0.78,
        "state": 0.72,
        "urgency": 0.62,
        "impact": 0.68,
    },

    "delivery_agent": {
        "state": 0.90,
        "urgency": 0.84,
        "impact": 0.80,
        "priority": 0.70,
        "category": 0.63,
        "assignment_group": 0.55,
    },

    "billing_agent": {
        "state": 0.80,
        "category": 0.60,
        "assignment_group": 0.50,
        "impact": 0.52,
        "urgency": 0.44,
        "priority": 0.38,
    },
}


DIRECT_EFFECT = {
    "direct": 0.05,
    "inferred": -0.05,
}


MIXED_SHIFT_FACTS = {
    "state",
    "priority",
    "assignment_group",
}


ADAPTATION_PER_REGIME = 300
EVALUATION_PER_REGIME = 900

STRESS_PER_CATEGORY_PER_REGIME = 150


def clamp(
    value,
    low=0.05,
    high=0.95
):
    return max(
        low,
        min(high, value)
    )


def normalize(value):
    return (
        str(value)
        .strip()
        .lower()
    )


def wrong_value(
    rng,
    fact_type,
    truth
):
    choices = [
        value
        for value
        in VALUES[fact_type]
        if value != truth
    ]

    return rng.choice(
        choices
    )


def aligned_reliabilities():
    return deepcopy(
        EXPERT_PRIORS
    )


def shifted_reliabilities():
    result = {
        agent: {}
        for agent in AGENTS
    }

    for fact_type in FACT_TYPES:
        intake_value = (
            EXPERT_PRIORS[
                "intake_agent"
            ][fact_type]
        )

        delivery_value = (
            EXPERT_PRIORS[
                "delivery_agent"
            ][fact_type]
        )

        billing_value = (
            EXPERT_PRIORS[
                "billing_agent"
            ][fact_type]
        )

        result[
            "intake_agent"
        ][fact_type] = (
            billing_value
        )

        result[
            "delivery_agent"
        ][fact_type] = (
            intake_value
        )

        result[
            "billing_agent"
        ][fact_type] = (
            delivery_value
        )

    return result


def mixed_reliabilities():
    aligned = (
        aligned_reliabilities()
    )

    shifted = (
        shifted_reliabilities()
    )

    result = deepcopy(
        aligned
    )

    for fact_type in (
        MIXED_SHIFT_FACTS
    ):
        for agent_id in AGENTS:
            result[
                agent_id
            ][fact_type] = (
                shifted[
                    agent_id
                ][fact_type]
            )

    return result


def build_regimes():
    return {
        "aligned":
            aligned_reliabilities(),

        "mixed":
            mixed_reliabilities(),

        "shifted":
            shifted_reliabilities(),
    }


def choose_extraction_type(
    rng
):
    return rng.choice([
        "direct",
        "inferred",
    ])


def effective_probability(
    source_reliability,
    extraction_type
):
    return clamp(
        source_reliability
        + DIRECT_EFFECT[
            extraction_type
        ]
    )


def make_observation(
    scenario_id,
    index,
    agent_id,
    fact_type,
    value,
    extraction_type,
    observed_at,
    source_reliability,
    effective_probability_value,
    forced=False,
):
    return {
        "fact_id":
            f"{scenario_id}_F{index}",

        "agent_id":
            agent_id,

        "fact_type":
            fact_type,

        "value":
            value,

        "extraction_type":
            extraction_type,

        "observed_at":
            observed_at.isoformat(),

        "source_reliability":
            round(
                source_reliability,
                6
            ),

        "effective_correct_probability":
            round(
                effective_probability_value,
                6
            ),

        "forced":
            forced,
    }


def generate_natural_scenario(
    rng,
    regime,
    reliabilities,
    phase,
    scenario_index,
):
    fact_type = FACT_TYPES[
        scenario_index
        % len(FACT_TYPES)
    ]

    truth = rng.choice(
        VALUES[fact_type]
    )

    scenario_id = (
        f"V2R_{regime.upper()}_"
        f"{phase.upper()}_"
        f"{scenario_index:05d}"
    )

    base_time = (
        datetime(
            2025,
            1,
            1,
            tzinfo=timezone.utc
        )
        + timedelta(
            days=scenario_index
        )
    )

    report_order = (
        AGENTS.copy()
    )

    rng.shuffle(
        report_order
    )

    observations = []

    for time_rank, agent_id in enumerate(
        report_order,
        start=1
    ):
        extraction_type = (
            choose_extraction_type(
                rng
            )
        )

        source_reliability = (
            reliabilities[
                agent_id
            ][fact_type]
        )

        effective = (
            effective_probability(
                source_reliability,
                extraction_type
            )
        )

        correct = (
            rng.random()
            < effective
        )

        if correct:
            value = truth
        else:
            value = wrong_value(
                rng,
                fact_type,
                truth
            )

        observed_at = (
            base_time
            + timedelta(
                hours=time_rank
            )
        )

        observations.append(
            make_observation(
                scenario_id=
                    scenario_id,

                index=time_rank,

                agent_id=
                    agent_id,

                fact_type=
                    fact_type,

                value=value,

                extraction_type=
                    extraction_type,

                observed_at=
                    observed_at,

                source_reliability=
                    source_reliability,

                effective_probability_value=
                    effective,

                forced=False,
            )
        )

    distinct_values = {
        normalize(
            observation[
                "value"
            ]
        )
        for observation
        in observations
    }

    return {
        "scenario_id":
            scenario_id,

        "regime":
            regime,

        "phase":
            phase,

        "fact_type":
            fact_type,

        "reference_value":
            truth,

        "is_conflict":
            (
                len(
                    distinct_values
                ) > 1
            ),

        "observations":
            observations,
    }


def closest_agent_pair(
    reliabilities,
    fact_type
):
    pairs = []

    for i in range(
        len(AGENTS)
    ):
        for j in range(
            i + 1,
            len(AGENTS)
        ):
            a = AGENTS[i]
            b = AGENTS[j]

            difference = abs(
                reliabilities[
                    a
                ][fact_type]
                -
                reliabilities[
                    b
                ][fact_type]
            )

            pairs.append(
                (
                    difference,
                    a,
                    b,
                )
            )

    pairs.sort()

    return (
        pairs[0][1],
        pairs[0][2],
    )


def generate_stress_scenario(
    rng,
    regime,
    reliabilities,
    category,
    scenario_index,
):
    fact_type = FACT_TYPES[
        scenario_index
        % len(FACT_TYPES)
    ]

    scenario_id = (
        f"V2S_{regime.upper()}_"
        f"{category.upper()}_"
        f"{scenario_index:05d}"
    )

    base_time = (
        datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc
        )
        + timedelta(
            days=scenario_index
        )
    )

    if category == "ambiguous":
        agent_a, agent_b = (
            closest_agent_pair(
                reliabilities,
                fact_type
            )
        )

        values = rng.sample(
            VALUES[fact_type],
            2
        )

        observations = []

        for index, (
            agent_id,
            value
        ) in enumerate(
            [
                (
                    agent_a,
                    values[0]
                ),
                (
                    agent_b,
                    values[1]
                ),
            ],
            start=1
        ):
            source_reliability = (
                reliabilities[
                    agent_id
                ][fact_type]
            )

            extraction_type = (
                "direct"
            )

            effective = (
                effective_probability(
                    source_reliability,
                    extraction_type
                )
            )

            observations.append(
                make_observation(
                    scenario_id=
                        scenario_id,

                    index=index,

                    agent_id=
                        agent_id,

                    fact_type=
                        fact_type,

                    value=value,

                    extraction_type=
                        extraction_type,

                    observed_at=
                        base_time,

                    source_reliability=
                        source_reliability,

                    effective_probability_value=
                        effective,

                    forced=True,
                )
            )

        return {
            "scenario_id":
                scenario_id,

            "regime":
                regime,

            "category":
                category,

            "fact_type":
                fact_type,

            "reference_value":
                None,

            "expected_action":
                "contest",

            "observations":
                observations,
        }

    truth = rng.choice(
        VALUES[fact_type]
    )

    report_order = (
        AGENTS.copy()
    )

    rng.shuffle(
        report_order
    )

    observations = []

    for time_rank, agent_id in enumerate(
        report_order,
        start=1
    ):
        extraction_type = (
            choose_extraction_type(
                rng
            )
        )

        source_reliability = (
            reliabilities[
                agent_id
            ][fact_type]
        )

        effective = (
            effective_probability(
                source_reliability,
                extraction_type
            )
        )

        correct = (
            rng.random()
            < effective
        )

        if correct:
            value = truth
        else:
            value = wrong_value(
                rng,
                fact_type,
                truth
            )

        observations.append(
            make_observation(
                scenario_id=
                    scenario_id,

                index=time_rank,

                agent_id=
                    agent_id,

                fact_type=
                    fact_type,

                value=value,

                extraction_type=
                    extraction_type,

                observed_at=(
                    base_time
                    + timedelta(
                        hours=time_rank
                    )
                ),

                source_reliability=
                    source_reliability,

                effective_probability_value=
                    effective,

                forced=False,
            )
        )

    latest = observations[-1]

    if category == "newer_correct":
        latest["value"] = truth
        latest["forced"] = True

        earlier = (
            observations[:-1]
        )

        if all(
            normalize(
                observation[
                    "value"
                ]
            )
            == normalize(truth)
            for observation
            in earlier
        ):
            earlier[0][
                "value"
            ] = wrong_value(
                rng,
                fact_type,
                truth
            )

            earlier[0][
                "forced"
            ] = True

    elif category == "newer_wrong":
        latest[
            "value"
        ] = wrong_value(
            rng,
            fact_type,
            truth
        )

        latest[
            "forced"
        ] = True

        earlier = (
            observations[:-1]
        )

        if not any(
            normalize(
                observation[
                    "value"
                ]
            )
            == normalize(truth)
            for observation
            in earlier
        ):
            earlier[0][
                "value"
            ] = truth

            earlier[0][
                "forced"
            ] = True

    else:
        raise ValueError(
            f"Unknown stress category: "
            f"{category}"
        )

    return {
        "scenario_id":
            scenario_id,

        "regime":
            regime,

        "category":
            category,

        "fact_type":
            fact_type,

        "reference_value":
            truth,

        "expected_action":
            "resolve",

        "observations":
            observations,
    }


def generate_reliability_benchmark(
    rng,
    regimes
):
    output = {}

    global_index = 1

    for regime, reliabilities in (
        regimes.items()
    ):
        adaptation = []

        for _ in range(
            ADAPTATION_PER_REGIME
        ):
            adaptation.append(
                generate_natural_scenario(
                    rng=rng,
                    regime=regime,
                    reliabilities=
                        reliabilities,
                    phase="adaptation",
                    scenario_index=
                        global_index,
                )
            )

            global_index += 1

        evaluation = []

        for _ in range(
            EVALUATION_PER_REGIME
        ):
            evaluation.append(
                generate_natural_scenario(
                    rng=rng,
                    regime=regime,
                    reliabilities=
                        reliabilities,
                    phase="evaluation",
                    scenario_index=
                        global_index,
                )
            )

            global_index += 1

        output[regime] = {
            "true_source_reliabilities":
                reliabilities,

            "adaptation":
                adaptation,

            "evaluation":
                evaluation,
        }

    return output


def generate_stress_benchmark(
    rng,
    regimes
):
    categories = [
        "newer_correct",
        "newer_wrong",
        "ambiguous",
    ]

    output = {}

    global_index = 1

    for regime, reliabilities in (
        regimes.items()
    ):
        scenarios = []

        for category in categories:
            for _ in range(
                STRESS_PER_CATEGORY_PER_REGIME
            ):
                scenarios.append(
                    generate_stress_scenario(
                        rng=rng,
                        regime=regime,
                        reliabilities=
                            reliabilities,
                        category=category,
                        scenario_index=
                            global_index,
                    )
                )

                global_index += 1

        rng.shuffle(
            scenarios
        )

        output[regime] = (
            scenarios
        )

    return output


def write_json(
    path,
    data
):
    path.write_text(
        json.dumps(
            data,
            indent=2
        ),
        encoding="utf-8"
    )


def sha256_file(path):
    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda:
                file.read(65536),
            b""
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def main():
    rng = random.Random(
        SEED
    )

    output_dir = (
        Path(__file__)
        .resolve()
        .parent
        / "v2_review"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    regimes = build_regimes()

    reliability_data = (
        generate_reliability_benchmark(
            rng,
            regimes
        )
    )

    stress_data = (
        generate_stress_benchmark(
            rng,
            regimes
        )
    )

    reliability_path = (
        output_dir
        / "dev_reliability.json"
    )

    stress_path = (
        output_dir
        / "dev_stress.json"
    )

    write_json(
        reliability_path,
        {
            "seed": SEED,
            "regimes":
                reliability_data,
        }
    )

    write_json(
        stress_path,
        {
            "seed": SEED,
            "regimes":
                stress_data,
        }
    )

    metadata = {
        "version":
            "verimem_v2_review_aligned_dev",

        "seed":
            SEED,

        "confidence_formula":
            (
                "0.35 source + "
                "0.30 corroboration + "
                "0.15 directness - "
                "0.20 time penalty"
            ),

        "adaptation_labels_per_regime":
            ADAPTATION_PER_REGIME,

        "evaluation_cases_per_regime":
            EVALUATION_PER_REGIME,

        "stress_cases_per_category_per_regime":
            (
                STRESS_PER_CATEGORY_PER_REGIME
            ),

        "regime_definition": {
            "aligned":
                (
                    "True source authority "
                    "matches expert priors."
                ),

            "mixed":
                (
                    "Source authority is "
                    "reassigned for selected "
                    "fact types."
                ),

            "shifted":
                (
                    "Source authority is "
                    "reassigned for all "
                    "fact types."
                ),
        },

        "truth_policy":
            (
                "Reference truth is chosen "
                "before reports and report "
                "order is randomized "
                "independently."
            ),

        "hashes": {
            "dev_reliability":
                sha256_file(
                    reliability_path
                ),

            "dev_stress":
                sha256_file(
                    stress_path
                ),
        },
    }

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    write_json(
        metadata_path,
        metadata
    )

    print(
        "Review-aligned V2 DEV "
        "benchmark generated."
    )

    print(
        "Reliability benchmark:"
    )

    for regime in regimes:
        print(
            f"  {regime}: "
            f"{ADAPTATION_PER_REGIME} "
            f"adaptation + "
            f"{EVALUATION_PER_REGIME} "
            f"evaluation"
        )

    print(
        "Stress benchmark:"
    )

    for regime in regimes:
        print(
            f"  {regime}: "
            f"{len(stress_data[regime])}"
        )

    print(
        "No CAL or TEST files "
        "were generated."
    )


if __name__ == "__main__":
    main()