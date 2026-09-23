import json
import random
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone


SEED = 42

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
        "Awaiting User Info",
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
        "High",
        "Medium",
        "Low",
    ],
    "impact": [
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


def clamp(value, low=0.05, high=0.95):
    return max(low, min(high, value))


def wrong_value(rng, fact_type, truth):
    options = [
        value
        for value in VALUES[fact_type]
        if value != truth
    ]

    return rng.choice(options)


def base_reliability(agent_id, fact_type, regime):
    prior = EXPERT_PRIORS[agent_id][fact_type]

    if regime == "aligned":
        return clamp(prior)

    if regime == "mixed":
        if fact_type in {
            "state",
            "priority",
            "assignment_group",
        }:
            return clamp(1.0 - prior)

        return clamp(prior)

    if regime == "shifted":
        return clamp(1.0 - prior)

    raise ValueError(
        f"Unknown regime: {regime}"
    )


def extraction_adjustment(extraction_type):
    if extraction_type == "direct":
        return 0.05

    return -0.05


def true_reliability(
    agent_id,
    fact_type,
    extraction_type,
    regime
):
    base = base_reliability(
        agent_id,
        fact_type,
        regime
    )

    return clamp(
        base
        + extraction_adjustment(
            extraction_type
        )
    )


def choose_extraction_type(rng):
    return rng.choice(
        ["direct", "inferred"]
    )


def make_observation(
    scenario_id,
    index,
    agent_id,
    fact_type,
    value,
    extraction_type,
    observed_at,
    true_probability,
):
    return {
        "fact_id": (
            f"{scenario_id}_F{index}"
        ),
        "agent_id": agent_id,
        "fact_type": fact_type,
        "value": value,
        "extraction_type": extraction_type,
        "observed_at": observed_at.isoformat(),
        "true_reliability": round(
            true_probability,
            6
        ),
    }


def generate_standard_observations(
    rng,
    scenario_id,
    fact_type,
    truth,
    regime,
    base_time,
):
    observations = []

    agent_order = AGENTS.copy()
    rng.shuffle(agent_order)

    for index, agent_id in enumerate(
        agent_order,
        start=1
    ):
        extraction_type = (
            choose_extraction_type(rng)
        )

        reliability = true_reliability(
            agent_id,
            fact_type,
            extraction_type,
            regime
        )

        correct = (
            rng.random()
            < reliability
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
            + timedelta(hours=index * 2)
        )

        observations.append(
            make_observation(
                scenario_id=scenario_id,
                index=index,
                agent_id=agent_id,
                fact_type=fact_type,
                value=value,
                extraction_type=
                    extraction_type,
                observed_at=observed_at,
                true_probability=
                    reliability,
            )
        )

    return observations


def ensure_conflict(
    rng,
    observations,
    fact_type,
    truth
):
    values = {
        observation["value"]
        for observation in observations
    }

    if len(values) > 1:
        return

    observation = observations[-1]

    observation["value"] = wrong_value(
        rng,
        fact_type,
        truth
    )


def force_newer_correct(
    rng,
    observations,
    fact_type,
    truth
):
    ensure_conflict(
        rng,
        observations,
        fact_type,
        truth
    )

    latest = max(
        observations,
        key=lambda x: x["observed_at"]
    )

    latest["value"] = truth

    earlier = [
        observation
        for observation in observations
        if observation is not latest
    ]

    if all(
        observation["value"] == truth
        for observation in earlier
    ):
        earlier[0]["value"] = wrong_value(
            rng,
            fact_type,
            truth
        )


def force_newer_wrong(
    rng,
    observations,
    fact_type,
    truth
):
    latest = max(
        observations,
        key=lambda x: x["observed_at"]
    )

    latest["value"] = wrong_value(
        rng,
        fact_type,
        truth
    )

    earlier = [
        observation
        for observation in observations
        if observation is not latest
    ]

    if not any(
        observation["value"] == truth
        for observation in earlier
    ):
        earlier[0]["value"] = truth


def make_ambiguous(
    rng,
    scenario_id,
    fact_type,
    regime,
    base_time,
):
    values = rng.sample(
        VALUES[fact_type],
        2
    )

    selected_agents = rng.sample(
        AGENTS,
        2
    )

    observations = []

    for index, (
        agent_id,
        value
    ) in enumerate(
        zip(
            selected_agents,
            values
        ),
        start=1
    ):
        extraction_type = (
            choose_extraction_type(rng)
        )

        reliability = true_reliability(
            agent_id,
            fact_type,
            extraction_type,
            regime
        )

        observations.append(
            make_observation(
                scenario_id=scenario_id,
                index=index,
                agent_id=agent_id,
                fact_type=fact_type,
                value=value,
                extraction_type=
                    extraction_type,
                observed_at=(
                    base_time
                    + timedelta(
                        hours=index * 2
                    )
                ),
                true_probability=
                    reliability,
            )
        )

    return observations


def generate_scenario(
    rng,
    scenario_number,
    category,
    regime,
    split_name,
):
    scenario_id = (
        f"V2_{split_name.upper()}_"
        f"{scenario_number:05d}"
    )

    fact_type = rng.choice(
        FACT_TYPES
    )

    truth = rng.choice(
        VALUES[fact_type]
    )

    base_time = datetime(
        2025,
        1,
        1,
        tzinfo=timezone.utc
    ) + timedelta(
        days=scenario_number
    )

    if category == "ambiguous":
        observations = make_ambiguous(
            rng=rng,
            scenario_id=scenario_id,
            fact_type=fact_type,
            regime=regime,
            base_time=base_time,
        )

        return {
            "scenario_id": scenario_id,
            "incident_id": (
                f"V2INC{scenario_number:07d}"
            ),
            "split": split_name,
            "category": category,
            "regime": regime,
            "fact_type": fact_type,
            "reference_value": None,
            "expected_action": "contest",
            "observations": observations,
        }

    observations = (
        generate_standard_observations(
            rng=rng,
            scenario_id=scenario_id,
            fact_type=fact_type,
            truth=truth,
            regime=regime,
            base_time=base_time,
        )
    )

    if category == "newer_correct":
        force_newer_correct(
            rng,
            observations,
            fact_type,
            truth
        )

    elif category == "newer_wrong":
        force_newer_wrong(
            rng,
            observations,
            fact_type,
            truth
        )

    elif category in {
        "source_specialization",
        "reliability_shift",
    }:
        ensure_conflict(
            rng,
            observations,
            fact_type,
            truth
        )

    else:
        raise ValueError(
            f"Unknown category: {category}"
        )

    return {
        "scenario_id": scenario_id,
        "incident_id": (
            f"V2INC{scenario_number:07d}"
        ),
        "split": split_name,
        "category": category,
        "regime": regime,
        "fact_type": fact_type,
        "reference_value": truth,
        "expected_action": "resolve",
        "observations": observations,
    }


def build_split(
    rng,
    split_name,
    start_number,
    count
):
    categories = [
        "newer_correct",
        "newer_wrong",
        "ambiguous",
        "source_specialization",
        "reliability_shift",
    ]

    regimes = [
        "aligned",
        "mixed",
        "shifted",
    ]

    scenarios = []

    for offset in range(count):
        scenario_number = (
            start_number + offset
        )

        category = categories[
            offset % len(categories)
        ]

        if category == "reliability_shift":
            regime = rng.choice(
                ["mixed", "shifted"]
            )
        else:
            regime = regimes[
                offset % len(regimes)
            ]

        scenario = generate_scenario(
            rng=rng,
            scenario_number=
                scenario_number,
            category=category,
            regime=regime,
            split_name=split_name,
        )

        scenarios.append(
            scenario
        )

    rng.shuffle(scenarios)

    return scenarios


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2
        ),
        encoding="utf-8"
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(65536),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    rng = random.Random(SEED)

    output_dir = (
        Path(__file__).resolve().parent
        / "v2"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dev = build_split(
        rng=rng,
        split_name="dev",
        start_number=1,
        count=1800,
    )

    calibration = build_split(
        rng=rng,
        split_name="calibration",
        start_number=100001,
        count=600,
    )

    test = build_split(
        rng=rng,
        split_name="test",
        start_number=200001,
        count=600,
    )

    dev_path = (
        output_dir / "dev.json"
    )

    calibration_path = (
        output_dir
        / "calibration.json"
    )

    test_path = (
        output_dir
        / "test_LOCKED.json"
    )

    write_json(
        dev_path,
        {"scenarios": dev}
    )

    write_json(
        calibration_path,
        {"scenarios": calibration}
    )

    write_json(
        test_path,
        {"scenarios": test}
    )

    metadata = {
        "version": "verimem_v2_controlled_1",
        "seed": SEED,
        "truth_definition": (
            "Latent reference truth is generated "
            "independently of report order."
        ),
        "test_policy": (
            "TEST must not be used during "
            "development or threshold tuning."
        ),
        "split_sizes": {
            "dev": len(dev),
            "calibration":
                len(calibration),
            "test": len(test),
        },
        "hashes": {
            "dev":
                sha256_file(dev_path),
            "calibration":
                sha256_file(
                    calibration_path
                ),
            "test_LOCKED":
                sha256_file(
                    test_path
                ),
        },
    }

    metadata_path = (
        output_dir / "metadata.json"
    )

    write_json(
        metadata_path,
        metadata
    )

    print(
        "V2 benchmark generated."
    )

    print(
        f"DEV: {len(dev)}"
    )

    print(
        f"CALIBRATION: "
        f"{len(calibration)}"
    )

    print(
        f"TEST LOCKED: {len(test)}"
    )

    print(
        "TEST SHA256:",
        metadata["hashes"][
            "test_LOCKED"
        ]
    )


if __name__ == "__main__":
    main()