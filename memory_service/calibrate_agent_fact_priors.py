import argparse
import csv
import hashlib
import json
import math

from collections import defaultdict
from pathlib import Path

from verimem_core.config import (
    EXPERT_PRIORS,
    get_expert_prior,
)

from verimem_core.resolver import (
    normalize_value,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help=(
            "Directory containing labelled "
            "DEV scenario JSON files."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "memory_service/"
            "results_agent_fact_calibration"
        )
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5
    )

    parser.add_argument(
        "--prior-strength",
        type=float,
        default=5.0
    )

    return parser.parse_args()


def load_json(path):
    try:
        with path.open(
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception:
        return None


def scenario_list(data):
    if data is None:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(
            data.get("scenarios"),
            list
        ):
            return data[
                "scenarios"
            ]

        return [data]

    return []


def get_truth(scenario):
    ground_truth = scenario.get(
        "ground_truth",
        {}
    )

    candidates = []

    if isinstance(
        ground_truth,
        dict
    ):
        # Generic truth fields
        candidates.extend([
            ground_truth.get(
                "correct_value"
            ),
            ground_truth.get(
                "reference_value"
            ),
            ground_truth.get(
                "truth"
            ),

            # Corroboration scenarios
            ground_truth.get(
                "corroborated_value"
            ),

            # Legitimate-update scenarios
            ground_truth.get(
                "final_value"
            ),

            # Conflict-specific winning values
            ground_truth.get(
                "winning_priority"
            ),
            ground_truth.get(
                "winning_category"
            ),
            ground_truth.get(
                "winning_assignment_group"
            ),
            ground_truth.get(
                "winning_state"
            ),
            ground_truth.get(
                "winning_urgency"
            ),
            ground_truth.get(
                "winning_impact"
            ),
            ground_truth.get(
                "winning_resolved_by"
            ),
            ground_truth.get(
                "winning_opened_date"
            ),
        ])

        correct_resolution = (
            ground_truth.get(
                "correct_resolution"
            )
        )

        if (
            correct_resolution is not None
            and str(
                correct_resolution
            ).strip()
        ):
            candidates.append(
                correct_resolution
            )

    candidates.extend([
        scenario.get(
            "reference_value"
        ),
        scenario.get(
            "correct_value"
        ),
        scenario.get(
            "final_value"
        ),
    ])

    for value in candidates:
        if (
            value is not None
            and not isinstance(
                value,
                (
                    dict,
                    list,
                )
            )
            and str(value).strip()
        ):
            return str(value)

    return None


def get_observations(scenario):
    for key in [
        "turns",
        "observations",
        "reports",
    ]:
        value = scenario.get(
            key
        )

        if isinstance(
            value,
            list
        ):
            return value

    return []


def get_agent(observation):
    for key in [
        "agent_id",
        "agent",
        "source_agent",
    ]:
        value = observation.get(
            key
        )

        if value:
            return str(value)

    return None


def get_fact_type(
    observation,
    scenario
):
    value = observation.get(
        "fact_type"
    )

    if value:
        return str(value)

    value = scenario.get(
        "fact_type"
    )

    if value:
        return str(value)

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
            return str(value)

    return None


def get_value(observation):
    for key in [
        "value",
        "raw_value",
        "claim_value",
    ]:
        value = observation.get(
            key
        )

        if value is not None:
            return str(value)

    return None


def scenario_identifier(
    scenario,
    file_path,
    index
):
    for key in [
        "scenario_id",
        "id",
        "name",
    ]:
        value = scenario.get(
            key
        )

        if value:
            return (
                f"{file_path.name}:"
                f"{value}"
            )

    return (
        f"{file_path.name}:"
        f"{index}"
    )


def fold_for_scenario(
    scenario_id,
    folds
):
    digest = hashlib.sha256(
        scenario_id.encode(
            "utf-8"
        )
    ).hexdigest()

    number = int(
        digest[:12],
        16
    )

    return number % folds


def collect_records(
    input_dir,
    folds
):
    records = []

    skipped = defaultdict(int)

    files = sorted(
        input_dir.rglob(
            "*.json"
        )
    )

    usable_files = []

    for path in files:
        lowered = str(path).lower()

        # Do not accidentally calibrate
        # priors from synthetic V2 data.
        if (
            "v2_review" in lowered
            or "multiseed" in lowered
        ):
            continue

        usable_files.append(
            path
        )

    for file_path in usable_files:

        data = load_json(
            file_path
        )

        for index, scenario in enumerate(
            scenario_list(data)
        ):
            if not isinstance(
                scenario,
                dict
            ):
                continue

            truth = get_truth(
                scenario
            )

            if truth is None:
                skipped[
                    "missing_truth"
                ] += 1
                continue

            observations = (
                get_observations(
                    scenario
                )
            )

            if not observations:
                skipped[
                    "missing_observations"
                ] += 1
                continue

            scenario_id = (
                scenario_identifier(
                    scenario,
                    file_path,
                    index
                )
            )

            fold = (
                fold_for_scenario(
                    scenario_id,
                    folds
                )
            )

            normalized_truth = (
                normalize_value(
                    truth
                )
            )

            # One contribution per
            # agent/fact/scenario.
            # If an agent appears repeatedly,
            # retain its final report.
            deduplicated = {}

            all_values = set()
            all_agents = set()

            for observation in observations:

                if not isinstance(
                    observation,
                    dict
                ):
                    continue

                agent = get_agent(
                    observation
                )

                fact_type = (
                    get_fact_type(
                        observation,
                        scenario
                    )
                )

                value = get_value(
                    observation
                )

                if (
                    agent is None
                    or fact_type is None
                    or value is None
                ):
                    skipped[
                        "incomplete_observation"
                    ] += 1
                    continue

                all_values.add(
                    normalize_value(
                        value
                    )
                )

                all_agents.add(
                    normalize_value(
                        agent
                    )
                )

                deduplicated[
                    (
                        agent,
                        fact_type,
                    )
                ] = {
                    "agent_id":
                        agent,

                    "fact_type":
                        fact_type,

                    "value":
                        value,
                }

            # Some old datasets use
            # correct_resolution to mean
            # winner agent rather than value.
            if (
                normalized_truth
                in all_agents
                and normalized_truth
                not in all_values
            ):
                skipped[
                    "truth_looks_like_agent"
                ] += 1
                continue

            for observation in (
                deduplicated.values()
            ):
                correct = (
                    normalize_value(
                        observation[
                            "value"
                        ]
                    )
                    == normalized_truth
                )

                records.append({
                    "scenario_id":
                        scenario_id,

                    "fold":
                        fold,

                    "agent_id":
                        observation[
                            "agent_id"
                        ],

                    "fact_type":
                        observation[
                            "fact_type"
                        ],

                    "value":
                        observation[
                            "value"
                        ],

                    "truth":
                        truth,

                    "correct":
                        1
                        if correct
                        else 0,

                    "source_file":
                        str(
                            file_path
                        ),
                })

    return (
        records,
        usable_files,
        dict(skipped),
    )


def count_by(
    records,
    key_function
):
    counts = defaultdict(
        lambda: {
            "successes": 0,
            "failures": 0,
        }
    )

    for record in records:
        key = key_function(
            record
        )

        if record["correct"]:
            counts[key][
                "successes"
            ] += 1
        else:
            counts[key][
                "failures"
            ] += 1

    return counts


def beta_posterior_mean(
    prior,
    successes,
    failures,
    prior_strength
):
    total = (
        successes
        + failures
    )

    return (
        (
            prior_strength
            * prior
        )
        + successes
    ) / (
        prior_strength
        + total
    )


def neutral_posterior(
    successes,
    failures,
    strength=2.0
):
    total = (
        successes
        + failures
    )

    return (
        0.5 * strength
        + successes
    ) / (
        strength
        + total
    )


def wilson_interval(
    successes,
    total,
    z=1.96
):
    if total == 0:
        return (
            None,
            None,
        )

    p = successes / total

    denominator = (
        1.0
        + (
            z * z
            / total
        )
    )

    centre = (
        p
        + (
            z * z
            / (
                2.0
                * total
            )
        )
    )

    adjustment = (
        z
        * math.sqrt(
            (
                p * (
                    1.0 - p
                )
                / total
            )
            + (
                z * z
                / (
                    4.0
                    * total
                    * total
                )
            )
        )
    )

    low = (
        centre - adjustment
    ) / denominator

    high = (
        centre + adjustment
    ) / denominator

    return (
        low,
        high,
    )


def clip_probability(p):
    return max(
        1e-6,
        min(
            1.0 - 1e-6,
            p
        )
    )


def metric_summary(
    predictions
):
    if not predictions:
        return {
            "n": 0,
            "brier": None,
            "log_loss": None,
        }

    brier = sum(
        (
            p - y
        ) ** 2
        for p, y
        in predictions
    ) / len(
        predictions
    )

    log_loss = 0.0

    for p, y in predictions:
        p = clip_probability(
            p
        )

        if y == 1:
            log_loss += (
                -math.log(p)
            )
        else:
            log_loss += (
                -math.log(
                    1.0 - p
                )
            )

    log_loss /= len(
        predictions
    )

    return {
        "n":
            len(
                predictions
            ),

        "brier":
            brier,

        "log_loss":
            log_loss,
    }


def cross_validate(
    records,
    folds,
    prior_strength
):
    predictions = {
        "uniform": [],
        "agent_only": [],
        "fact_only": [],
        "expert_prior": [],
        "agent_fact_posterior": [],
    }

    for fold in range(
        folds
    ):
        train = [
            row
            for row in records
            if row["fold"] != fold
        ]

        test = [
            row
            for row in records
            if row["fold"] == fold
        ]

        agent_counts = (
            count_by(
                train,
                lambda row:
                    row[
                        "agent_id"
                    ]
            )
        )

        fact_counts = (
            count_by(
                train,
                lambda row:
                    row[
                        "fact_type"
                    ]
            )
        )

        agent_fact_counts = (
            count_by(
                train,
                lambda row: (
                    row[
                        "agent_id"
                    ],
                    row[
                        "fact_type"
                    ],
                )
            )
        )

        for row in test:

            y = row[
                "correct"
            ]

            agent = row[
                "agent_id"
            ]

            fact_type = row[
                "fact_type"
            ]

            predictions[
                "uniform"
            ].append(
                (
                    0.5,
                    y,
                )
            )

            agent_stat = (
                agent_counts.get(
                    agent,
                    {
                        "successes": 0,
                        "failures": 0,
                    }
                )
            )

            agent_probability = (
                neutral_posterior(
                    agent_stat[
                        "successes"
                    ],
                    agent_stat[
                        "failures"
                    ],
                )
            )

            predictions[
                "agent_only"
            ].append(
                (
                    agent_probability,
                    y,
                )
            )

            fact_stat = (
                fact_counts.get(
                    fact_type,
                    {
                        "successes": 0,
                        "failures": 0,
                    }
                )
            )

            fact_probability = (
                neutral_posterior(
                    fact_stat[
                        "successes"
                    ],
                    fact_stat[
                        "failures"
                    ],
                )
            )

            predictions[
                "fact_only"
            ].append(
                (
                    fact_probability,
                    y,
                )
            )

            expert_prior = (
                get_expert_prior(
                    agent,
                    fact_type
                )
            )

            predictions[
                "expert_prior"
            ].append(
                (
                    expert_prior,
                    y,
                )
            )

            pair_stat = (
                agent_fact_counts.get(
                    (
                        agent,
                        fact_type,
                    ),
                    {
                        "successes": 0,
                        "failures": 0,
                    }
                )
            )

            posterior = (
                beta_posterior_mean(
                    prior=(
                        expert_prior
                    ),
                    successes=(
                        pair_stat[
                            "successes"
                        ]
                    ),
                    failures=(
                        pair_stat[
                            "failures"
                        ]
                    ),
                    prior_strength=(
                        prior_strength
                    ),
                )
            )

            predictions[
                "agent_fact_posterior"
            ].append(
                (
                    posterior,
                    y,
                )
            )

    return {
        name:
            metric_summary(
                values
            )
        for name, values
        in predictions.items()
    }


def final_matrix(
    records,
    prior_strength
):
    counts = count_by(
        records,
        lambda row: (
            row["agent_id"],
            row["fact_type"],
        )
    )

    agents = set(
        EXPERT_PRIORS.keys()
    )

    facts = set()

    for (
        agent_id,
        fact_type
    ) in counts.keys():
        agents.add(
            agent_id
        )

        facts.add(
            fact_type
        )

    for (
        agent_id,
        mapping
    ) in EXPERT_PRIORS.items():
        for fact_type in mapping:
            if fact_type != "default":
                facts.add(
                    fact_type
                )

    rows = []

    matrix = {}

    for agent_id in sorted(
        agents
    ):
        matrix[
            agent_id
        ] = {}

        for fact_type in sorted(
            facts
        ):
            stat = counts.get(
                (
                    agent_id,
                    fact_type,
                ),
                {
                    "successes": 0,
                    "failures": 0,
                }
            )

            successes = stat[
                "successes"
            ]

            failures = stat[
                "failures"
            ]

            total = (
                successes
                + failures
            )

            expert_prior = (
                get_expert_prior(
                    agent_id,
                    fact_type
                )
            )

            posterior = (
                beta_posterior_mean(
                    prior=expert_prior,
                    successes=successes,
                    failures=failures,
                    prior_strength=(
                        prior_strength
                    ),
                )
            )

            empirical = (
                successes / total
                if total
                else None
            )

            (
                ci_low,
                ci_high,
            ) = wilson_interval(
                successes,
                total,
            )

            if total == 0:
                evidence_band = (
                    "expert_only"
                )

            elif total < 5:
                evidence_band = (
                    "n_1_to_4"
                )

            elif total < 10:
                evidence_band = (
                    "n_5_to_9"
                )

            else:
                evidence_band = (
                    "n_10_plus"
                )

            matrix[
                agent_id
            ][fact_type] = (
                round(
                    posterior,
                    6
                )
            )

            rows.append({
                "agent_id":
                    agent_id,

                "fact_type":
                    fact_type,

                "expert_prior":
                    expert_prior,

                "successes":
                    successes,

                "failures":
                    failures,

                "n":
                    total,

                "empirical_accuracy":
                    empirical,

                "wilson_95_low":
                    ci_low,

                "wilson_95_high":
                    ci_high,

                "posterior_m5":
                    posterior,

                "evidence_band":
                    evidence_band,
            })

    return (
        rows,
        matrix,
    )


def main():
    args = parse_args()

    input_dir = Path(
        args.input_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        records,
        files,
        skipped,
    ) = collect_records(
        input_dir,
        args.folds,
    )

    if not records:
        raise RuntimeError(
            "No usable labelled observations "
            "were found."
        )

    scenario_count = len({
        row[
            "scenario_id"
        ]
        for row in records
    })

    print(
        "Agent × Fact-Type "
        "Reliability Calibration"
    )

    print(
        "Input:",
        input_dir
    )

    print(
        "JSON files inspected:",
        len(files)
    )

    print(
        "Usable scenarios:",
        scenario_count
    )

    print(
        "Usable source observations:",
        len(records)
    )

    print(
        "Prior strength:",
        args.prior_strength
    )

    print(
        "Cross-validation folds:",
        args.folds
    )

    print(
        "Skipped:",
        skipped
    )

    cv = cross_validate(
        records=records,
        folds=args.folds,
        prior_strength=(
            args.prior_strength
        ),
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUT-OF-FOLD RELIABILITY "
        "CALIBRATION"
    )

    print(
        "Lower Brier and lower "
        "log-loss are better."
    )

    print(
        "=" * 80
    )

    for (
        name,
        result
    ) in cv.items():

        print(
            f"{name:<24} "
            f"n={result['n']:<5} "
            f"Brier="
            f"{result['brier']:.4f} "
            f"LogLoss="
            f"{result['log_loss']:.4f}"
        )

    (
        rows,
        matrix,
    ) = final_matrix(
        records,
        args.prior_strength,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL AGENT × FACT "
        "POSTERIOR ESTIMATES"
    )

    print(
        "=" * 80
    )

    for row in rows:
        if row["n"] == 0:
            continue

        empirical = (
            f"{row['empirical_accuracy']:.3f}"
            if (
                row[
                    "empirical_accuracy"
                ]
                is not None
            )
            else "N/A"
        )

        print(
            f"{row['agent_id']:<20} "
            f"{row['fact_type']:<20} "
            f"N={row['n']:<3} "
            f"raw={empirical:<6} "
            f"expert="
            f"{row['expert_prior']:.3f} "
            f"posterior="
            f"{row['posterior_m5']:.3f}"
        )

    csv_path = (
        output_dir
        / "agent_fact_prior_calibration.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    json_path = (
        output_dir
        / "agent_fact_prior_calibration.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "input_dir":
                    str(input_dir),

                "prior_strength":
                    args.prior_strength,

                "folds":
                    args.folds,

                "scenario_count":
                    scenario_count,

                "observation_count":
                    len(records),

                "skipped":
                    skipped,

                "cross_validation":
                    cv,

                "posterior_matrix":
                    matrix,

                "rows":
                    rows,
            },
            file,
            indent=2
        )

    print(
        "\nSaved CSV:",
        csv_path
    )

    print(
        "Saved JSON:",
        json_path
    )


if __name__ == "__main__":
    main()