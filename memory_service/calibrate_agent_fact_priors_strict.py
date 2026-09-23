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


# These scenario families are deliberately excluded from
# SOURCE-RELIABILITY learning.
EXCLUDED_SOURCE_TRUST_CATEGORIES = {
    "category_D_legitimate_update.json":
        "temporal_update",

    "category_F_stale_source.json":
        "temporal_staleness",

    "category_I_no_contradiction.json":
        "no_independent_truth",
}


TRUTH_FIELD_TO_FACT = {
    "winning_priority":
        "priority",

    "winning_category":
        "category",

    "winning_assignment_group":
        "assignment_group",

    "winning_state":
        "state",

    "winning_urgency":
        "urgency",

    "winning_impact":
        "impact",

    "winning_resolved_by":
        "resolved_by",

    "winning_opened_date":
        "opened_date",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "memory_service/"
            "results_agent_fact_calibration_strict"
        ),
        type=str,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--prior-strength",
        type=float,
        default=5.0,
    )

    return parser.parse_args()


def canonical_agent(value):
    if value is None:
        return None

    key = str(
        value
    ).strip().lower()

    return AGENT_ALIASES.get(
        key,
        key,
    )


def canonical_fact(value):
    if value is None:
        return None

    key = str(
        value
    ).strip().lower()

    return FACT_TYPE_ALIASES.get(
        key,
        key,
    )


def load_json(path):
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(
                file
            )

    except Exception:
        return None


def scenario_list(data):
    if data is None:
        return []

    if isinstance(
        data,
        list,
    ):
        return data

    if isinstance(
        data,
        dict,
    ):
        if isinstance(
            data.get(
                "scenarios"
            ),
            list,
        ):
            return data[
                "scenarios"
            ]

        return [data]

    return []


def get_observations(
    scenario
):
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
            list,
        ):
            return value

    return []


def get_agent(
    observation
):
    for key in [
        "agent_id",
        "agent",
        "source_agent",
    ]:
        value = observation.get(
            key
        )

        if value:
            return canonical_agent(
                value
            )

    return None


def get_fact_type(
    observation,
    scenario,
):
    value = observation.get(
        "fact_type"
    )

    if value:
        return canonical_fact(
            value
        )

    value = scenario.get(
        "fact_type"
    )

    if value:
        return canonical_fact(
            value
        )

    ground_truth = scenario.get(
        "ground_truth",
        {}
    )

    if isinstance(
        ground_truth,
        dict,
    ):
        value = ground_truth.get(
            "fact_type"
        )

        if value:
            return canonical_fact(
                value
            )

    return None


def get_value(
    observation
):
    for key in [
        "value",
        "raw_value",
        "claim_value",
    ]:
        value = observation.get(
            key
        )

        if value is not None:
            return str(
                value
            )

    return None


def scenario_identifier(
    scenario,
    file_path,
    index,
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
    folds,
):
    digest = hashlib.sha256(
        scenario_id.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        int(
            digest[:12],
            16,
        )
        % folds
    )


def infer_truth_and_fact(
    scenario,
    observations,
):
    ground_truth = scenario.get(
        "ground_truth",
        {}
    )

    if not isinstance(
        ground_truth,
        dict,
    ):
        ground_truth = {}

    # ----------------------------------
    # Explicit fact-specific truth fields
    # ----------------------------------

    for (
        truth_field,
        fact_type,
    ) in TRUTH_FIELD_TO_FACT.items():

        value = ground_truth.get(
            truth_field
        )

        if (
            value is not None
            and str(value).strip()
        ):
            return (
                str(value),
                canonical_fact(
                    fact_type
                ),
                truth_field,
            )

    # ----------------------------------
    # Explicit generic truth fields
    # ----------------------------------

    candidate_fields = [
        "correct_value",
        "reference_value",
        "truth",
        "corroborated_value",
    ]

    for field in candidate_fields:

        value = ground_truth.get(
            field
        )

        if (
            value is None
            or not str(value).strip()
        ):
            continue

        truth = str(
            value
        )

        # Prefer explicit scenario fact type.
        fact_type = (
            scenario.get(
                "fact_type"
            )
            or ground_truth.get(
                "fact_type"
            )
        )

        if fact_type:
            return (
                truth,
                canonical_fact(
                    fact_type
                ),
                field,
            )

        # Otherwise infer the fact type
        # from observations whose value
        # matches the labelled truth.
        matching_facts = set()

        for observation in observations:

            if not isinstance(
                observation,
                dict,
            ):
                continue

            obs_value = get_value(
                observation
            )

            obs_fact = get_fact_type(
                observation,
                scenario,
            )

            if (
                obs_value is not None
                and obs_fact is not None
                and normalize_value(
                    obs_value
                )
                == normalize_value(
                    truth
                )
            ):
                matching_facts.add(
                    obs_fact
                )

        if len(
            matching_facts
        ) == 1:
            return (
                truth,
                next(
                    iter(
                        matching_facts
                    )
                ),
                field,
            )

    # ----------------------------------
    # correct_resolution is only usable
    # if it clearly represents a value,
    # not an agent identifier.
    # ----------------------------------

    value = ground_truth.get(
        "correct_resolution"
    )

    if (
        value is not None
        and str(value).strip()
    ):
        truth = str(value)

        agents = {
            canonical_agent(
                get_agent(
                    observation
                )
            )
            for observation
            in observations
            if isinstance(
                observation,
                dict,
            )
        }

        if (
            canonical_agent(
                truth
            )
            not in agents
        ):
            matching_facts = set()

            for observation in observations:

                if not isinstance(
                    observation,
                    dict,
                ):
                    continue

                obs_value = get_value(
                    observation
                )

                obs_fact = get_fact_type(
                    observation,
                    scenario,
                )

                if (
                    obs_value is not None
                    and obs_fact is not None
                    and normalize_value(
                        obs_value
                    )
                    == normalize_value(
                        truth
                    )
                ):
                    matching_facts.add(
                        obs_fact
                    )

            if len(
                matching_facts
            ) == 1:
                return (
                    truth,
                    next(
                        iter(
                            matching_facts
                        )
                    ),
                    "correct_resolution",
                )

    # Top-level fallback.
    for field in [
        "reference_value",
        "correct_value",
    ]:
        value = scenario.get(
            field
        )

        if (
            value is not None
            and str(value).strip()
        ):
            truth = str(
                value
            )

            matching_facts = set()

            for observation in observations:

                if not isinstance(
                    observation,
                    dict,
                ):
                    continue

                obs_value = get_value(
                    observation
                )

                obs_fact = get_fact_type(
                    observation,
                    scenario,
                )

                if (
                    obs_value is not None
                    and obs_fact is not None
                    and normalize_value(
                        obs_value
                    )
                    == normalize_value(
                        truth
                    )
                ):
                    matching_facts.add(
                        obs_fact
                    )

            if len(
                matching_facts
            ) == 1:
                return (
                    truth,
                    next(
                        iter(
                            matching_facts
                        )
                    ),
                    field,
                )

    return (
        None,
        None,
        None,
    )


def collect_records(
    input_dir,
    folds,
):
    records = []

    skipped = defaultdict(
        int
    )

    excluded = defaultdict(
        int
    )

    inspected_files = []

    for file_path in sorted(
        input_dir.rglob(
            "*.json"
        )
    ):
        lowered = str(
            file_path
        ).lower()

        if (
            "v2_review" in lowered
            or "multiseed" in lowered
        ):
            continue

        inspected_files.append(
            file_path
        )

        exclusion_reason = (
            EXCLUDED_SOURCE_TRUST_CATEGORIES.get(
                file_path.name
            )
        )

        data = load_json(
            file_path
        )

        for index, scenario in enumerate(
            scenario_list(data)
        ):

            if not isinstance(
                scenario,
                dict,
            ):
                continue

            if exclusion_reason:
                excluded[
                    exclusion_reason
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

            (
                truth,
                target_fact,
                truth_source,
            ) = infer_truth_and_fact(
                scenario,
                observations,
            )

            if (
                truth is None
                or target_fact is None
            ):
                skipped[
                    "missing_or_ambiguous_truth"
                ] += 1
                continue

            scenario_id = (
                scenario_identifier(
                    scenario,
                    file_path,
                    index,
                )
            )

            fold = (
                fold_for_scenario(
                    scenario_id,
                    folds,
                )
            )

            truth_normalized = (
                normalize_value(
                    truth
                )
            )

            # Only observations about the
            # labelled target fact contribute
            # to source-reliability learning.
            deduplicated = {}

            for observation in observations:

                if not isinstance(
                    observation,
                    dict,
                ):
                    continue

                agent = get_agent(
                    observation
                )

                fact_type = get_fact_type(
                    observation,
                    scenario,
                )

                value = get_value(
                    observation
                )

                if (
                    agent is None
                    or fact_type is None
                    or value is None
                ):
                    continue

                if (
                    fact_type
                    != target_fact
                ):
                    continue

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

            if not deduplicated:
                skipped[
                    "no_target_fact_observations"
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
                    == truth_normalized
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

                    "truth_source":
                        truth_source,

                    "correct":
                        1
                        if correct
                        else 0,

                    "source_file":
                        file_path.name,
                })

    return (
        records,
        inspected_files,
        dict(skipped),
        dict(excluded),
    )


def count_by(
    records,
    key_function,
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

        if record[
            "correct"
        ]:
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
    prior_strength,
):
    return (
        prior_strength
        * prior
        + successes
    ) / (
        prior_strength
        + successes
        + failures
    )


def neutral_posterior(
    successes,
    failures,
    strength=2.0,
):
    return (
        0.5 * strength
        + successes
    ) / (
        strength
        + successes
        + failures
    )


def clip_probability(p):
    return max(
        1e-6,
        min(
            1.0 - 1e-6,
            p,
        ),
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
            probability
            - truth
        ) ** 2
        for probability, truth
        in predictions
    ) / len(
        predictions
    )

    log_loss = 0.0

    for probability, truth in predictions:

        probability = (
            clip_probability(
                probability
            )
        )

        if truth:
            log_loss += (
                -math.log(
                    probability
                )
            )

        else:
            log_loss += (
                -math.log(
                    1.0
                    - probability
                )
            )

    log_loss /= len(
        predictions
    )

    return {
        "n":
            len(predictions),

        "brier":
            brier,

        "log_loss":
            log_loss,
    }


def cross_validate(
    records,
    folds,
    prior_strength,
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
            if row[
                "fold"
            ] != fold
        ]

        test = [
            row
            for row in records
            if row[
                "fold"
            ] == fold
        ]

        agent_counts = (
            count_by(
                train,
                lambda row:
                    row[
                        "agent_id"
                    ],
            )
        )

        fact_counts = (
            count_by(
                train,
                lambda row:
                    row[
                        "fact_type"
                    ],
            )
        )

        pair_counts = (
            count_by(
                train,
                lambda row: (
                    row[
                        "agent_id"
                    ],
                    row[
                        "fact_type"
                    ],
                ),
            )
        )

        for row in test:

            y = row[
                "correct"
            ]

            agent = row[
                "agent_id"
            ]

            fact = row[
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
                    },
                )
            )

            predictions[
                "agent_only"
            ].append(
                (
                    neutral_posterior(
                        agent_stat[
                            "successes"
                        ],
                        agent_stat[
                            "failures"
                        ],
                    ),
                    y,
                )
            )

            fact_stat = (
                fact_counts.get(
                    fact,
                    {
                        "successes": 0,
                        "failures": 0,
                    },
                )
            )

            predictions[
                "fact_only"
            ].append(
                (
                    neutral_posterior(
                        fact_stat[
                            "successes"
                        ],
                        fact_stat[
                            "failures"
                        ],
                    ),
                    y,
                )
            )

            prior = (
                get_expert_prior(
                    agent,
                    fact,
                )
            )

            predictions[
                "expert_prior"
            ].append(
                (
                    prior,
                    y,
                )
            )

            pair_stat = (
                pair_counts.get(
                    (
                        agent,
                        fact,
                    ),
                    {
                        "successes": 0,
                        "failures": 0,
                    },
                )
            )

            posterior = (
                beta_posterior_mean(
                    prior=prior,
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
    prior_strength,
):
    counts = count_by(
        records,
        lambda row: (
            row[
                "agent_id"
            ],
            row[
                "fact_type"
            ],
        ),
    )

    rows = []

    for (
        agent,
        fact,
    ) in sorted(
        counts.keys()
    ):

        stats = counts[
            (
                agent,
                fact,
            )
        ]

        successes = stats[
            "successes"
        ]

        failures = stats[
            "failures"
        ]

        n = (
            successes
            + failures
        )

        prior = (
            get_expert_prior(
                agent,
                fact,
            )
        )

        posterior = (
            beta_posterior_mean(
                prior=prior,
                successes=successes,
                failures=failures,
                prior_strength=(
                    prior_strength
                ),
            )
        )

        raw = (
            successes / n
            if n
            else None
        )

        rows.append({
            "agent_id":
                agent,

            "fact_type":
                fact,

            "successes":
                successes,

            "failures":
                failures,

            "n":
                n,

            "raw_accuracy":
                raw,

            "expert_prior":
                prior,

            "posterior":
                posterior,
        })

    return rows


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
        exist_ok=True,
    )

    (
        records,
        files,
        skipped,
        excluded,
    ) = collect_records(
        input_dir,
        args.folds,
    )

    if not records:
        raise RuntimeError(
            "No usable strict source-trust "
            "records were found."
        )

    scenarios = {
        row[
            "scenario_id"
        ]
        for row in records
    }

    print(
        "STRICT AGENT × FACT "
        "SOURCE-TRUST CALIBRATION"
    )

    print(
        "Files inspected:",
        len(files)
    )

    print(
        "Usable source-trust scenarios:",
        len(scenarios)
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
        "Skipped:",
        skipped
    )

    print(
        "Excluded by design:",
        excluded
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
        "OUT-OF-FOLD CALIBRATION"
    )

    print(
        "Lower Brier and log-loss "
        "are better."
    )

    print(
        "=" * 80
    )

    for (
        name,
        result,
    ) in cv.items():

        print(
            f"{name:<24} "
            f"n={result['n']:<4} "
            f"Brier="
            f"{result['brier']:.4f} "
            f"LogLoss="
            f"{result['log_loss']:.4f}"
        )

    rows = final_matrix(
        records,
        args.prior_strength,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "AGENT × FACT POSTERIORS"
    )

    print(
        "=" * 80
    )

    for row in rows:

        print(
            f"{row['agent_id']:<20} "
            f"{row['fact_type']:<20} "
            f"N={row['n']:<3} "
            f"raw="
            f"{row['raw_accuracy']:.3f} "
            f"prior="
            f"{row['expert_prior']:.3f} "
            f"posterior="
            f"{row['posterior']:.3f}"
        )

    output = {
        "method":
            (
                "Strict epistemic source-trust "
                "calibration"
            ),

        "prior_strength":
            args.prior_strength,

        "excluded_categories":
            EXCLUDED_SOURCE_TRUST_CATEGORIES,

        "usable_scenarios":
            len(scenarios),

        "usable_observations":
            len(records),

        "skipped":
            skipped,

        "excluded":
            excluded,

        "cross_validation":
            cv,

        "rows":
            rows,
    }

    json_path = (
        output_dir
        / "strict_agent_fact_calibration.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    csv_path = (
        output_dir
        / "strict_agent_fact_calibration.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        "\nSaved JSON:",
        json_path
    )

    print(
        "Saved CSV:",
        csv_path
    )


if __name__ == "__main__":
    main()