import json

from collections import Counter
from datetime import datetime
from pathlib import Path
import statistics
import math
from verimem_core.providers import (
    FixedTrustProvider,
    InMemoryTrustProvider,
    OracleTrustProvider,
)

from verimem_core.resolver import (
    normalize_value,
    resolve,
)

from verimem_core.types import (
    Observation,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_DIR = (
    ROOT
    / "dataset"
    / "v2_review"
)

RELIABILITY_PATH = (
    DATA_DIR
    / "dev_reliability.json"
)

STRESS_PATH = (
    DATA_DIR
    / "dev_stress.json"
)

METADATA_PATH = (
    DATA_DIR
    / "metadata.json"
)

RESULTS_DIR = (
    ROOT
    / "memory_service"
    / "results_v2_review"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# These are DEV-only trust-prior-strength candidates.
# They do NOT alter the frozen VeriMem confidence weights.
PRIOR_STRENGTHS = [
    5.0,
    10.0,
    20.0,
]


# Number of verified adaptation SCENARIOS.
# Each scenario contains reports from multiple agents.
LEARNING_CHECKPOINTS = [
    0,
    25,
    50,
    100,
    200,
    300,
]


CORROBORATION_METHODS = [
    "noisy_or",
    "log_odds",
]


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def normalize(value):
    if value is None:
        return None

    return normalize_value(value)


def to_observations(scenario):
    observations = []

    for item in scenario["observations"]:
        observations.append(
            Observation(
                fact_id=item["fact_id"],
                agent_id=item["agent_id"],
                fact_type=item["fact_type"],
                value=item["value"],
                extraction_type=item[
                    "extraction_type"
                ],
                observed_at=(
                    datetime.fromisoformat(
                        item["observed_at"]
                    )
                ),
            )
        )

    return observations


def is_correct(
    prediction,
    reference
):
    if (
        prediction is None
        or reference is None
    ):
        return False

    return (
        normalize(prediction)
        == normalize(reference)
    )


# ---------------------------------------------------------
# SIMPLE BASELINES
# ---------------------------------------------------------

def lww_prediction(scenario):
    latest = max(
        scenario["observations"],
        key=lambda item:
            item["observed_at"],
    )

    return latest["value"]


def vote_prediction(scenario):
    counts = Counter(
        normalize(item["value"])
        for item
        in scenario["observations"]
    )

    ranked = counts.most_common()

    if not ranked:
        return None

    # Plain vote abstains on an exact vote tie.
    if (
        len(ranked) > 1
        and ranked[0][1]
        == ranked[1][1]
    ):
        return None

    winner_key = ranked[0][0]

    for item in scenario[
        "observations"
    ]:
        if (
            normalize(item["value"])
            == winner_key
        ):
            return item["value"]

    return None


# ---------------------------------------------------------
# VERIFIED-OUTCOME TRUST ADAPTATION
# ---------------------------------------------------------

def train_provider(
    scenarios,
    count
):
    """
    Learn contextual reliability from externally
    verified outcomes.

    IMPORTANT:
    An automatic VeriMem decision is NEVER used as
    its own training label here.

    Each verified scenario reveals the reference truth.
    Every source report in that scenario then contributes
    one source-level correctness observation.
    """

    provider = (
        InMemoryTrustProvider()
    )

    selected = scenarios[:count]

    for scenario in selected:
        truth = normalize(
            scenario[
                "reference_value"
            ]
        )

        for observation in (
            scenario["observations"]
        ):
            correct = (
                normalize(
                    observation["value"]
                )
                == truth
            )

            provider.record_verified_outcome(
                agent_id=(
                    observation[
                        "agent_id"
                    ]
                ),
                fact_type=(
                    observation[
                        "fact_type"
                    ]
                ),
                correct=correct,
            )

    return provider


# ---------------------------------------------------------
# BASELINE EVALUATION
# ---------------------------------------------------------

def evaluate_baseline(
    scenarios,
    predictor
):
    total = 0
    correct = 0
    abstained = 0

    conflict_total = 0
    conflict_correct = 0
    conflict_abstained = 0

    for scenario in scenarios:
        prediction = predictor(
            scenario
        )

        total += 1

        if prediction is None:
            abstained += 1

        elif is_correct(
            prediction,
            scenario[
                "reference_value"
            ]
        ):
            correct += 1

        if scenario["is_conflict"]:
            conflict_total += 1

            if prediction is None:
                conflict_abstained += 1

            elif is_correct(
                prediction,
                scenario[
                    "reference_value"
                ]
            ):
                conflict_correct += 1

    decided = (
        total - abstained
    )

    conflict_decided = (
        conflict_total
        - conflict_abstained
    )

    return {
        "total":
            total,

        "correct":
            correct,

        "accuracy":
            (
                correct / total
                if total
                else 0.0
            ),

        "coverage":
            (
                decided / total
                if total
                else 0.0
            ),

        "selective_accuracy":
            (
                correct / decided
                if decided
                else 0.0
            ),

        "conflict_total":
            conflict_total,

        "conflict_correct":
            conflict_correct,

        "conflict_accuracy":
            (
                conflict_correct
                / conflict_total
                if conflict_total
                else 0.0
            ),

        "conflict_coverage":
            (
                conflict_decided
                / conflict_total
                if conflict_total
                else 0.0
            ),

        "conflict_selective_accuracy":
            (
                conflict_correct
                / conflict_decided
                if conflict_decided
                else 0.0
            ),
    }


# ---------------------------------------------------------
# VERIMEM CANDIDATE-SELECTION EVALUATION
# ---------------------------------------------------------

def evaluate_resolver(
    scenarios,
    provider,
    prior_strength,
    corroboration_method,
):
    """
    Measures top-ranked candidate-selection accuracy.

    Thresholds are deliberately disabled here because
    threshold calibration belongs to CAL, not DEV.

    Therefore this is NOT final auto-resolution accuracy.
    """

    total = 0
    correct = 0

    conflict_total = 0
    conflict_correct = 0

    conflict_margins = []

    exact_ties = 0

    for scenario in scenarios:
        observations = (
            to_observations(
                scenario
            )
        )

        result = resolve(
            observations=
                observations,

            provider=
                provider,

            # Disable auto-resolution thresholding.
            # Stage 3 is evaluating candidate ranking.
            min_winner_score=
                -999.0,

            min_margin=
                -999.0,

            corroboration_method=
                corroboration_method,

            prior_strength=
                prior_strength,
        )

        if (
            result.get("reason")
            == "tie"
        ):
            exact_ties += 1

        winner = result.get(
            "winner"
        )

        prediction = (
            winner["value"]
            if winner
            else None
        )

        total += 1

        correct_now = is_correct(
            prediction,
            scenario[
                "reference_value"
            ]
        )

        if correct_now:
            correct += 1

        if scenario["is_conflict"]:
            conflict_total += 1

            if correct_now:
                conflict_correct += 1

            margin = result.get(
                "margin"
            )

            if margin is not None:
                conflict_margins.append(
                    margin
                )

    return {
        "total":
            total,

        "correct":
            correct,

        "accuracy":
            (
                correct / total
                if total
                else 0.0
            ),

        "conflict_total":
            conflict_total,

        "conflict_correct":
            conflict_correct,

        "conflict_accuracy":
            (
                conflict_correct
                / conflict_total
                if conflict_total
                else 0.0
            ),

        "mean_conflict_margin":
            (
                sum(conflict_margins)
                / len(conflict_margins)
                if conflict_margins
                else 0.0
            ),

        "exact_tie_count":
            exact_ties,
    }


# ---------------------------------------------------------
# TRUST-CONVERGENCE DIAGNOSTICS
# ---------------------------------------------------------

def context_diagnostics(
    provider,
    true_reliabilities,
    prior_strength
):
    rows = []

    for agent_id, facts in (
        true_reliabilities.items()
    ):
        for fact_type, true_value in (
            facts.items()
        ):
            learned = (
                provider.get_trust(
                    agent_id=agent_id,
                    fact_type=fact_type,
                    prior_strength=(
                        prior_strength
                    ),
                )
            )

            rows.append({
                "agent_id":
                    agent_id,

                "fact_type":
                    fact_type,

                "true_reliability":
                    true_value,

                "prior":
                    learned["prior"],

                "learned":
                    learned[
                        "probability"
                    ],

                "evidence_count":
                    learned[
                        "evidence_count"
                    ],

                "absolute_error":
                    abs(
                        learned[
                            "probability"
                        ]
                        - true_value
                    ),
            })

    mae = (
        sum(
            row["absolute_error"]
            for row in rows
        )
        / len(rows)
    )

    return {
        "mean_absolute_error":
            mae,

        "contexts":
            rows,
    }


# ---------------------------------------------------------
# PRIOR-STRENGTH SANITY CHECK
# ---------------------------------------------------------

def prior_strength_sanity_check(
    adaptation,
):
    """
    Confirms that m is actually threaded into posterior trust.

    Uses a deliberately early checkpoint where prior strength
    should still have a visible effect.
    """

    provider = train_provider(
        adaptation,
        count=min(
            25,
            len(adaptation)
        ),
    )

    checks = []

    contexts = sorted(
        provider.snapshot().keys()
    )

    for (
        agent_id,
        fact_type
    ) in contexts:
        values = {}

        for strength in (
            PRIOR_STRENGTHS
        ):
            trust = (
                provider.get_trust(
                    agent_id=agent_id,
                    fact_type=fact_type,
                    prior_strength=
                        strength,
                )
            )

            values[
                str(strength)
            ] = trust[
                "probability"
            ]

        spread = (
            max(values.values())
            - min(values.values())
        )

        checks.append({
            "agent_id":
                agent_id,

            "fact_type":
                fact_type,

            "probabilities":
                values,

            "spread":
                spread,
        })

    max_spread = (
        max(
            (
                item["spread"]
                for item in checks
            ),
            default=0.0,
        )
    )

    return {
        "checkpoint_scenarios":
            min(
                25,
                len(adaptation)
            ),

        "max_context_probability_spread":
            max_spread,

        "contexts":
            checks,
    }


# ---------------------------------------------------------
# GOVERNANCE STRESS TEST
# ---------------------------------------------------------

def evaluate_stress(
    scenarios,
    provider,
    prior_strength,
    corroboration_method,
):
    category_stats = {
        "newer_correct": {
            "total": 0,
            "winner_correct": 0,
            "legacy_gap_contest": 0,
            "margin_sum": 0.0,
            "margin_count": 0,
        },

        "newer_wrong": {
            "total": 0,
            "winner_correct": 0,
            "legacy_gap_contest": 0,
            "margin_sum": 0.0,
            "margin_count": 0,
        },

        "ambiguous": {
            "total": 0,
            "winner_correct": 0,
            "legacy_gap_contest": 0,
            "margin_sum": 0.0,
            "margin_count": 0,
        },
    }

    for scenario in scenarios:
        category = scenario[
            "category"
        ]

        result = resolve(
            observations=
                to_observations(
                    scenario
                ),

            provider=
                provider,

            min_winner_score=
                -999.0,

            min_margin=
                -999.0,

            corroboration_method=
                corroboration_method,

            prior_strength=
                prior_strength,
        )

        stats = category_stats[
            category
        ]

        stats["total"] += 1

        winner = result.get(
            "winner"
        )

        winner_value = (
            winner["value"]
            if winner
            else None
        )

        if (
            scenario[
                "reference_value"
            ]
            is not None
            and is_correct(
                winner_value,
                scenario[
                    "reference_value"
                ]
            )
        ):
            stats[
                "winner_correct"
            ] += 1

        margin = result.get(
            "margin"
        )

        if margin is not None:
            stats[
                "margin_sum"
            ] += margin

            stats[
                "margin_count"
            ] += 1

            # Historical/pre-existing VeriMem gap.
            # This is a diagnostic only.
            if margin < 0.30:
                stats[
                    "legacy_gap_contest"
                ] += 1

    output = {}

    for (
        category,
        stats
    ) in category_stats.items():

        total = stats["total"]

        margin_count = stats[
            "margin_count"
        ]

        output[category] = {
            "total":
                total,

            "winner_accuracy":
                (
                    stats[
                        "winner_correct"
                    ]
                    / total
                    if (
                        total
                        and category
                        != "ambiguous"
                    )
                    else None
                ),

            "legacy_gap_contest_rate":
                (
                    stats[
                        "legacy_gap_contest"
                    ]
                    / total
                    if total
                    else 0.0
                ),

            "mean_margin":
                (
                    stats[
                        "margin_sum"
                    ]
                    / margin_count
                    if margin_count
                    else 0.0
                ),
        }

    return output


# ---------------------------------------------------------
# PRINT HELPERS
# ---------------------------------------------------------

def print_metric(
    name,
    result
):
    print(
        f"{name:<29}"
        f"all="
        f"{result['accuracy']:.3f}  "
        f"conflict="
        f"{result['conflict_accuracy']:.3f}"
    )


def print_stress_summary(
    stress_result
):
    newer_correct = (
        stress_result[
            "newer_correct"
        ]
    )

    newer_wrong = (
        stress_result[
            "newer_wrong"
        ]
    )

    ambiguous = (
        stress_result[
            "ambiguous"
        ]
    )

    print(
        "  newer-correct accuracy:",
        f"{newer_correct['winner_accuracy']:.3f}"
    )

    print(
        "  newer-wrong prevention:",
        f"{newer_wrong['winner_accuracy']:.3f}"
    )

    print(
        "  ambiguous <0.30 gap:",
        f"{ambiguous['legacy_gap_contest_rate']:.3f}"
    )
def summarize_seed_values(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    mean = statistics.mean(values)

    sd = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    margin = (
        1.96 * sd / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )

    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    reliability_data = load_json(
        RELIABILITY_PATH
    )

    stress_data = load_json(
        STRESS_PATH
    )

    metadata = (
        load_json(
            METADATA_PATH
        )
        if METADATA_PATH.exists()
        else {}
    )

    benchmark_seed = (
        reliability_data.get(
            "seed",
            metadata.get(
                "seed"
            )
        )
    )

    final_results = {
        "dataset":
            (
                "V2 controlled "
                "review-aligned DEV"
            ),

        "benchmark_seed":
            benchmark_seed,

        "important_interpretation": {
            "benchmark_type":
                (
                    "Controlled synthetic "
                    "mechanism benchmark "
                    "using ITSM-style "
                    "categorical values."
                ),

            "not_final_test":
                True,

            "candidate_accuracy_definition":
                (
                    "Accuracy of the "
                    "top-ranked candidate "
                    "before calibration of "
                    "automatic-resolution "
                    "thresholds."
                ),

            "adaptation_unit":
                (
                    "300 verified scenarios; "
                    "each source report in a "
                    "verified scenario yields "
                    "one source-level "
                    "correctness observation."
                ),

            "oracle_definition":
                (
                    "Oracle-source VeriMem "
                    "knows the generating "
                    "source reliability only. "
                    "It does not know the "
                    "correct candidate and "
                    "is not a strict finite-"
                    "sample upper bound."
                ),

            "trust_context":
                (
                    "Learned trust is "
                    "R(agent, fact_type). "
                    "Extraction directness "
                    "remains a separate "
                    "confidence component."
                ),
        },

        "confidence_weights": {
            "source":
                0.35,

            "corroboration":
                0.30,

            "directness":
                0.15,

            "time_penalty":
                -0.20,
        },

        "regimes": {},
    }

    print(
        "V2 REVIEW-ALIGNED STAGE 3"
    )

    print(
        "Controlled DEV benchmark only."
    )

    print(
        "CAL and TEST are not loaded."
    )

    print(
        "Benchmark seed:",
        benchmark_seed
    )

    print(
        "Metric: top-ranked "
        "candidate-selection accuracy."
    )

    print(
        "This is NOT final calibrated "
        "auto-resolution accuracy."
    )

    for regime in [
        "aligned",
        "mixed",
        "shifted",
    ]:
        print(
            "\n"
            + "=" * 70
        )

        print(
            f"REGIME: "
            f"{regime.upper()}"
        )

        regime_data = (
            reliability_data[
                "regimes"
            ][regime]
        )

        adaptation = (
            regime_data[
                "adaptation"
            ]
        )

        evaluation = (
            regime_data[
                "evaluation"
            ]
        )

        true_reliabilities = (
            regime_data[
                "true_source_reliabilities"
            ]
        )

        regime_results = {
            "adaptation_scenarios":
                len(adaptation),

            "evaluation_scenarios":
                len(evaluation),

            "baselines": {},

            "methods": {},

            "learning_curves": {},

            "context_diagnostics": {},

            "prior_strength_sanity":
                None,

            "stress": {},
        }

        # -----------------------------
        # Baselines
        # -----------------------------

        lww = evaluate_baseline(
            evaluation,
            lww_prediction
        )

        vote = evaluate_baseline(
            evaluation,
            vote_prediction
        )

        regime_results[
            "baselines"
        ]["last_write_wins"] = (
            lww
        )

        regime_results[
            "baselines"
        ]["plain_vote"] = vote

        print("\nBaselines")

        print_metric(
            "Last-write-wins",
            lww
        )

        print_metric(
            "Plain vote",
            vote
        )

        # -----------------------------
        # Fixed / Learned / Oracle-source
        # -----------------------------

        for method in (
            CORROBORATION_METHODS
        ):
            print(
                f"\nCorroboration: "
                f"{method}"
            )

            fixed_provider = (
                FixedTrustProvider()
            )

            fixed = evaluate_resolver(
                scenarios=
                    evaluation,

                provider=
                    fixed_provider,

                prior_strength=
                    20.0,

                corroboration_method=
                    method,
            )

            oracle_provider = (
                OracleTrustProvider(
                    true_reliabilities
                )
            )

            oracle_source = (
                evaluate_resolver(
                    scenarios=
                        evaluation,

                    provider=
                        oracle_provider,

                    prior_strength=
                        20.0,

                    corroboration_method=
                        method,
                )
            )

            method_results = {
                "fixed":
                    fixed,

                "learned":
                    {},

                "oracle_source":
                    oracle_source,
            }

            print_metric(
                "Fixed VeriMem",
                fixed
            )

            for prior_strength in (
                PRIOR_STRENGTHS
            ):
                learned_provider = (
                    train_provider(
                        adaptation,
                        count=300
                    )
                )

                learned = (
                    evaluate_resolver(
                        scenarios=
                            evaluation,

                        provider=
                            learned_provider,

                        prior_strength=
                            prior_strength,

                        corroboration_method=
                            method,
                    )
                )

                method_results[
                    "learned"
                ][
                    str(
                        prior_strength
                    )
                ] = learned

                print_metric(
                    (
                        "Learned VeriMem "
                        f"m="
                        f"{prior_strength:g}"
                    ),
                    learned
                )

            print_metric(
                "Oracle-source VeriMem",
                oracle_source
            )

            regime_results[
                "methods"
            ][method] = (
                method_results
            )

            # -------------------------
            # Learning curves
            # -------------------------

            curves_for_method = {}

            for prior_strength in (
                PRIOR_STRENGTHS
            ):
                points = []

                for checkpoint in (
                    LEARNING_CHECKPOINTS
                ):
                    provider = (
                        train_provider(
                            adaptation,
                            count=
                                checkpoint,
                        )
                    )

                    result = (
                        evaluate_resolver(
                            scenarios=
                                evaluation,

                            provider=
                                provider,

                            prior_strength=
                                prior_strength,

                            corroboration_method=
                                method,
                        )
                    )

                    points.append({
                        "verified_scenarios":
                            checkpoint,

                        "source_outcome_upper_bound":
                            (
                                checkpoint
                                * 3
                            ),

                        "accuracy":
                            result[
                                "accuracy"
                            ],

                        "conflict_accuracy":
                            result[
                                "conflict_accuracy"
                            ],
                    })

                curves_for_method[
                    str(
                        prior_strength
                    )
                ] = points

            regime_results[
                "learning_curves"
            ][method] = (
                curves_for_method
            )

        # -----------------------------
        # Trust convergence
        # -----------------------------

        diagnostic_provider = (
            train_provider(
                adaptation,
                count=300
            )
        )

        for prior_strength in (
            PRIOR_STRENGTHS
        ):
            diagnostics = (
                context_diagnostics(
                    provider=
                        diagnostic_provider,

                    true_reliabilities=
                        true_reliabilities,

                    prior_strength=
                        prior_strength,
                )
            )

            regime_results[
                "context_diagnostics"
            ][
                str(
                    prior_strength
                )
            ] = diagnostics

            print(
                "\nTrust convergence "
                f"m={prior_strength:g}: "
                "MAE="
                f"{diagnostics['mean_absolute_error']:.3f}"
            )

        # -----------------------------
        # Confirm m is threaded
        # -----------------------------

        m_sanity = (
            prior_strength_sanity_check(
                adaptation
            )
        )

        regime_results[
            "prior_strength_sanity"
        ] = m_sanity

        print(
            "Prior-strength sanity "
            "(25 scenarios), "
            "max context spread=",
            f"{m_sanity['max_context_probability_spread']:.4f}",
            sep=""
        )

        # -----------------------------
        # Stress benchmark
        # -----------------------------

        regime_stress = (
            stress_data[
                "regimes"
            ][regime]
        )

        for method in (
            CORROBORATION_METHODS
        ):
            regime_results[
                "stress"
            ][method] = {}

            for prior_strength in (
                PRIOR_STRENGTHS
            ):
                provider = (
                    train_provider(
                        adaptation,
                        count=300
                    )
                )

                stress_result = (
                    evaluate_stress(
                        scenarios=
                            regime_stress,

                        provider=
                            provider,

                        prior_strength=
                            prior_strength,

                        corroboration_method=
                            method,
                    )
                )

                regime_results[
                    "stress"
                ][method][
                    str(
                        prior_strength
                    )
                ] = stress_result

        print(
            "\nStress diagnostic "
            "(noisy_or, m=5)"
        )

        print_stress_summary(
            regime_results[
                "stress"
            ][
                "noisy_or"
            ][
                "5.0"
            ]
        )

        final_results[
            "regimes"
        ][regime] = (
            regime_results
        )

    output_path = (
        RESULTS_DIR
        / (
            "stage3_review_aligned_"
            "dev_results.json"
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            final_results,
            file,
            indent=2
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Saved:",
        output_path
    )

    print(
        "No CAL or TEST data "
        "was accessed."
    )


if __name__ == "__main__":
    main()