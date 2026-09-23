import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from verimem_core.types import Observation
from verimem_core.providers import (
    InMemoryTrustProvider,
)
from verimem_core.resolver import resolve
from verimem_core.config import (
    get_expert_prior,
)


ROOT = Path(__file__).resolve().parents[1]

DEV_PATH = (
    ROOT
    / "dataset"
    / "v2"
    / "dev.json"
)

RESULTS_DIR = (
    ROOT
    / "memory_service"
    / "results_v2"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def normalize(value):
    if value is None:
        return None

    return str(value).strip().lower()


def load_dev():
    with DEV_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)[
            "scenarios"
        ]


def to_observations(scenario):
    observations = []

    for item in scenario[
        "observations"
    ]:
        observations.append(
            Observation(
                fact_id=item["fact_id"],
                agent_id=item[
                    "agent_id"
                ],
                fact_type=item[
                    "fact_type"
                ],
                value=item["value"],
                extraction_type=item[
                    "extraction_type"
                ],
                observed_at=
                    datetime.fromisoformat(
                        item[
                            "observed_at"
                        ]
                    ),
            )
        )

    return observations


def lww_prediction(scenario):
    latest = max(
        scenario["observations"],
        key=lambda x: x[
            "observed_at"
        ]
    )

    return latest["value"]


def majority_prediction(scenario):
    counts = Counter(
        normalize(x["value"])
        for x in scenario[
            "observations"
        ]
    )

    ranked = counts.most_common()

    if (
        len(ranked) > 1
        and
        ranked[0][1] == ranked[1][1]
    ):
        return None

    winning_normalized = (
        ranked[0][0]
    )

    for observation in scenario[
        "observations"
    ]:
        if (
            normalize(
                observation["value"]
            )
            == winning_normalized
        ):
            return observation["value"]

    return None


def fixed_prior_prediction(
    scenario
):
    candidate_scores = (
        defaultdict(float)
    )

    original_values = {}

    seen_agents = set()

    for observation in scenario[
        "observations"
    ]:
        agent_id = observation[
            "agent_id"
        ]

        value_key = normalize(
            observation["value"]
        )

        dedupe_key = (
            agent_id,
            value_key
        )

        if dedupe_key in seen_agents:
            continue

        seen_agents.add(
            dedupe_key
        )

        candidate_scores[
            value_key
        ] += get_expert_prior(
            agent_id,
            scenario["fact_type"]
        )

        original_values[
            value_key
        ] = observation[
            "value"
        ]

    ranked = sorted(
        candidate_scores.items(),
        key=lambda x: (
            -x[1],
            x[0]
        )
    )

    if len(ranked) < 2:
        return original_values[
            ranked[0][0]
        ]

    if abs(
        ranked[0][1]
        - ranked[1][1]
    ) <= 1e-9:
        return None

    return original_values[
        ranked[0][0]
    ]


def update_verified_history(
    provider,
    scenario
):
    truth = scenario[
        "reference_value"
    ]

    if truth is None:
        return

    truth_key = normalize(truth)

    seen = set()

    for observation in scenario[
        "observations"
    ]:
        key = (
            observation["agent_id"],
            observation["fact_type"],
            observation[
                "extraction_type"
            ],
        )

        if key in seen:
            continue

        seen.add(key)

        correct = (
            normalize(
                observation["value"]
            )
            == truth_key
        )

        provider.record_verified_outcome(
            agent_id=observation[
                "agent_id"
            ],
            fact_type=observation[
                "fact_type"
            ],
            extraction_type=
                observation[
                    "extraction_type"
                ],
            correct=correct,
        )


def evaluate_baseline(
    scenarios,
    predictor
):
    correct = 0
    total = 0
    abstained = 0

    for scenario in scenarios:
        if (
            scenario[
                "expected_action"
            ]
            != "resolve"
        ):
            continue

        prediction = predictor(
            scenario
        )

        total += 1

        if prediction is None:
            abstained += 1
            continue

        if (
            normalize(prediction)
            == normalize(
                scenario[
                    "reference_value"
                ]
            )
        ):
            correct += 1

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    coverage = (
        (total - abstained) / total
        if total
        else 0.0
    )

    selective_accuracy = (
        correct
        / (total - abstained)
        if (
            total - abstained
        )
        else 0.0
    )

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "abstained": abstained,
        "coverage": coverage,
        "selective_accuracy":
            selective_accuracy,
    }


def evaluate_adaptive(
    scenarios,
    corroboration_method
):
    provider = (
        InMemoryTrustProvider()
    )

    results = []

    correct = 0
    resolvable_total = 0

    auto_total = 0
    auto_correct = 0

    ambiguous_total = 0
    ambiguous_contested = 0

    category_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
            "contested": 0,
        }
    )

    regime_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    for scenario in scenarios:
        observations = (
            to_observations(
                scenario
            )
        )

        result = resolve(
            observations=
                observations,
            provider=provider,

            # DEV exploratory values only.
            # These are NOT frozen.
            min_winner_score=0.60,
            min_margin=0.10,

            corroboration_method=
                corroboration_method,

            source_weight=0.50,
            corroboration_weight=0.30,
            temporal_weight=0.20,
        )

        expected_action = (
            scenario[
                "expected_action"
            ]
        )

        category = scenario[
            "category"
        ]

        regime = scenario[
            "regime"
        ]

        if expected_action == "contest":
            ambiguous_total += 1

            if (
                result["decision"]
                == "contested"
            ):
                ambiguous_contested += 1

            results.append({
                "scenario_id":
                    scenario[
                        "scenario_id"
                    ],
                "category":
                    category,
                "regime":
                    regime,
                "expected":
                    "contest",
                "decision":
                    result[
                        "decision"
                    ],
            })

            continue

        resolvable_total += 1

        category_stats[
            category
        ]["total"] += 1

        regime_stats[
            regime
        ]["total"] += 1

        winner = result.get(
            "winner"
        )

        winner_value = (
            winner["value"]
            if winner
            else None
        )

        is_correct = (
            normalize(
                winner_value
            )
            == normalize(
                scenario[
                    "reference_value"
                ]
            )
        )

        if is_correct:
            correct += 1

            category_stats[
                category
            ]["correct"] += 1

            regime_stats[
                regime
            ]["correct"] += 1

        if (
            result["decision"]
            == "contested"
        ):
            category_stats[
                category
            ]["contested"] += 1

        if (
            result["decision"]
            == "auto_resolve"
        ):
            auto_total += 1

            if is_correct:
                auto_correct += 1

        results.append({
            "scenario_id":
                scenario[
                    "scenario_id"
                ],
            "category":
                category,
            "regime":
                regime,
            "reference":
                scenario[
                    "reference_value"
                ],
            "winner":
                winner_value,
            "decision":
                result[
                    "decision"
                ],
            "margin":
                result.get(
                    "margin"
                ),
            "correct":
                is_correct,
        })

        # IMPORTANT:
        # prediction happens BEFORE
        # verified outcome is revealed.
        #
        # This is prequential:
        # predict -> verify -> learn.
        update_verified_history(
            provider,
            scenario
        )

    winner_accuracy = (
        correct / resolvable_total
        if resolvable_total
        else 0.0
    )

    auto_accuracy = (
        auto_correct / auto_total
        if auto_total
        else 0.0
    )

    auto_coverage = (
        auto_total
        / resolvable_total
        if resolvable_total
        else 0.0
    )

    ambiguity_accuracy = (
        ambiguous_contested
        / ambiguous_total
        if ambiguous_total
        else 0.0
    )

    return {
        "corroboration_method":
            corroboration_method,

        "winner_accuracy":
            winner_accuracy,

        "resolvable_total":
            resolvable_total,

        "auto_total":
            auto_total,

        "auto_accuracy":
            auto_accuracy,

        "auto_coverage":
            auto_coverage,

        "ambiguous_total":
            ambiguous_total,

        "ambiguous_contested":
            ambiguous_contested,

        "ambiguity_contest_rate":
            ambiguity_accuracy,

        "category_stats":
            dict(category_stats),

        "regime_stats":
            dict(regime_stats),

        "predictions":
            results,
    }


def print_baseline(
    name,
    result
):
    print(
        f"\n{name}"
    )

    print(
        "  Accuracy:",
        f"{result['accuracy']:.3f}"
    )

    print(
        "  Coverage:",
        f"{result['coverage']:.3f}"
    )

    print(
        "  Selective accuracy:",
        f"{result['selective_accuracy']:.3f}"
    )


def print_adaptive(result):
    print(
        "\nAdaptive VeriMem -",
        result[
            "corroboration_method"
        ]
    )

    print(
        "  Winner accuracy:",
        f"{result['winner_accuracy']:.3f}"
    )

    print(
        "  Auto coverage:",
        f"{result['auto_coverage']:.3f}"
    )

    print(
        "  Auto accuracy:",
        f"{result['auto_accuracy']:.3f}"
    )

    print(
        "  Ambiguous contest rate:",
        f"{result['ambiguity_contest_rate']:.3f}"
    )


def main():
    scenarios = load_dev()

    print(
        "Loaded DEV scenarios:",
        len(scenarios)
    )

    print(
        "IMPORTANT: CALIBRATION and TEST "
        "were not loaded."
    )

    lww = evaluate_baseline(
        scenarios,
        lww_prediction
    )

    majority = evaluate_baseline(
        scenarios,
        majority_prediction
    )

    fixed_prior = evaluate_baseline(
        scenarios,
        fixed_prior_prediction
    )

    noisy_or_result = (
        evaluate_adaptive(
            scenarios,
            "noisy_or"
        )
    )

    log_odds_result = (
        evaluate_adaptive(
            scenarios,
            "log_odds"
        )
    )

    print_baseline(
        "Last-write-wins",
        lww
    )

    print_baseline(
        "Majority vote",
        majority
    )

    print_baseline(
        "Fixed expert-prior vote",
        fixed_prior
    )

    print_adaptive(
        noisy_or_result
    )

    print_adaptive(
        log_odds_result
    )

    output = {
        "dataset": "V2 DEV only",
        "scenario_count":
            len(scenarios),

        "baselines": {
            "last_write_wins":
                lww,
            "majority_vote":
                majority,
            "fixed_prior":
                fixed_prior,
        },

        "adaptive": {
            "noisy_or":
                noisy_or_result,
            "log_odds":
                log_odds_result,
        },
    }

    output_path = (
        RESULTS_DIR
        / "stage3_v2_dev_results.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2
        )

    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":
    main()