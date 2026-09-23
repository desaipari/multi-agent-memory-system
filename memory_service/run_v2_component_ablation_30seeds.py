import json
import math
import statistics

from contextlib import contextmanager
from pathlib import Path

import stage3_v2_review_aligned as stage3
import verimem_core.resolver as resolver_module


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT
    / "dataset"
    / "v2_review_multiseed"
)

OUTPUT_DIR = (
    ROOT
    / "memory_service"
    / "results_v2_component_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEEDS = list(range(73, 103))

REGIMES = [
    "aligned",
    "mixed",
    "shifted",
]


# ---------------------------------------------------------
# PREDECLARED WEIGHT CONFIGURATIONS
# ---------------------------------------------------------

WEIGHT_CONFIGS = {
    # Current frozen design
    "full": {
        "source": 0.35,
        "corroboration": 0.30,
        "directness": 0.15,
        "time_penalty": 0.20,
    },

    # Leave-one-component-out ablations
    "no_source": {
        "source": 0.00,
        "corroboration": 0.30,
        "directness": 0.15,
        "time_penalty": 0.20,
    },

    "no_corroboration": {
        "source": 0.35,
        "corroboration": 0.00,
        "directness": 0.15,
        "time_penalty": 0.20,
    },

    "no_directness": {
        "source": 0.35,
        "corroboration": 0.30,
        "directness": 0.00,
        "time_penalty": 0.20,
    },

    "no_time": {
        "source": 0.35,
        "corroboration": 0.30,
        "directness": 0.15,
        "time_penalty": 0.00,
    },

    # Limited sensitivity configurations.
    # These are NOT a hyperparameter search.
    "balanced": {
        "source": 0.30,
        "corroboration": 0.30,
        "directness": 0.20,
        "time_penalty": 0.20,
    },

    "source_heavy": {
        "source": 0.45,
        "corroboration": 0.25,
        "directness": 0.15,
        "time_penalty": 0.15,
    },

    "corroboration_heavy": {
        "source": 0.30,
        "corroboration": 0.40,
        "directness": 0.15,
        "time_penalty": 0.15,
    },

    "time_light": {
        "source": 0.35,
        "corroboration": 0.30,
        "directness": 0.20,
        "time_penalty": 0.15,
    },
}


@contextmanager
def temporary_weights(weights):
    original = dict(
        resolver_module.CONFIDENCE_WEIGHTS
    )

    resolver_module.CONFIDENCE_WEIGHTS.clear()

    resolver_module.CONFIDENCE_WEIGHTS.update(
        weights
    )

    try:
        yield

    finally:
        resolver_module.CONFIDENCE_WEIGHTS.clear()

        resolver_module.CONFIDENCE_WEIGHTS.update(
            original
        )


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def summarize(values):
    n = len(values)

    mean = statistics.mean(values)

    sd = (
        statistics.stdev(values)
        if n > 1
        else 0.0
    )

    se = (
        sd / math.sqrt(n)
        if n
        else 0.0
    )

    margin = 1.96 * se

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def format_summary(summary):
    return (
        f"{summary['mean']:.3f} "
        f"± {summary['sd']:.3f} "
        f"[{summary['ci95_low']:.3f}, "
        f"{summary['ci95_high']:.3f}]"
    )


def paired_difference(
    full_values,
    variant_values
):
    return [
        full - variant
        for full, variant
        in zip(
            full_values,
            variant_values
        )
    ]


def evaluate_seed(
    seed,
    weights
):
    seed_dir = (
        DATA_ROOT
        / f"seed_{seed:03d}"
    )

    reliability = load_json(
        seed_dir
        / "dev_reliability.json"
    )

    stress = load_json(
        seed_dir
        / "dev_stress.json"
    )

    results = {}

    with temporary_weights(weights):

        for regime in REGIMES:

            regime_data = (
                reliability[
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

            provider = (
                stage3.train_provider(
                    adaptation,
                    count=300
                )
            )

            resolver_result = (
                stage3.evaluate_resolver(
                    scenarios=evaluation,
                    provider=provider,
                    prior_strength=5.0,
                    corroboration_method="noisy_or",
                )
            )

            stress_result = (
                stage3.evaluate_stress(
                    scenarios=(
                        stress[
                            "regimes"
                        ][regime]
                    ),
                    provider=provider,
                    prior_strength=5.0,
                    corroboration_method="noisy_or",
                )
            )

            results[regime] = {
                "accuracy":
                    resolver_result[
                        "accuracy"
                    ],

                "conflict_accuracy":
                    resolver_result[
                        "conflict_accuracy"
                    ],

                "newer_correct":
                    stress_result[
                        "newer_correct"
                    ][
                        "winner_accuracy"
                    ],

                "newer_wrong":
                    stress_result[
                        "newer_wrong"
                    ][
                        "winner_accuracy"
                    ],

                "ambiguous_low_gap":
                    stress_result[
                        "ambiguous"
                    ][
                        "legacy_gap_contest_rate"
                    ],
            }

    return results


def main():
    raw_results = {
        name: {
            regime: []
            for regime in REGIMES
        }
        for name in WEIGHT_CONFIGS
    }

    print(
        "VeriMem V2 component ablation"
    )

    print(
        "30 seeds: 73-102"
    )

    print(
        "Primary resolver: "
        "Learned m=5 + noisy-OR"
    )

    print(
        "CAL and TEST are not accessed."
    )

    for seed_index, seed in enumerate(
        SEEDS,
        start=1
    ):
        print(
            f"Seed {seed} "
            f"({seed_index}/30)"
        )

        for (
            config_name,
            weights
        ) in WEIGHT_CONFIGS.items():

            seed_result = (
                evaluate_seed(
                    seed,
                    weights
                )
            )

            for regime in REGIMES:
                raw_results[
                    config_name
                ][regime].append(
                    seed_result[
                        regime
                    ]
                )

    aggregate = {
        "seeds": SEEDS,
        "primary_model":
            "learned_m5_noisy_or",
        "weight_configs":
            WEIGHT_CONFIGS,
        "results": {},
    }

    print(
        "\n"
        + "=" * 95
    )

    print(
        "LEAVE-ONE-COMPONENT-OUT"
    )

    print(
        "Positive Full-Ablation means "
        "removing that component hurts."
    )

    print(
        "=" * 95
    )

    for regime in REGIMES:
        print(
            f"\nREGIME: "
            f"{regime.upper()}"
        )

        aggregate[
            "results"
        ][regime] = {}

        full_conflict = [
            row[
                "conflict_accuracy"
            ]
            for row
            in raw_results[
                "full"
            ][regime]
        ]

        full_wrong = [
            row[
                "newer_wrong"
            ]
            for row
            in raw_results[
                "full"
            ][regime]
        ]

        print(
            "Full conflict:            ",
            format_summary(
                summarize(
                    full_conflict
                )
            )
        )

        for config_name in [
            "no_source",
            "no_corroboration",
            "no_directness",
            "no_time",
        ]:
            variant_conflict = [
                row[
                    "conflict_accuracy"
                ]
                for row
                in raw_results[
                    config_name
                ][regime]
            ]

            difference = (
                paired_difference(
                    full_conflict,
                    variant_conflict
                )
            )

            variant_wrong = [
                row[
                    "newer_wrong"
                ]
                for row
                in raw_results[
                    config_name
                ][regime]
            ]

            wrong_difference = (
                paired_difference(
                    full_wrong,
                    variant_wrong
                )
            )

            aggregate[
                "results"
            ][regime][
                config_name
            ] = {
                "conflict_accuracy":
                    summarize(
                        variant_conflict
                    ),

                "full_minus_ablation":
                    summarize(
                        difference
                    ),

                "newer_wrong_prevention":
                    summarize(
                        variant_wrong
                    ),

                "full_minus_ablation_newer_wrong":
                    summarize(
                        wrong_difference
                    ),
            }

            print(
                f"{config_name:<24}",
                format_summary(
                    summarize(
                        variant_conflict
                    )
                )
            )

            print(
                f"  Full - {config_name:<17}",
                format_summary(
                    summarize(
                        difference
                    )
                )
            )

        aggregate[
            "results"
        ][regime][
            "full"
        ] = {
            "conflict_accuracy":
                summarize(
                    full_conflict
                ),

            "newer_wrong_prevention":
                summarize(
                    full_wrong
                ),
        }

    print(
        "\n"
        + "=" * 95
    )

    print(
        "LIMITED WEIGHT SENSITIVITY"
    )

    print(
        "This is robustness analysis, "
        "not an optimization search."
    )

    print(
        "=" * 95
    )

    for regime in REGIMES:

        print(
            f"\nREGIME: "
            f"{regime.upper()}"
        )

        for config_name in [
            "full",
            "balanced",
            "source_heavy",
            "corroboration_heavy",
            "time_light",
        ]:
            values = [
                row[
                    "conflict_accuracy"
                ]
                for row
                in raw_results[
                    config_name
                ][regime]
            ]

            aggregate[
                "results"
            ][regime].setdefault(
                "sensitivity",
                {}
            )

            aggregate[
                "results"
            ][regime][
                "sensitivity"
            ][config_name] = (
                summarize(values)
            )

            print(
                f"{config_name:<24}",
                format_summary(
                    summarize(values)
                )
            )

    output_path = (
        OUTPUT_DIR
        / "component_ablation_30seeds.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            aggregate,
            file,
            indent=2
        )

    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":
    main()