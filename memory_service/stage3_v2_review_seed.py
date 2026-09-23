import argparse
import json

from pathlib import Path

import stage3_v2_review_aligned as base


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(
        args.data_dir
    ).resolve()

    results_dir = Path(
        args.results_dir
    ).resolve()

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    reliability_path = (
        data_dir
        / "dev_reliability.json"
    )

    stress_path = (
        data_dir
        / "dev_stress.json"
    )

    metadata_path = (
        data_dir
        / "metadata.json"
    )

    reliability_data = (
        base.load_json(
            reliability_path
        )
    )

    stress_data = (
        base.load_json(
            stress_path
        )
    )

    metadata = (
        base.load_json(
            metadata_path
        )
    )

    seed = metadata[
        "seed"
    ]

    final_results = {
        "seed":
            seed,

        "dataset":
            (
                "V2 controlled "
                "multi-seed DEV"
            ),

        "confidence_weights": {
            "source": 0.35,
            "corroboration": 0.30,
            "directness": 0.15,
            "time_penalty": -0.20,
        },

        "regimes": {},
    }

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"SEED {seed}"
    )

    print(
        "=" * 72
    )

    for regime in [
        "aligned",
        "mixed",
        "shifted",
    ]:
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
            "baselines": {},
            "methods": {},
            "learning_curves": {},
            "context_diagnostics": {},
            "stress": {},
            "stress_comparison": {},
        }

        # --------------------------------
        # Baselines
        # --------------------------------

        lww = base.evaluate_baseline(
            evaluation,
            base.lww_prediction
        )

        vote = base.evaluate_baseline(
            evaluation,
            base.vote_prediction
        )

        regime_results[
            "baselines"
        ][
            "last_write_wins"
        ] = lww

        regime_results[
            "baselines"
        ][
            "plain_vote"
        ] = vote

        # --------------------------------
        # Resolver methods
        # --------------------------------

        for method in (
            base.CORROBORATION_METHODS
        ):
            fixed_provider = (
                base.FixedTrustProvider()
            )

            fixed = (
                base.evaluate_resolver(
                    scenarios=
                        evaluation,

                    provider=
                        fixed_provider,

                    prior_strength=
                        20.0,

                    corroboration_method=
                        method,
                )
            )

            oracle_provider = (
                base.OracleTrustProvider(
                    true_reliabilities
                )
            )

            oracle_source = (
                base.evaluate_resolver(
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

            for prior_strength in (
                base.PRIOR_STRENGTHS
            ):
                learned_provider = (
                    base.train_provider(
                        adaptation,
                        count=300
                    )
                )

                learned = (
                    base.evaluate_resolver(
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

            regime_results[
                "methods"
            ][method] = (
                method_results
            )

            # --------------------------------
            # Learning curves
            # --------------------------------

            curves = {}

            for prior_strength in (
                base.PRIOR_STRENGTHS
            ):
                points = []

                for checkpoint in (
                    base.LEARNING_CHECKPOINTS
                ):
                    provider = (
                        base.train_provider(
                            adaptation,
                            count=checkpoint
                        )
                    )

                    result = (
                        base.evaluate_resolver(
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

                        "accuracy":
                            result[
                                "accuracy"
                            ],

                        "conflict_accuracy":
                            result[
                                "conflict_accuracy"
                            ],
                    })

                curves[
                    str(
                        prior_strength
                    )
                ] = points

            regime_results[
                "learning_curves"
            ][method] = curves

        # --------------------------------
        # Trust convergence
        # --------------------------------

        learned_300_provider = (
            base.train_provider(
                adaptation,
                count=300
            )
        )

        for prior_strength in (
            base.PRIOR_STRENGTHS
        ):
            diagnostics = (
                base.context_diagnostics(
                    provider=
                        learned_300_provider,

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

        # --------------------------------
        # Stress tests
        # --------------------------------

        regime_stress = (
            stress_data[
                "regimes"
            ][regime]
        )

        for method in (
            base.CORROBORATION_METHODS
        ):
            regime_results[
                "stress"
            ][method] = {}

            for prior_strength in (
                base.PRIOR_STRENGTHS
            ):
                learned_provider = (
                    base.train_provider(
                        adaptation,
                        count=300
                    )
                )

                stress_result = (
                    base.evaluate_stress(
                        scenarios=
                            regime_stress,

                        provider=
                            learned_provider,

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

            # Additional fixed/learned/oracle
            # stress comparison for reviewer.

            fixed_stress = (
                base.evaluate_stress(
                    scenarios=
                        regime_stress,

                    provider=
                        base.FixedTrustProvider(),

                    prior_strength=
                        20.0,

                    corroboration_method=
                        method,
                )
            )

            learned_primary_provider = (
                base.train_provider(
                    adaptation,
                    count=300
                )
            )

            learned_primary_stress = (
                base.evaluate_stress(
                    scenarios=
                        regime_stress,

                    provider=
                        learned_primary_provider,

                    prior_strength=
                        5.0,

                    corroboration_method=
                        method,
                )
            )

            oracle_stress = (
                base.evaluate_stress(
                    scenarios=
                        regime_stress,

                    provider=
                        base.OracleTrustProvider(
                            true_reliabilities
                        ),

                    prior_strength=
                        20.0,

                    corroboration_method=
                        method,
                )
            )

            regime_results[
                "stress_comparison"
            ][method] = {
                "fixed":
                    fixed_stress,

                "learned_m5":
                    learned_primary_stress,

                "oracle_source":
                    oracle_stress,
            }

        final_results[
            "regimes"
        ][regime] = (
            regime_results
        )

        primary = (
            regime_results[
                "methods"
            ][
                "noisy_or"
            ]
        )

        print(
            f"{regime:<10} "
            f"Fixed conflict="
            f"{primary['fixed']['conflict_accuracy']:.3f}  "
            f"Learned m=5 conflict="
            f"{primary['learned']['5.0']['conflict_accuracy']:.3f}  "
            f"Oracle-source conflict="
            f"{primary['oracle_source']['conflict_accuracy']:.3f}"
        )

    output_path = (
        results_dir
        / "stage3_review_aligned_dev_results.json"
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
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()