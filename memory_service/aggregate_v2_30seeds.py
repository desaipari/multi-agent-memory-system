import csv
import json
import math
import statistics

from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULT_ROOT = (
    ROOT
    / "memory_service"
    / "results_v2_multiseed"
)

SEEDS = list(
    range(73, 103)
)

REGIMES = [
    "aligned",
    "mixed",
    "shifted",
]

METHODS = [
    "noisy_or",
    "log_odds",
]

PRIOR_STRENGTHS = [
    "5.0",
    "10.0",
    "20.0",
]


def load_seed(seed):
    path = (
        RESULT_ROOT
        / f"seed_{seed:03d}"
        / "stage3_review_aligned_dev_results.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing result for "
            f"seed {seed}: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def summarize(values):
    n = len(values)

    mean = statistics.mean(
        values
    )

    if n > 1:
        sd = statistics.stdev(
            values
        )
    else:
        sd = 0.0

    se = (
        sd / math.sqrt(n)
        if n
        else 0.0
    )

    margin = (
        1.96 * se
    )

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_low":
            mean - margin,
        "ci95_high":
            mean + margin,
    }


def collect_method_metric(
    all_results,
    regime,
    method,
    model,
    metric,
    prior_strength=None,
):
    values = []

    for result in all_results:
        block = (
            result[
                "regimes"
            ][regime][
                "methods"
            ][method]
        )

        if model == "learned":
            value = (
                block[
                    "learned"
                ][
                    prior_strength
                ][metric]
            )

        else:
            value = (
                block[
                    model
                ][metric]
            )

        values.append(
            value
        )

    return values


def collect_baseline_metric(
    all_results,
    regime,
    baseline,
    metric,
):
    return [
        result[
            "regimes"
        ][regime][
            "baselines"
        ][baseline][metric]

        for result
        in all_results
    ]


def collect_stress_metric(
    all_results,
    regime,
    method,
    model,
    category,
    metric,
):
    return [
        result[
            "regimes"
        ][regime][
            "stress_comparison"
        ][method][model][
            category
        ][metric]

        for result
        in all_results
    ]


def paired_difference(
    a,
    b
):
    return [
        x - y
        for x, y
        in zip(a, b)
    ]


def fmt(summary):
    return (
        f"{summary['mean']:.3f} "
        f"± {summary['sd']:.3f} "
        f"[{summary['ci95_low']:.3f}, "
        f"{summary['ci95_high']:.3f}]"
    )


def main():
    all_results = [
        load_seed(seed)
        for seed in SEEDS
    ]

    aggregate = {
        "seeds": SEEDS,
        "n_seeds":
            len(SEEDS),
        "ci_method":
            (
                "Normal approximation: "
                "mean ± 1.96 * SD/sqrt(n)"
            ),
        "regimes": {},
    }

    csv_rows = []

    print(
        "\n"
        + "=" * 88
    )

    print(
        "VERIMEM V2 — "
        "30-SEED AGGREGATE"
    )

    print(
        "Primary model: "
        "noisy_or, learned m=5"
    )

    print(
        "=" * 88
    )

    for regime in REGIMES:
        print(
            f"\nREGIME: "
            f"{regime.upper()}"
        )

        aggregate[
            "regimes"
        ][regime] = {}

        # --------------------------
        # Baselines
        # --------------------------

        lww_conflict = (
            collect_baseline_metric(
                all_results,
                regime,
                "last_write_wins",
                "conflict_accuracy",
            )
        )

        vote_conflict = (
            collect_baseline_metric(
                all_results,
                regime,
                "plain_vote",
                "conflict_accuracy",
            )
        )

        aggregate[
            "regimes"
        ][regime][
            "lww_conflict"
        ] = summarize(
            lww_conflict
        )

        aggregate[
            "regimes"
        ][regime][
            "vote_conflict"
        ] = summarize(
            vote_conflict
        )

        print(
            "LWW conflict:       ",
            fmt(
                summarize(
                    lww_conflict
                )
            )
        )

        print(
            "Vote conflict:      ",
            fmt(
                summarize(
                    vote_conflict
                )
            )
        )

        # --------------------------
        # Primary noisy-OR m=5
        # --------------------------

        fixed_conflict = (
            collect_method_metric(
                all_results,
                regime,
                "noisy_or",
                "fixed",
                "conflict_accuracy",
            )
        )

        learned_conflict = (
            collect_method_metric(
                all_results,
                regime,
                "noisy_or",
                "learned",
                "conflict_accuracy",
                "5.0",
            )
        )

        oracle_conflict = (
            collect_method_metric(
                all_results,
                regime,
                "noisy_or",
                "oracle_source",
                "conflict_accuracy",
            )
        )

        fixed_all = (
            collect_method_metric(
                all_results,
                regime,
                "noisy_or",
                "fixed",
                "accuracy",
            )
        )

        learned_all = (
            collect_method_metric(
                all_results,
                regime,
                "noisy_or",
                "learned",
                "accuracy",
                "5.0",
            )
        )

        oracle_all = (
            collect_method_metric(
                all_results,
                regime,
                "noisy_or",
                "oracle_source",
                "accuracy",
            )
        )

        effect_conflict = (
            paired_difference(
                learned_conflict,
                fixed_conflict,
            )
        )

        effect_all = (
            paired_difference(
                learned_all,
                fixed_all,
            )
        )

        aggregate[
            "regimes"
        ][regime][
            "primary"
        ] = {
            "fixed_conflict":
                summarize(
                    fixed_conflict
                ),

            "learned_conflict":
                summarize(
                    learned_conflict
                ),

            "oracle_source_conflict":
                summarize(
                    oracle_conflict
                ),

            "learned_minus_fixed_conflict":
                summarize(
                    effect_conflict
                ),

            "fixed_all":
                summarize(
                    fixed_all
                ),

            "learned_all":
                summarize(
                    learned_all
                ),

            "oracle_source_all":
                summarize(
                    oracle_all
                ),

            "learned_minus_fixed_all":
                summarize(
                    effect_all
                ),
        }

        print(
            "Fixed conflict:     ",
            fmt(
                summarize(
                    fixed_conflict
                )
            )
        )

        print(
            "Learned conflict:   ",
            fmt(
                summarize(
                    learned_conflict
                )
            )
        )

        print(
            "Oracle-source:      ",
            fmt(
                summarize(
                    oracle_conflict
                )
            )
        )

        print(
            "Learned-Fixed diff: ",
            fmt(
                summarize(
                    effect_conflict
                )
            )
        )

        # --------------------------
        # Stress metrics
        # --------------------------

        fixed_newer_wrong = (
            collect_stress_metric(
                all_results,
                regime,
                "noisy_or",
                "fixed",
                "newer_wrong",
                "winner_accuracy",
            )
        )

        learned_newer_wrong = (
            collect_stress_metric(
                all_results,
                regime,
                "noisy_or",
                "learned_m5",
                "newer_wrong",
                "winner_accuracy",
            )
        )

        learned_newer_correct = (
            collect_stress_metric(
                all_results,
                regime,
                "noisy_or",
                "learned_m5",
                "newer_correct",
                "winner_accuracy",
            )
        )

        ambiguous_gap = (
            collect_stress_metric(
                all_results,
                regime,
                "noisy_or",
                "learned_m5",
                "ambiguous",
                "legacy_gap_contest_rate",
            )
        )

        wrong_effect = (
            paired_difference(
                learned_newer_wrong,
                fixed_newer_wrong,
            )
        )

        aggregate[
            "regimes"
        ][regime][
            "stress"
        ] = {
            "fixed_newer_wrong_prevention":
                summarize(
                    fixed_newer_wrong
                ),

            "learned_newer_wrong_prevention":
                summarize(
                    learned_newer_wrong
                ),

            "learned_minus_fixed_newer_wrong":
                summarize(
                    wrong_effect
                ),

            "learned_newer_correct_acceptance":
                summarize(
                    learned_newer_correct
                ),

            "learned_ambiguous_low_gap":
                summarize(
                    ambiguous_gap
                ),
        }

        print(
            "Newer-correct:      ",
            fmt(
                summarize(
                    learned_newer_correct
                )
            )
        )

        print(
            "Newer-wrong:        ",
            fmt(
                summarize(
                    learned_newer_wrong
                )
            )
        )

        print(
            "Wrong L-F diff:      ",
            fmt(
                summarize(
                    wrong_effect
                )
            )
        )

        print(
            "Ambiguous low-gap:  ",
            fmt(
                summarize(
                    ambiguous_gap
                )
            )
        )

        # --------------------------
        # All ablation cells
        # --------------------------

        aggregate[
            "regimes"
        ][regime][
            "ablations"
        ] = {}

        for method in METHODS:
            aggregate[
                "regimes"
            ][regime][
                "ablations"
            ][method] = {}

            fixed_values = (
                collect_method_metric(
                    all_results,
                    regime,
                    method,
                    "fixed",
                    "conflict_accuracy",
                )
            )

            oracle_values = (
                collect_method_metric(
                    all_results,
                    regime,
                    method,
                    "oracle_source",
                    "conflict_accuracy",
                )
            )

            block = {
                "fixed_conflict":
                    summarize(
                        fixed_values
                    ),

                "oracle_source_conflict":
                    summarize(
                        oracle_values
                    ),

                "learned":
                    {},
            }

            for strength in (
                PRIOR_STRENGTHS
            ):
                learned_values = (
                    collect_method_metric(
                        all_results,
                        regime,
                        method,
                        "learned",
                        "conflict_accuracy",
                        strength,
                    )
                )

                block[
                    "learned"
                ][strength] = (
                    summarize(
                        learned_values
                    )
                )

            aggregate[
                "regimes"
            ][regime][
                "ablations"
            ][method] = block

        # CSV primary table row.
        primary = (
            aggregate[
                "regimes"
            ][regime][
                "primary"
            ]
        )

        stress = (
            aggregate[
                "regimes"
            ][regime][
                "stress"
            ]
        )

        csv_rows.append({
            "regime":
                regime,

            "fixed_conflict_mean":
                primary[
                    "fixed_conflict"
                ]["mean"],

            "fixed_conflict_ci_low":
                primary[
                    "fixed_conflict"
                ]["ci95_low"],

            "fixed_conflict_ci_high":
                primary[
                    "fixed_conflict"
                ]["ci95_high"],

            "learned_conflict_mean":
                primary[
                    "learned_conflict"
                ]["mean"],

            "learned_conflict_ci_low":
                primary[
                    "learned_conflict"
                ]["ci95_low"],

            "learned_conflict_ci_high":
                primary[
                    "learned_conflict"
                ]["ci95_high"],

            "learned_minus_fixed_mean":
                primary[
                    "learned_minus_fixed_conflict"
                ]["mean"],

            "learned_minus_fixed_ci_low":
                primary[
                    "learned_minus_fixed_conflict"
                ]["ci95_low"],

            "learned_minus_fixed_ci_high":
                primary[
                    "learned_minus_fixed_conflict"
                ]["ci95_high"],

            "newer_correct_mean":
                stress[
                    "learned_newer_correct_acceptance"
                ]["mean"],

            "newer_wrong_mean":
                stress[
                    "learned_newer_wrong_prevention"
                ]["mean"],

            "newer_wrong_learned_minus_fixed_mean":
                stress[
                    "learned_minus_fixed_newer_wrong"
                ]["mean"],

            "newer_wrong_learned_minus_fixed_ci_low":
                stress[
                    "learned_minus_fixed_newer_wrong"
                ]["ci95_low"],

            "newer_wrong_learned_minus_fixed_ci_high":
                stress[
                    "learned_minus_fixed_newer_wrong"
                ]["ci95_high"],
        })

    output_json = (
        RESULT_ROOT
        / "aggregate_30seeds.json"
    )

    with output_json.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            aggregate,
            file,
            indent=2
        )

    output_csv = (
        RESULT_ROOT
        / "aggregate_primary_table.csv"
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=
                csv_rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "Saved JSON:",
        output_json
    )

    print(
        "Saved CSV:",
        output_csv
    )


if __name__ == "__main__":
    main()