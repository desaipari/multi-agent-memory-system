"""
VeriMem V2 threshold calibration.

This file intentionally reuses the existing weight_calibration.py filename so
we do not add another experiment script to the repository.

CALIBRATION PROTOCOL (fixed before looking at CAL results)
----------------------------------------------------------
Fresh CAL seeds: 103-132 (30 seeds)
Frozen resolver: learned R(agent, fact), prior strength m=5, noisy-OR
Frozen confidence formula: 0.35S + 0.30Cr + 0.15D - 0.20T

Threshold grid:
  minimum winner score = 0.20 ... 0.50
  minimum margin       = 0.05 ... 0.30

Selection rule:
  1. auto-resolution accuracy >= 90% on natural conflict CAL cases
  2. ambiguous stress contest rate >= 95%
  3. among qualifying pairs, choose highest auto-resolution coverage
  4. exact metric ties use the more conservative (higher) thresholds

TEST data is never accessed here.
"""

import json
import math
import random
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = ROOT / "memory_service"
DATASET_DIR = ROOT / "dataset"

# Allow this existing script to import both sibling experiment modules.
sys.path.insert(0, str(MEMORY_DIR))
sys.path.insert(0, str(DATASET_DIR))

import generate_v2_review_benchmark as generator
import stage3_v2_review_aligned as stage3
from verimem_core.resolver import resolve


# -------------------------------------------------------------------
# FROZEN V2 SETTINGS
# -------------------------------------------------------------------

CAL_SEEDS = list(range(103, 133))
REGIMES = ["aligned", "mixed", "shifted"]

PRIOR_STRENGTH = 5.0
CORROBORATION_METHOD = "noisy_or"

WINNER_SCORE_THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

MARGIN_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

MIN_AUTO_RESOLUTION_ACCURACY = 0.90
MIN_AMBIGUOUS_CONTEST_RATE = 0.95

OUTPUT_PATH = MEMORY_DIR / "threshold_sensitivity.json"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return None, None

    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total

    centre = (
        p
        + z2 / (2.0 * total)
    ) / denominator

    adjustment = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )

    return (
        centre - adjustment,
        centre + adjustment,
    )


def mean_ci(values):
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


def threshold_auto_resolves(record, score_threshold, margin_threshold):
    # The resolver always contests exact score ties, regardless of thresholds.
    if record["tie"]:
        return False

    if record["winner_score"] is None:
        return False

    if record["margin"] is None:
        return False

    return (
        record["winner_score"] >= score_threshold
        and record["margin"] >= margin_threshold
    )


def create_record(seed, regime, kind, scenario, provider):
    """
    Run the frozen V2 resolver once with thresholds disabled.

    Thresholds only change auto_resolve vs contested; they do not change the
    candidate scores or ranking. Therefore we calculate scores once and apply
    all 42 threshold pairs to the stored winner score and margin. This makes
    the calibration much faster without changing the decision logic.
    """

    result = resolve(
        observations=stage3.to_observations(scenario),
        provider=provider,
        min_winner_score=-999.0,
        min_margin=-999.0,
        corroboration_method=CORROBORATION_METHOD,
        prior_strength=PRIOR_STRENGTH,
    )

    winner = result.get("winner")

    prediction = (
        winner["value"]
        if winner
        else None
    )

    reference = scenario.get("reference_value")

    winner_correct = (
        None
        if reference is None
        else stage3.is_correct(
            prediction,
            reference,
        )
    )

    return {
        "seed": seed,
        "regime": regime,
        "kind": kind,
        "winner_score": (
            winner["score"]
            if winner
            else None
        ),
        "margin": result.get("margin"),
        "tie": result.get("reason") == "tie",
        "winner_correct": winner_correct,
    }


# -------------------------------------------------------------------
# BUILD FRESH CAL RECORDS
# -------------------------------------------------------------------

def build_calibration_records():
    records = []

    total_natural = 0
    total_stress = 0

    for index, seed in enumerate(CAL_SEEDS, start=1):
        print(
            f"Preparing CAL seed {seed} "
            f"({index}/{len(CAL_SEEDS)})"
        )

        # Same deterministic benchmark generator as Stage 3, but with a
        # completely fresh seed range that was not used for DEV ablation.
        rng = random.Random(seed)
        regimes = generator.build_regimes()

        reliability_data = (
            generator.generate_reliability_benchmark(
                rng,
                regimes,
            )
        )

        stress_data = (
            generator.generate_stress_benchmark(
                rng,
                regimes,
            )
        )

        for regime in REGIMES:
            regime_data = reliability_data[regime]

            adaptation = regime_data["adaptation"]
            evaluation = regime_data["evaluation"]

            # Frozen Stage-3 adaptive trust configuration.
            provider = stage3.train_provider(
                adaptation,
                count=300,
            )

            total_natural += len(evaluation)

            # Auto-resolution thresholds only apply when there are multiple
            # competing values, so non-conflict natural cases are counted for
            # dataset accounting but excluded from threshold selection.
            for scenario in evaluation:
                if not scenario["is_conflict"]:
                    continue

                records.append(
                    create_record(
                        seed=seed,
                        regime=regime,
                        kind="natural_conflict",
                        scenario=scenario,
                        provider=provider,
                    )
                )

            regime_stress = stress_data[regime]
            total_stress += len(regime_stress)

            for scenario in regime_stress:
                records.append(
                    create_record(
                        seed=seed,
                        regime=regime,
                        kind=scenario["category"],
                        scenario=scenario,
                        provider=provider,
                    )
                )

    return records, total_natural, total_stress


# -------------------------------------------------------------------
# THRESHOLD METRICS
# -------------------------------------------------------------------

def evaluate_threshold(records, score_threshold, margin_threshold):
    natural = [
        row
        for row in records
        if row["kind"] == "natural_conflict"
    ]

    auto_natural = [
        row
        for row in natural
        if threshold_auto_resolves(
            row,
            score_threshold,
            margin_threshold,
        )
    ]

    correct_auto = sum(
        1
        for row in auto_natural
        if row["winner_correct"]
    )

    wrong_auto = (
        len(auto_natural)
        - correct_auto
    )

    auto_accuracy = (
        correct_auto / len(auto_natural)
        if auto_natural
        else None
    )

    coverage = (
        len(auto_natural) / len(natural)
        if natural
        else 0.0
    )

    contested_rate = 1.0 - coverage

    # Ambiguous cases have no correct winner by construction. The desired
    # behavior is abstention/contest rather than forced auto-resolution.
    ambiguous = [
        row
        for row in records
        if row["kind"] == "ambiguous"
    ]

    ambiguous_contested = sum(
        1
        for row in ambiguous
        if not threshold_auto_resolves(
            row,
            score_threshold,
            margin_threshold,
        )
    )

    ambiguous_contest_rate = (
        ambiguous_contested / len(ambiguous)
        if ambiguous
        else 0.0
    )

    # Targeted governance diagnostics. These are reported, but are not used
    # to change the predeclared selection rule after seeing CAL results.
    newer_wrong = [
        row
        for row in records
        if row["kind"] == "newer_wrong"
    ]

    newer_wrong_wrong_auto = sum(
        1
        for row in newer_wrong
        if (
            threshold_auto_resolves(
                row,
                score_threshold,
                margin_threshold,
            )
            and not row["winner_correct"]
        )
    )

    newer_wrong_safe_rate = (
        1.0
        - newer_wrong_wrong_auto / len(newer_wrong)
        if newer_wrong
        else 0.0
    )

    newer_correct = [
        row
        for row in records
        if row["kind"] == "newer_correct"
    ]

    newer_correct_correct_auto = sum(
        1
        for row in newer_correct
        if (
            threshold_auto_resolves(
                row,
                score_threshold,
                margin_threshold,
            )
            and row["winner_correct"]
        )
    )

    newer_correct_correct_auto_rate = (
        newer_correct_correct_auto / len(newer_correct)
        if newer_correct
        else 0.0
    )

    ci_low, ci_high = wilson_interval(
        correct_auto,
        len(auto_natural),
    )

    return {
        "winner_score_threshold": score_threshold,
        "margin_threshold": margin_threshold,

        "natural_conflicts": len(natural),
        "auto_count": len(auto_natural),
        "contested_count": (
            len(natural) - len(auto_natural)
        ),
        "correct_auto": correct_auto,
        "wrong_auto": wrong_auto,

        "coverage": coverage,
        "contested_rate": contested_rate,
        "auto_resolution_accuracy": auto_accuracy,
        "selective_error": (
            wrong_auto / len(auto_natural)
            if auto_natural
            else None
        ),
        "auto_accuracy_wilson_95_low": ci_low,
        "auto_accuracy_wilson_95_high": ci_high,

        "ambiguous_total": len(ambiguous),
        "ambiguous_contested": ambiguous_contested,
        "ambiguous_contest_rate": ambiguous_contest_rate,

        "newer_wrong_total": len(newer_wrong),
        "newer_wrong_wrong_auto": newer_wrong_wrong_auto,
        "newer_wrong_wrong_auto_rate": (
            newer_wrong_wrong_auto / len(newer_wrong)
            if newer_wrong
            else 0.0
        ),
        "newer_wrong_safe_rate": newer_wrong_safe_rate,

        "newer_correct_total": len(newer_correct),
        "newer_correct_correct_auto": newer_correct_correct_auto,
        "newer_correct_correct_auto_rate": (
            newer_correct_correct_auto_rate
        ),
    }


def qualifies(row):
    accuracy = row["auto_resolution_accuracy"]

    return (
        accuracy is not None
        and accuracy >= MIN_AUTO_RESOLUTION_ACCURACY
        and row["ambiguous_contest_rate"]
        >= MIN_AMBIGUOUS_CONTEST_RATE
    )


def select_threshold(rows):
    qualifying = [
        row
        for row in rows
        if qualifies(row)
    ]

    if not qualifying:
        return None, []

    # Primary objective: maximum coverage subject to the two safety floors.
    # If all measured metrics tie, prefer the stricter thresholds.
    selected = max(
        qualifying,
        key=lambda row: (
            row["coverage"],
            row["auto_resolution_accuracy"],
            row["ambiguous_contest_rate"],
            row["winner_score_threshold"],
            row["margin_threshold"],
        ),
    )

    return selected, qualifying


# -------------------------------------------------------------------
# SELECTED-THRESHOLD BREAKDOWNS
# -------------------------------------------------------------------

def filter_records(records, seed=None, regime=None):
    output = records

    if seed is not None:
        output = [
            row
            for row in output
            if row["seed"] == seed
        ]

    if regime is not None:
        output = [
            row
            for row in output
            if row["regime"] == regime
        ]

    return output


def selected_breakdowns(records, selected):
    score_threshold = selected[
        "winner_score_threshold"
    ]

    margin_threshold = selected[
        "margin_threshold"
    ]

    per_regime = {}

    for regime in REGIMES:
        per_regime[regime] = evaluate_threshold(
            filter_records(
                records,
                regime=regime,
            ),
            score_threshold,
            margin_threshold,
        )

    seed_metrics = []

    for seed in CAL_SEEDS:
        metric = evaluate_threshold(
            filter_records(
                records,
                seed=seed,
            ),
            score_threshold,
            margin_threshold,
        )

        metric["seed"] = seed
        seed_metrics.append(metric)

    seed_summary = {
        "coverage": mean_ci([
            row["coverage"]
            for row in seed_metrics
        ]),
        "auto_resolution_accuracy": mean_ci([
            row["auto_resolution_accuracy"]
            for row in seed_metrics
        ]),
        "ambiguous_contest_rate": mean_ci([
            row["ambiguous_contest_rate"]
            for row in seed_metrics
        ]),
        "newer_wrong_safe_rate": mean_ci([
            row["newer_wrong_safe_rate"]
            for row in seed_metrics
        ]),
        "newer_correct_correct_auto_rate": mean_ci([
            row["newer_correct_correct_auto_rate"]
            for row in seed_metrics
        ]),
    }

    return (
        per_regime,
        seed_metrics,
        seed_summary,
    )


def format_percent(value):
    if value is None:
        return "N/A"

    return f"{100.0 * value:.1f}%"


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    print("=" * 92)
    print("VERIMEM V2 THRESHOLD CALIBRATION")
    print("=" * 92)

    print("Fresh CAL seeds: 103-132 (30 seeds)")
    print("DEV seeds 73-102 are not reused.")
    print("TEST is not accessed.")
    print("Frozen resolver: learned m=5 + noisy-OR")
    print(
        "Frozen confidence formula: "
        "0.35S + 0.30Cr + 0.15D - 0.20T"
    )

    print("\nPredeclared selection rule:")
    print("  1. Auto-resolution accuracy >= 90%")
    print("  2. Ambiguous contest rate >= 95%")
    print("  3. Choose highest coverage among qualifying pairs")
    print("  4. Exact metric ties -> stricter thresholds")

    records, total_natural, total_stress = (
        build_calibration_records()
    )

    natural_conflict_count = sum(
        1
        for row in records
        if row["kind"] == "natural_conflict"
    )

    print("\n" + "=" * 92)
    print("CAL DATASET")
    print("=" * 92)
    print(f"Natural CAL scenarios generated: {total_natural}")
    print(f"Natural conflict cases used:     {natural_conflict_count}")
    print(f"Stress CAL scenarios:            {total_stress}")
    print(f"Total fresh CAL scenarios:       {total_natural + total_stress}")

    rows = []

    for score_threshold in WINNER_SCORE_THRESHOLDS:
        for margin_threshold in MARGIN_THRESHOLDS:
            rows.append(
                evaluate_threshold(
                    records,
                    score_threshold,
                    margin_threshold,
                )
            )

    print("\n" + "=" * 110)
    print("THRESHOLD GRID")
    print("=" * 110)
    print(
        f"{'Score':>6} "
        f"{'Margin':>7} "
        f"{'AutoN':>8} "
        f"{'Coverage':>10} "
        f"{'AutoAcc':>10} "
        f"{'SelErr':>9} "
        f"{'AmbContest':>11} "
        f"{'WrongNewSafe':>12} "
        f"{'NewCorrectAuto':>14}"
    )
    print("-" * 110)

    for row in rows:
        print(
            f"{row['winner_score_threshold']:>6.2f} "
            f"{row['margin_threshold']:>7.2f} "
            f"{row['auto_count']:>8} "
            f"{format_percent(row['coverage']):>10} "
            f"{format_percent(row['auto_resolution_accuracy']):>10} "
            f"{format_percent(row['selective_error']):>9} "
            f"{format_percent(row['ambiguous_contest_rate']):>11} "
            f"{format_percent(row['newer_wrong_safe_rate']):>12} "
            f"{format_percent(row['newer_correct_correct_auto_rate']):>14}"
        )

    selected, qualifying = select_threshold(rows)

    print("\n" + "=" * 92)
    print("SELECTION")
    print("=" * 92)
    print(f"Qualifying threshold pairs: {len(qualifying)} / {len(rows)}")

    if selected is None:
        print("NO THRESHOLD PAIR MET THE PREDECLARED SAFETY RULE.")
        print(
            "Do not invent a fallback threshold. "
            "Review the CAL results first."
        )

        output = {
            "stage": "v2_calibration",
            "cal_seeds": CAL_SEEDS,
            "test_used": False,
            "selected_threshold": None,
            "selection_rule": {
                "minimum_auto_resolution_accuracy": (
                    MIN_AUTO_RESOLUTION_ACCURACY
                ),
                "minimum_ambiguous_contest_rate": (
                    MIN_AMBIGUOUS_CONTEST_RATE
                ),
                "objective": (
                    "maximize natural-conflict auto-resolution coverage"
                ),
            },
            "dataset_counts": {
                "natural_generated": total_natural,
                "natural_conflicts": natural_conflict_count,
                "stress": total_stress,
                "total_generated": total_natural + total_stress,
            },
            "threshold_grid": rows,
        }

    else:
        print(
            "Selected minimum winner score:",
            f"{selected['winner_score_threshold']:.2f}",
        )
        print(
            "Selected minimum margin:      ",
            f"{selected['margin_threshold']:.2f}",
        )
        print(
            "Natural conflict coverage:    ",
            format_percent(selected["coverage"]),
        )
        print(
            "Auto-resolution accuracy:     ",
            format_percent(
                selected["auto_resolution_accuracy"]
            ),
        )
        print(
            "Auto-accuracy Wilson 95% CI:  ",
            f"[{format_percent(selected['auto_accuracy_wilson_95_low'])}, "
            f"{format_percent(selected['auto_accuracy_wilson_95_high'])}]",
        )
        print(
            "Selective error:              ",
            format_percent(selected["selective_error"]),
        )
        print(
            "Ambiguous contest rate:       ",
            format_percent(
                selected["ambiguous_contest_rate"]
            ),
        )
        print(
            "Newer-wrong safe rate:        ",
            format_percent(
                selected["newer_wrong_safe_rate"]
            ),
        )
        print(
            "Newer-correct correct-auto:   ",
            format_percent(
                selected[
                    "newer_correct_correct_auto_rate"
                ]
            ),
        )

        (
            per_regime,
            seed_metrics,
            seed_summary,
        ) = selected_breakdowns(
            records,
            selected,
        )

        print("\nSelected-threshold regime breakdown:")

        for regime in REGIMES:
            row = per_regime[regime]
            print(
                f"  {regime:<8} "
                f"coverage={format_percent(row['coverage'])}  "
                f"auto_acc={format_percent(row['auto_resolution_accuracy'])}  "
                f"amb_contest={format_percent(row['ambiguous_contest_rate'])}  "
                f"wrong_new_safe={format_percent(row['newer_wrong_safe_rate'])}"
            )

        print("\n30-seed mean ± 95% CI:")

        for name, summary in seed_summary.items():
            print(
                f"  {name:<34} "
                f"{format_percent(summary['mean'])} "
                f"[{format_percent(summary['ci95_low'])}, "
                f"{format_percent(summary['ci95_high'])}]"
            )

        print("\nValues to place in main.py AFTER reviewing this output:")
        print(
            "V2_MIN_WINNER_SCORE =",
            f"{selected['winner_score_threshold']:.2f}"
        )
        print(
            "V2_MIN_MARGIN =",
            f"{selected['margin_threshold']:.2f}"
        )

        output = {
            "stage": "v2_calibration",
            "cal_seeds": CAL_SEEDS,
            "dev_seeds_reused": False,
            "test_used": False,
            "frozen_model": {
                "prior_strength": PRIOR_STRENGTH,
                "corroboration_method": CORROBORATION_METHOD,
                "confidence_formula": (
                    "0.35S + 0.30Cr + 0.15D - 0.20T"
                ),
            },
            "selection_rule": {
                "minimum_auto_resolution_accuracy": (
                    MIN_AUTO_RESOLUTION_ACCURACY
                ),
                "minimum_ambiguous_contest_rate": (
                    MIN_AMBIGUOUS_CONTEST_RATE
                ),
                "objective": (
                    "maximize natural-conflict auto-resolution coverage"
                ),
                "tie_break": (
                    "prefer stricter thresholds if measured metrics tie"
                ),
            },
            "dataset_counts": {
                "natural_generated": total_natural,
                "natural_conflicts": natural_conflict_count,
                "stress": total_stress,
                "total_generated": total_natural + total_stress,
            },
            "selected_threshold": selected,
            "selected_per_regime": per_regime,
            "selected_seed_metrics": seed_metrics,
            "selected_seed_summary": seed_summary,
            "qualifying_pair_count": len(qualifying),
            "threshold_grid": rows,
        }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
