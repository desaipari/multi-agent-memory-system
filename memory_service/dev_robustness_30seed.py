import json
import math
import random
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = ROOT / "memory_service"
DATASET_DIR = ROOT / "dataset"

sys.path.insert(0, str(MEMORY_DIR))
sys.path.insert(0, str(DATASET_DIR))

import generate_v2_review_benchmark as generator
import stage3_v2_review_aligned as stage3

from verimem_core.providers import FixedTrustProvider


# ------------------------------------------------------------
# FROZEN DEV SETTINGS
# ------------------------------------------------------------

DEV_SEEDS = list(range(73, 103))

REGIMES = [
    "aligned",
    "mixed",
    "shifted",
]

CORROBORATION_METHOD = "noisy_or"

LEARNED_PRIOR_STRENGTH = 5.0

OUTPUT_PATH = (
    MEMORY_DIR
    / "dev_robustness_30seed_summary.json"
)


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

def summarize(values):
    values = [
        float(v)
        for v in values
        if v is not None
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
        "ci95_low": max(0.0, mean - margin),
        "ci95_high": min(1.0, mean + margin),
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("=" * 84)
    print("VERIMEM V2 — 30-SEED DEV ROBUSTNESS SUMMARY")
    print("=" * 84)

    print(
        f"DEV seeds: "
        f"{DEV_SEEDS[0]}-{DEV_SEEDS[-1]} "
        f"({len(DEV_SEEDS)} seeds)"
    )

    print(
        "Metric: top-ranked candidate "
        "conflict accuracy"
    )

    print(
        "Learned trust: "
        "m=5 + noisy-OR"
    )

    print(
        "Fixed trust: "
        "FixedTrustProvider + noisy-OR"
    )

    per_seed = {
        regime: {
            "fixed": [],
            "learned": [],
        }
        for regime in REGIMES
    }

    per_seed_rows = []

    for index, seed in enumerate(
        DEV_SEEDS,
        start=1,
    ):
        print(
            f"\nPreparing DEV seed "
            f"{seed} "
            f"({index}/{len(DEV_SEEDS)})"
        )

        rng = random.Random(seed)

        regimes = generator.build_regimes()

        reliability_data = (
            generator.generate_reliability_benchmark(
                rng,
                regimes,
            )
        )

        for regime in REGIMES:
            regime_data = reliability_data[regime]

            adaptation = regime_data[
                "adaptation"
            ]

            evaluation = regime_data[
                "evaluation"
            ]

            # ------------------------------------------------
            # FIXED EXPERT TRUST
            # Mirrors the fixed-provider Stage-3 evaluation.
            # ------------------------------------------------

            fixed_provider = FixedTrustProvider()

            fixed_result = stage3.evaluate_resolver(
                scenarios=evaluation,
                provider=fixed_provider,
                prior_strength=20.0,
                corroboration_method=(
                    CORROBORATION_METHOD
                ),
            )

            fixed_conflict_accuracy = (
                fixed_result[
                    "conflict_accuracy"
                ]
            )

            # ------------------------------------------------
            # LEARNED CONTEXTUAL TRUST
            # Frozen configuration: m = 5.
            # ------------------------------------------------

            learned_provider = (
                stage3.train_provider(
                    adaptation,
                    count=300,
                )
            )

            learned_result = stage3.evaluate_resolver(
                scenarios=evaluation,
                provider=learned_provider,
                prior_strength=(
                    LEARNED_PRIOR_STRENGTH
                ),
                corroboration_method=(
                    CORROBORATION_METHOD
                ),
            )

            learned_conflict_accuracy = (
                learned_result[
                    "conflict_accuracy"
                ]
            )

            per_seed[regime]["fixed"].append(
                fixed_conflict_accuracy
            )

            per_seed[regime]["learned"].append(
                learned_conflict_accuracy
            )

            per_seed_rows.append({
                "seed": seed,
                "regime": regime,
                "fixed_conflict_accuracy": (
                    fixed_conflict_accuracy
                ),
                "learned_conflict_accuracy": (
                    learned_conflict_accuracy
                ),
            })

            print(
                f"  {regime:<8} "
                f"fixed="
                f"{100 * fixed_conflict_accuracy:.2f}%  "
                f"learned="
                f"{100 * learned_conflict_accuracy:.2f}%"
            )

    # --------------------------------------------------------
    # 30-SEED SUMMARY
    # --------------------------------------------------------

    summary = {}

    for regime in REGIMES:
        summary[regime] = {
            "fixed": summarize(
                per_seed[regime]["fixed"]
            ),
            "learned": summarize(
                per_seed[regime]["learned"]
            ),
        }

    print("\n" + "=" * 84)
    print("30-SEED DEV ROBUSTNESS SUMMARY")
    print("=" * 84)

    for regime in REGIMES:
        print(f"\n{regime.upper()}")

        for method in [
            "fixed",
            "learned",
        ]:
            s = summary[regime][method]

            print(
                f"  {method:<8} "
                f"mean="
                f"{100 * s['mean']:.2f}%  "
                f"sd="
                f"{100 * s['sd']:.2f}%  "
                f"95% CI=["
                f"{100 * s['ci95_low']:.2f}%, "
                f"{100 * s['ci95_high']:.2f}%]"
            )

    output = {
        "experiment": (
            "30-seed DEV robustness"
        ),
        "seeds": DEV_SEEDS,
        "metric": (
            "top-ranked candidate "
            "conflict_accuracy"
        ),
        "corroboration_method": (
            CORROBORATION_METHOD
        ),
        "learned_prior_strength": (
            LEARNED_PRIOR_STRENGTH
        ),
        "per_seed": per_seed_rows,
        "summary": summary,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()