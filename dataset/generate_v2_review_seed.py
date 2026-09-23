import argparse
import json
import random

from pathlib import Path

import generate_v2_review_benchmark as base


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True
    )

    return parser.parse_args()


def main():
    args = parse_args()

    seed = args.seed

    output_dir = Path(
        args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rng = random.Random(
        seed
    )

    regimes = base.build_regimes()

    reliability_data = (
        base.generate_reliability_benchmark(
            rng,
            regimes
        )
    )

    stress_data = (
        base.generate_stress_benchmark(
            rng,
            regimes
        )
    )

    reliability_path = (
        output_dir
        / "dev_reliability.json"
    )

    stress_path = (
        output_dir
        / "dev_stress.json"
    )

    base.write_json(
        reliability_path,
        {
            "seed": seed,
            "regimes":
                reliability_data,
        }
    )

    base.write_json(
        stress_path,
        {
            "seed": seed,
            "regimes":
                stress_data,
        }
    )

    metadata = {
        "version":
            "verimem_v2_multiseed_dev",

        "seed":
            seed,

        "confidence_formula":
            (
                "0.35 source + "
                "0.30 corroboration + "
                "0.15 directness - "
                "0.20 time penalty"
            ),

        "adaptation_scenarios_per_regime":
            base.ADAPTATION_PER_REGIME,

        "evaluation_scenarios_per_regime":
            base.EVALUATION_PER_REGIME,

        "stress_scenarios_per_category_per_regime":
            (
                base.STRESS_PER_CATEGORY_PER_REGIME
            ),

        "truth_policy":
            (
                "Reference truth is generated "
                "before source reports and "
                "independently of report order."
            ),

        "regime_definition": {
            "aligned":
                (
                    "Source authority matches "
                    "the expert-prior matrix."
                ),

            "mixed":
                (
                    "Source authority is "
                    "reassigned for selected "
                    "fact types."
                ),

            "shifted":
                (
                    "Source authority is "
                    "reassigned for all "
                    "fact types."
                ),
        },

        "hashes": {
            "dev_reliability":
                base.sha256_file(
                    reliability_path
                ),

            "dev_stress":
                base.sha256_file(
                    stress_path
                ),
        },
    }

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    base.write_json(
        metadata_path,
        metadata
    )

    print(
        f"Generated seed {seed}"
    )

    print(
        "Output:",
        output_dir
    )


if __name__ == "__main__":
    main()