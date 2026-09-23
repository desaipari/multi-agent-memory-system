import subprocess
import sys

from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

GENERATOR = (
    ROOT
    / "dataset"
    / "generate_v2_review_seed.py"
)

EVALUATOR = (
    ROOT
    / "memory_service"
    / "stage3_v2_review_seed.py"
)

DATA_ROOT = (
    ROOT
    / "dataset"
    / "v2_review_multiseed"
)

RESULT_ROOT = (
    ROOT
    / "memory_service"
    / "results_v2_multiseed"
)


SEEDS = list(
    range(73, 103)
)


def run_command(command):
    subprocess.run(
        command,
        cwd=ROOT,
        check=True
    )


def main():
    DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Running VeriMem V2 "
        "30-seed DEV replication"
    )

    print(
        "Seeds:",
        SEEDS
    )

    print(
        "CAL and TEST are not used."
    )

    for index, seed in enumerate(
        SEEDS,
        start=1
    ):
        seed_name = (
            f"seed_{seed:03d}"
        )

        data_dir = (
            DATA_ROOT
            / seed_name
        )

        result_dir = (
            RESULT_ROOT
            / seed_name
        )

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"[{index}/"
            f"{len(SEEDS)}] "
            f"Seed {seed}"
        )

        print(
            "=" * 72
        )

        run_command([
            sys.executable,
            str(GENERATOR),
            "--seed",
            str(seed),
            "--output-dir",
            str(data_dir),
        ])

        run_command([
            sys.executable,
            str(EVALUATOR),
            "--data-dir",
            str(data_dir),
            "--results-dir",
            str(result_dir),
        ])

    print(
        "\nAll 30 seeds completed."
    )

    print(
        "Results:",
        RESULT_ROOT
    )


if __name__ == "__main__":
    main()