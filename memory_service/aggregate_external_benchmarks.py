import csv
import json
import statistics
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "benchmark_runs"

OUTPUT_JSON = BASE_DIR / "external_benchmark_summary.json"
OUTPUT_CSV = BASE_DIR / "external_benchmark_summary.csv"


def mean_sd(values):
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
            "min": None,
            "max": None,
        }

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "min": min(values),
        "max": max(values),
    }


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def aggregate_mem0():
    files = sorted(
        RUNS_DIR.glob(
            "mem0_benchmark_run_*.json"
        )
    )

    rows = []

    for path in files:
        data = load_json(path)
        summary = data["summary"]

        rows.append({
            "file": path.name,

            "overall_accuracy":
                summary[
                    "top1_accuracy_on_returned"
                ],

            "effective_accuracy":
                summary[
                    "effective_accuracy"
                ],

            "newer_correct":
                summary[
                    "newer_correct"
                ][
                    "accuracy_on_returned"
                ],

            "newer_wrong":
                summary[
                    "newer_wrong"
                ][
                    "accuracy_on_returned"
                ],

            "api_success":
                summary[
                    "api_success_rate"
                ],

            "retrieval_coverage":
                summary[
                    "retrieval_coverage"
                ],
        })

    return {
        "runs": rows,

        "overall_accuracy":
            mean_sd([
                row["overall_accuracy"]
                for row in rows
            ]),

        "newer_correct":
            mean_sd([
                row["newer_correct"]
                for row in rows
            ]),

        "newer_wrong_safety":
            mean_sd([
                row["newer_wrong"]
                for row in rows
            ]),

        "api_success":
            mean_sd([
                row["api_success"]
                for row in rows
            ]),

        "retrieval_coverage":
            mean_sd([
                row["retrieval_coverage"]
                for row in rows
            ]),
    }


def aggregate_zep():
    files = sorted(
        RUNS_DIR.glob(
            "zep_benchmark_run_*.json"
        )
    )

    rows = []

    for path in files:
        data = load_json(path)
        summary = data["summary"]

        rows.append({
            "file": path.name,

            "raw_accuracy":
                summary[
                    "raw_top1"
                ][
                    "accuracy_on_returned"
                ],

            "current_valid_accuracy":
                summary[
                    "current_valid"
                ][
                    "accuracy_on_returned"
                ],

            "raw_newer_correct":
                summary[
                    "raw_top1_newer_correct"
                ][
                    "accuracy_on_returned"
                ],

            "raw_newer_wrong":
                summary[
                    "raw_top1_newer_wrong"
                ][
                    "accuracy_on_returned"
                ],

            "current_newer_correct":
                summary[
                    "current_valid_newer_correct"
                ][
                    "accuracy_on_returned"
                ],

            "current_newer_wrong":
                summary[
                    "current_valid_newer_wrong"
                ][
                    "accuracy_on_returned"
                ],

            "api_success":
                summary[
                    "api_success_rate"
                ],
        })

    return {
        "runs": rows,

        "raw_top1_accuracy":
            mean_sd([
                row["raw_accuracy"]
                for row in rows
            ]),

        "current_valid_accuracy":
            mean_sd([
                row["current_valid_accuracy"]
                for row in rows
            ]),

        "raw_newer_correct":
            mean_sd([
                row["raw_newer_correct"]
                for row in rows
            ]),

        "raw_newer_wrong_safety":
            mean_sd([
                row["raw_newer_wrong"]
                for row in rows
            ]),

        "current_valid_newer_correct":
            mean_sd([
                row["current_newer_correct"]
                for row in rows
            ]),

        "current_valid_newer_wrong_safety":
            mean_sd([
                row["current_newer_wrong"]
                for row in rows
            ]),

        "api_success":
            mean_sd([
                row["api_success"]
                for row in rows
            ]),
    }


def pct(value):
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def mean_sd_text(metric):
    if metric["mean"] is None:
        return "N/A"

    return (
        f"{metric['mean'] * 100:.2f}% "
        f"± {metric['sd'] * 100:.2f}%"
    )


def main():
    if not RUNS_DIR.exists():
        raise FileNotFoundError(
            f"Benchmark run folder not found: "
            f"{RUNS_DIR}"
        )

    mem0 = aggregate_mem0()
    zep = aggregate_zep()

    output = {
        "mem0": mem0,
        "zep": zep,

        "notes": {
            "repeated_runs":
                (
                    "Repeated cloud-system runs measure "
                    "run-to-run extraction/retrieval variability."
                ),

            "scenario_independence":
                (
                    "Repeated runs are not additional "
                    "independent scenarios."
                ),

            "verimem":
                (
                    "VeriMem uses one deterministic run "
                    "under the frozen V2 configuration."
                )
        }
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2
        )

    csv_rows = [
        {
            "system":
                "Mem0",

            "metric":
                "Overall top-1 accuracy",

            "runs":
                mem0[
                    "overall_accuracy"
                ]["n"],

            "mean":
                mem0[
                    "overall_accuracy"
                ]["mean"],

            "sd":
                mem0[
                    "overall_accuracy"
                ]["sd"],
        },

        {
            "system":
                "Mem0",

            "metric":
                "Newer-correct accuracy",

            "runs":
                mem0[
                    "newer_correct"
                ]["n"],

            "mean":
                mem0[
                    "newer_correct"
                ]["mean"],

            "sd":
                mem0[
                    "newer_correct"
                ]["sd"],
        },

        {
            "system":
                "Mem0",

            "metric":
                "Newer-wrong safety",

            "runs":
                mem0[
                    "newer_wrong_safety"
                ]["n"],

            "mean":
                mem0[
                    "newer_wrong_safety"
                ]["mean"],

            "sd":
                mem0[
                    "newer_wrong_safety"
                ]["sd"],
        },

        {
            "system":
                "Zep raw top-1",

            "metric":
                "Overall accuracy",

            "runs":
                zep[
                    "raw_top1_accuracy"
                ]["n"],

            "mean":
                zep[
                    "raw_top1_accuracy"
                ]["mean"],

            "sd":
                zep[
                    "raw_top1_accuracy"
                ]["sd"],
        },

        {
            "system":
                "Zep raw top-1",

            "metric":
                "Newer-correct accuracy",

            "runs":
                zep[
                    "raw_newer_correct"
                ]["n"],

            "mean":
                zep[
                    "raw_newer_correct"
                ]["mean"],

            "sd":
                zep[
                    "raw_newer_correct"
                ]["sd"],
        },

        {
            "system":
                "Zep raw top-1",

            "metric":
                "Newer-wrong safety",

            "runs":
                zep[
                    "raw_newer_wrong_safety"
                ]["n"],

            "mean":
                zep[
                    "raw_newer_wrong_safety"
                ]["mean"],

            "sd":
                zep[
                    "raw_newer_wrong_safety"
                ]["sd"],
        },

        {
            "system":
                "Zep current-valid",

            "metric":
                "Overall accuracy",

            "runs":
                zep[
                    "current_valid_accuracy"
                ]["n"],

            "mean":
                zep[
                    "current_valid_accuracy"
                ]["mean"],

            "sd":
                zep[
                    "current_valid_accuracy"
                ]["sd"],
        },

        {
            "system":
                "Zep current-valid",

            "metric":
                "Newer-correct accuracy",

            "runs":
                zep[
                    "current_valid_newer_correct"
                ]["n"],

            "mean":
                zep[
                    "current_valid_newer_correct"
                ]["mean"],

            "sd":
                zep[
                    "current_valid_newer_correct"
                ]["sd"],
        },

        {
            "system":
                "Zep current-valid",

            "metric":
                "Newer-wrong safety",

            "runs":
                zep[
                    "current_valid_newer_wrong_safety"
                ]["n"],

            "mean":
                zep[
                    "current_valid_newer_wrong_safety"
                ]["mean"],

            "sd":
                zep[
                    "current_valid_newer_wrong_safety"
                ]["sd"],
        },
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "system",
                "metric",
                "runs",
                "mean",
                "sd"
            ]
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    print("=" * 72)
    print("EXTERNAL BENCHMARK AGGREGATE")
    print("=" * 72)

    print(
        f"\nMem0 runs: "
        f"{len(mem0['runs'])}"
    )

    print(
        "Overall accuracy:          "
        + mean_sd_text(
            mem0[
                "overall_accuracy"
            ]
        )
    )

    print(
        "Newer-correct accuracy:    "
        + mean_sd_text(
            mem0[
                "newer_correct"
            ]
        )
    )

    print(
        "Newer-wrong safety:        "
        + mean_sd_text(
            mem0[
                "newer_wrong_safety"
            ]
        )
    )

    print(
        f"\nZep runs: "
        f"{len(zep['runs'])}"
    )

    print(
        "Raw top-1 accuracy:        "
        + mean_sd_text(
            zep[
                "raw_top1_accuracy"
            ]
        )
    )

    print(
        "Raw newer-correct:         "
        + mean_sd_text(
            zep[
                "raw_newer_correct"
            ]
        )
    )

    print(
        "Raw newer-wrong safety:    "
        + mean_sd_text(
            zep[
                "raw_newer_wrong_safety"
            ]
        )
    )

    print(
        "\nZep current-valid accuracy:"
    )

    print(
        "Overall:                   "
        + mean_sd_text(
            zep[
                "current_valid_accuracy"
            ]
        )
    )

    print(
        "Newer-correct:             "
        + mean_sd_text(
            zep[
                "current_valid_newer_correct"
            ]
        )
    )

    print(
        "Newer-wrong safety:        "
        + mean_sd_text(
            zep[
                "current_valid_newer_wrong_safety"
            ]
        )
    )

    print(
        f"\nSaved JSON: {OUTPUT_JSON}"
    )

    print(
        f"Saved CSV:  {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()