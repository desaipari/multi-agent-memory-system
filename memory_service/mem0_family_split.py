import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# FIND THE 5 EXISTING MEM0 RUN FILES
# ------------------------------------------------------------

run_files = sorted(
    ROOT.rglob("mem0_benchmark_run_*.json")
)

if len(run_files) != 5:
    raise RuntimeError(
        f"Expected exactly 5 Mem0 run files, "
        f"found {len(run_files)}"
    )


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def get_family(scenario_id):
    """
    Supports IDs such as:
      A_001, A001
      F_002, F002
      G_006, G006
      H_004, H004
    """

    if scenario_id is None:
        return None

    sid = str(scenario_id).strip().upper()

    match = re.match(
        r"^([AFGH])_?\d+",
        sid
    )

    if not match:
        return None

    prefix = match.group(1)

    if prefix in {"A", "F"}:
        return "A/F"

    if prefix in {"G", "H"}:
        return "G/H"

    return None


def get_results(data):
    """
    Final Mem0 benchmark files should contain
    per-scenario records under 'results'.
    """

    for key in [
        "results",
        "scenario_results",
        "cases",
        "scored_results",
    ]:
        rows = data.get(key)

        if isinstance(rows, list):
            return rows

    raise KeyError(
        "Could not find per-scenario result list. "
        f"Top-level keys: {list(data.keys())}"
    )


def get_correct(row):
    """
    Supports the correctness field names used
    across our benchmark versions.
    """

    for key in [
        "is_correct",
        "top1_correct",
        "correct",
    ]:
        if key in row:
            return bool(row[key])

    raise KeyError(
        "No correctness field found for "
        f"{row.get('scenario_id')}. "
        f"Available keys: {list(row.keys())}"
    )


def accuracy(values):
    if not values:
        return None

    return sum(values) / len(values)


def mean_sd(values):
    mean = statistics.mean(values)

    sd = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    return mean, sd


# ------------------------------------------------------------
# ANALYSE EACH RUN
# ------------------------------------------------------------

af_run_accuracies = []
gh_run_accuracies = []

per_run_output = []


print("=" * 78)
print("MEM0 A/F vs G/H FAMILY ANALYSIS")
print("=" * 78)

print(f"\nFound {len(run_files)} Mem0 runs:\n")

for path in run_files:
    print(path)


for run_index, path in enumerate(
    run_files,
    start=1,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    rows = get_results(data)

    af_results = []
    gh_results = []

    af_ids = []
    gh_ids = []

    for row in rows:
        scenario_id = row.get(
            "scenario_id"
        )

        family = get_family(
            scenario_id
        )

        if family is None:
            continue

        is_correct = get_correct(
            row
        )

        if family == "A/F":
            af_results.append(
                is_correct
            )
            af_ids.append(
                scenario_id
            )

        elif family == "G/H":
            gh_results.append(
                is_correct
            )
            gh_ids.append(
                scenario_id
            )

    # Expected scored benchmark:
    # A/F = 6 cases
    # G/H = 12 cases
    if len(af_results) != 6:
        raise RuntimeError(
            f"{path.name}: expected 6 A/F cases, "
            f"found {len(af_results)}. "
            f"IDs={af_ids}"
        )

    if len(gh_results) != 12:
        raise RuntimeError(
            f"{path.name}: expected 12 G/H cases, "
            f"found {len(gh_results)}. "
            f"IDs={gh_ids}"
        )

    af_accuracy = accuracy(
        af_results
    )

    gh_accuracy = accuracy(
        gh_results
    )

    af_run_accuracies.append(
        af_accuracy
    )

    gh_run_accuracies.append(
        gh_accuracy
    )

    run_summary = {
        "run": run_index,
        "file": str(path),
        "af_correct": sum(
            af_results
        ),
        "af_total": len(
            af_results
        ),
        "af_accuracy": af_accuracy,
        "gh_correct": sum(
            gh_results
        ),
        "gh_total": len(
            gh_results
        ),
        "gh_accuracy": gh_accuracy,
    }

    per_run_output.append(
        run_summary
    )

    print(
        f"\nRUN {run_index}"
    )

    print(
        f"  A/F: "
        f"{sum(af_results)}/"
        f"{len(af_results)} = "
        f"{100 * af_accuracy:.2f}%"
    )

    print(
        f"  G/H: "
        f"{sum(gh_results)}/"
        f"{len(gh_results)} = "
        f"{100 * gh_accuracy:.2f}%"
    )


# ------------------------------------------------------------
# FIVE-RUN SUMMARY
# ------------------------------------------------------------

af_mean, af_sd = mean_sd(
    af_run_accuracies
)

gh_mean, gh_sd = mean_sd(
    gh_run_accuracies
)


print("\n" + "=" * 78)
print("5-RUN MEM0 FAMILY SUMMARY")
print("=" * 78)

print(
    f"A/F mean ± SD: "
    f"{100 * af_mean:.2f}% ± "
    f"{100 * af_sd:.2f}%"
)

print(
    f"G/H mean ± SD: "
    f"{100 * gh_mean:.2f}% ± "
    f"{100 * gh_sd:.2f}%"
)


# ------------------------------------------------------------
# SAVE RESULT
# ------------------------------------------------------------

output = {
    "system": "Mem0",
    "n_runs": len(run_files),

    "family_definition": {
        "A/F": [
            "A",
            "F",
        ],
        "G/H": [
            "G",
            "H",
        ],
    },

    "per_run": per_run_output,

    "summary": {
        "af_mean": af_mean,
        "af_sd": af_sd,
        "gh_mean": gh_mean,
        "gh_sd": gh_sd,
    },
}


output_path = (
    ROOT
    / "memory_service"
    / "mem0_family_split_summary.json"
)

with output_path.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        output,
        f,
        indent=2,
    )


print(
    f"\nSaved: {output_path}"
)