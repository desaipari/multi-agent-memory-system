"""
Stage 2: Create incident-level development/test split.

Run:

    python dataset/stage2_split_dataset.py

Rules:
- Split by incident_id.
- No incident appears in both dev and test.
- Global scenario IDs must be unique.
- Fixed seed.
- Test scenarios are written separately.
"""

import os
import json
import random
import hashlib
from collections import defaultdict


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
)

GENERATED_DIR = os.path.join(
    DATASET_DIR,
    "generated",
)

SPLIT_DIR = os.path.join(
    DATASET_DIR,
    "splits",
)

SCENARIO_DIR = os.path.join(
    DATASET_DIR,
    "scenarios",
)

os.makedirs(
    SPLIT_DIR,
    exist_ok=True,
)


RANDOM_SEED = 42
TEST_FRACTION = 0.30


# ==============================================================
# LOADERS
# ==============================================================

def load_generated_candidates():

    path = os.path.join(
        GENERATED_DIR,
        "all_candidates.json",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. "
            f"Run Stage 1 first."
        )

    with open(
        path,
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return (
        data.get(
            "contradiction_scenarios",
            [],
        ),
        data.get(
            "corroboration_scenarios",
            [],
        ),
    )


def load_manual_scenarios():

    scenarios = []

    if not os.path.exists(
        SCENARIO_DIR
    ):
        return scenarios

    for filename in sorted(
        os.listdir(SCENARIO_DIR)
    ):

        if not filename.endswith(".json"):
            continue

        # Generated L/M/N scenarios are NOT loaded
        # here because Stage 1 is the canonical
        # generated dataset.
        if filename.startswith(
            (
                "category_L_",
                "category_M_",
                "category_N_",
            )
        ):
            continue

        filepath = os.path.join(
            SCENARIO_DIR,
            filename,
        )

        with open(
            filepath,
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        for scenario in data.get(
            "scenarios",
            [],
        ):

            scenario = dict(scenario)

            if "incident_id" not in scenario:

                scenario["incident_id"] = (
                    scenario.get(
                        "entity",
                        "UNKNOWN",
                    )
                )

            scenario["_source_file"] = (
                filename
            )

            scenario["_is_manual"] = True

            scenarios.append(
                scenario
            )

    print(
        f"Loaded {len(scenarios)} "
        f"manual scenarios."
    )

    return scenarios


# ==============================================================
# VALIDATION
# ==============================================================

def get_incident_id(scenario):

    return (
        scenario.get("incident_id")
        or scenario.get("entity")
        or "UNKNOWN"
    )


def assert_global_scenario_ids(
    scenarios
):

    seen = set()
    duplicates = []

    for scenario in scenarios:

        sid = scenario.get(
            "scenario_id"
        )

        if not sid:
            raise AssertionError(
                "Scenario without scenario_id."
            )

        if sid in seen:
            duplicates.append(sid)

        seen.add(sid)

    if duplicates:

        raise AssertionError(
            "Duplicate scenario IDs found "
            f"globally: {duplicates[:20]}"
        )

    print(
        f"  PASS: {len(seen)} globally "
        f"unique scenario IDs."
    )


def assert_no_duplicate_incident_fact(
    scenarios
):

    seen = set()
    duplicates = []

    for scenario in scenarios:

        key = (
            get_incident_id(scenario),
            scenario.get(
                "fact_type",
                "",
            ),
        )

        # Manual controlled scenarios may intentionally
        # reuse an incident/fact combination. We only
        # reject exact duplicate scenario records.
        fingerprint = (
            key,
            json.dumps(
                scenario.get(
                    "turns",
                    [],
                ),
                sort_keys=True,
            ),
        )

        if fingerprint in seen:
            duplicates.append(
                scenario.get(
                    "scenario_id"
                )
            )

        seen.add(fingerprint)

    if duplicates:

        raise AssertionError(
            "Exact duplicate incident/fact "
            f"scenarios found: "
            f"{duplicates[:20]}"
        )

    print(
        "  PASS: No exact duplicate "
        "incident/fact scenario records."
    )


def assert_no_overlap(
    dev,
    test,
):

    dev_incidents = {
        get_incident_id(s)
        for s in dev
    }

    test_incidents = {
        get_incident_id(s)
        for s in test
    }

    overlap = (
        dev_incidents &
        test_incidents
    )

    if overlap:

        raise AssertionError(
            f"Incident leakage detected: "
            f"{len(overlap)} incidents "
            f"appear in both splits.\n"
            f"{sorted(overlap)[:20]}"
        )

    print(
        "  PASS: Zero incident overlap."
    )


def assert_split_ids_unique(
    scenarios,
    name,
):

    ids = [
        s["scenario_id"]
        for s in scenarios
    ]

    if len(ids) != len(set(ids)):

        raise AssertionError(
            f"Duplicate IDs in {name}."
        )

    print(
        f"  PASS: {name} scenario IDs unique."
    )


# ==============================================================
# SPLITTING
# ==============================================================

def split_by_incident(
    scenarios,
    test_fraction,
    seed,
):

    rng = random.Random(seed)

    by_incident = defaultdict(list)

    for scenario in scenarios:

        incident_id = get_incident_id(
            scenario
        )

        by_incident[
            incident_id
        ].append(scenario)

    incident_ids = sorted(
        by_incident.keys()
    )

    if len(incident_ids) < 2:

        raise ValueError(
            "Need at least two incidents "
            "for a dev/test split."
        )

    rng.shuffle(incident_ids)

    n_test = max(
        1,
        int(
            len(incident_ids)
            * test_fraction
        ),
    )

    test_incidents = set(
        incident_ids[:n_test]
    )

    dev_incidents = set(
        incident_ids[n_test:]
    )

    dev = []
    test = []

    for incident_id, items in (
        by_incident.items()
    ):

        if incident_id in test_incidents:
            test.extend(items)
        else:
            dev.extend(items)

    return (
        dev,
        test,
        dev_incidents,
        test_incidents,
    )


# ==============================================================
# HASH
# ==============================================================

def dataset_hash(scenarios):

    canonical = []

    for scenario in scenarios:

        canonical.append({
            "scenario_id":
                scenario.get(
                    "scenario_id"
                ),
            "incident_id":
                get_incident_id(
                    scenario
                ),
            "fact_type":
                scenario.get(
                    "fact_type"
                ),
        })

    canonical.sort(
        key=lambda x:
        x["scenario_id"]
    )

    content = json.dumps(
        canonical,
        sort_keys=True,
    ).encode()

    return hashlib.sha256(
        content
    ).hexdigest()


# ==============================================================
# STATISTICS
# ==============================================================

def print_statistics(
    scenarios,
    name,
):

    contradictions = [
        s for s in scenarios
        if s.get(
            "ground_truth",
            {}
        ).get(
            "contradiction_expected"
        )
    ]

    corroborations = [
        s for s in scenarios
        if not s.get(
            "ground_truth",
            {}
        ).get(
            "contradiction_expected"
        )
    ]

    incidents = {
        get_incident_id(s)
        for s in scenarios
    }

    print(f"\n{name}")
    print(
        f"  Scenarios:      {len(scenarios)}"
    )
    print(
        f"  Contradictions: {len(contradictions)}"
    )
    print(
        f"  Corroborations: {len(corroborations)}"
    )
    print(
        f"  Incidents:      {len(incidents)}"
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print("STAGE 2 — INCIDENT-LEVEL DEV/TEST SPLIT")
    print("=" * 70)

    generated_contradictions, \
        generated_corroborations = (
            load_generated_candidates()
        )

    manual = load_manual_scenarios()

    generated = (
        generated_contradictions
        + generated_corroborations
    )

    all_scenarios = (
        generated + manual
    )

    if not all_scenarios:

        raise ValueError(
            "No scenarios found."
        )

    print(
        f"\nGenerated scenarios: "
        f"{len(generated)}"
    )

    print(
        f"Manual scenarios:    "
        f"{len(manual)}"
    )

    print(
        f"Combined scenarios:   "
        f"{len(all_scenarios)}"
    )

    # ----------------------------------------------------------
    # Global validation BEFORE split
    # ----------------------------------------------------------

    print("\nGlobal validation:")

    assert_global_scenario_ids(
        all_scenarios
    )

    assert_no_duplicate_incident_fact(
        all_scenarios
    )

    # ----------------------------------------------------------
    # Split
    # ----------------------------------------------------------

    dev, test, dev_incidents, \
        test_incidents = split_by_incident(
            all_scenarios,
            TEST_FRACTION,
            RANDOM_SEED,
        )

    print("\nSplit validation:")

    assert_no_overlap(
        dev,
        test,
    )

    assert_split_ids_unique(
        dev,
        "development",
    )

    assert_split_ids_unique(
        test,
        "test",
    )

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    print_statistics(
        dev,
        "DEVELOPMENT SET",
    )

    print_statistics(
        test,
        "TEST SET",
    )

    # ----------------------------------------------------------
    # Dataset hash
    # ----------------------------------------------------------

    data_hash = dataset_hash(
        all_scenarios
    )

    print(
        f"\nDataset hash: "
        f"{data_hash}"
    )

    # ----------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------

    manifest = {

        "random_seed":
            RANDOM_SEED,

        "test_fraction":
            TEST_FRACTION,

        "dataset_hash":
            data_hash,

        "total_scenarios":
            len(all_scenarios),

        "dev_scenario_count":
            len(dev),

        "test_scenario_count":
            len(test),

        "dev_incident_count":
            len(dev_incidents),

        "test_incident_count":
            len(test_incidents),

        "dev_scenario_ids":
            sorted(
                s["scenario_id"]
                for s in dev
            ),

        "test_scenario_ids":
            sorted(
                s["scenario_id"]
                for s in test
            ),

        "dev_incident_ids":
            sorted(
                dev_incidents
            ),

        "test_incident_ids":
            sorted(
                test_incidents
            ),

        "split_method":
            "Incident-level random split.",

        "selection_rule":
            "All parameter selection must use "
            "development data only.",

        "test_rule":
            "Test set must remain untouched "
            "until final evaluation.",
    }

    manifest_path = os.path.join(
        SPLIT_DIR,
        "split_manifest.json",
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    dev_path = os.path.join(
        SPLIT_DIR,
        "dev_scenarios.json",
    )

    with open(
        dev_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "scenarios": dev,
                "split": "development",
            },
            f,
            indent=2,
        )

    test_path = os.path.join(
        SPLIT_DIR,
        "test_scenarios_LOCKED.json",
    )

    with open(
        test_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "scenarios": test,
                "split": "test",
                "warning":
                    "LOCKED. Do not use "
                    "for parameter selection.",
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETE")
    print("=" * 70)

    print(
        f"Development: "
        f"{len(dev)} scenarios / "
        f"{len(dev_incidents)} incidents"
    )

    print(
        f"Test:        "
        f"{len(test)} scenarios / "
        f"{len(test_incidents)} incidents"
    )

    print(
        "\nIncident overlap: ZERO"
    )

    print(
        f"\nSaved:\n"
        f"  {manifest_path}\n"
        f"  {dev_path}\n"
        f"  {test_path}"
    )


if __name__ == "__main__":
    main()