"""
Stage 1: Generate evaluation scenarios from the ServiceNow event log.

Run from project root:

    python dataset/stage1_generate_dataset.py

Important methodological rules:
- Exactly ONE canonical version of each source is loaded:
    *_clean.csv if present, otherwise original .csv
- Reference value = latest non-null chronologically recorded value
  independently for each (incident_id, fact_type).
- Timestamp ties with different values are excluded as ambiguous.
- No position-based agent assignment.
- Agent/source mapping is based ONLY on source filename.
- Source files are treated as source-attributed benchmark views;
  this script does not claim they are independent real-world systems.
- Ground truth contains only the recorded reference value.
- No VeriMem decision is placed in ground truth.
- All candidates are preserved before any sampling.
- Duplicate candidates are removed.
"""

import os
import json
import re
import hashlib
from collections import defaultdict
from datetime import datetime

import pandas as pd


# ==============================================================
# PATHS
# ==============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "demo_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "generated")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================
# CONFIGURATION
# ==============================================================

COLUMN_TO_FACT_TYPE = {
    "priority": "priority",
    "incident_state": "state",
    "assignment_group": "assignment_group",
    "category": "category",
    "urgency": "urgency",
    "impact": "impact",
}


# This mapping is benchmark metadata, NOT a claim that the
# underlying ServiceNow event log contains explicit agent roles.
#
# If your source construction does not justify one of these
# mappings, remove that source from the source-conditional
# benchmark rather than inventing an agent identity.
SOURCE_TO_AGENT = {
    "ticket_intake": {
        "agent": "intake_agent",
        "extraction_type": "direct",
    },
    "monitoring_logs": {
        "agent": "delivery_agent",
        "extraction_type": "direct",
    },
    "field_reports": {
        "agent": "billing_agent",
        "extraction_type": "inferred",
    },
}


# ==============================================================
# HELPERS
# ==============================================================

exclusion_log = []


def log_exclusion(incident_id, fact_type, reason, details=None):
    exclusion_log.append({
        "incident_id": incident_id,
        "fact_type": fact_type,
        "reason": reason,
        "details": details or {},
    })


def clean_value(value):
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return None

    return text


def normalize_for_compare(value):
    """
    Used ONLY for value equivalence checking.

    Keeps raw values untouched for evaluation output.
    """
    if value is None:
        return ""

    return re.sub(
        r"[\s\-_]+",
        "",
        str(value).lower().strip(),
    )


def parse_timestamp(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def sha256_file(filepath):
    h = hashlib.sha256()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def get_source_key(filepath):
    """
    Convert filename into the benchmark source key.
    """
    stem = os.path.splitext(
        os.path.basename(filepath)
    )[0].lower()

    if stem.endswith("_clean"):
        stem = stem[:-6]

    return stem


# ==============================================================
# CANONICAL SOURCE SELECTION
# ==============================================================

def select_canonical_source_files():
    """
    For each source:
        use *_clean.csv if available
        otherwise use original .csv

    Never load both versions.
    """

    selected = []

    for source_key in SOURCE_TO_AGENT:

        clean_path = os.path.join(
            DATA_DIR,
            f"{source_key}_clean.csv",
        )

        original_path = os.path.join(
            DATA_DIR,
            f"{source_key}.csv",
        )

        if os.path.exists(clean_path):
            selected.append(clean_path)

        elif os.path.exists(original_path):
            selected.append(original_path)

        else:
            print(
                f"WARNING: No source file found for "
                f"{source_key}"
            )

    if not selected:
        raise FileNotFoundError(
            f"No canonical source files found in {DATA_DIR}"
        )

    print("\nCanonical source files:")
    for path in selected:
        print(
            f"  {os.path.basename(path)} "
            f"(source={get_source_key(path)})"
        )

    return selected


# ==============================================================
# SOURCE LOADING
# ==============================================================

def load_source_file(filepath):
    source_key = get_source_key(filepath)

    if source_key not in SOURCE_TO_AGENT:
        print(
            f"UNSUITABLE: {os.path.basename(filepath)} "
            f"has no source-role mapping."
        )
        return None

    metadata = SOURCE_TO_AGENT[source_key]

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        print(
            f"ERROR loading {filepath}: {exc}"
        )
        return None

    print(
        f"  Loaded {os.path.basename(filepath)}: "
        f"{len(df)} rows | "
        f"agent={metadata['agent']} | "
        f"type={metadata['extraction_type']}"
    )

    return {
        "df": df,
        "source_key": source_key,
        "source_file": os.path.basename(filepath),
        "agent": metadata["agent"],
        "extraction_type": metadata["extraction_type"],
    }


# ==============================================================
# TIMELINE
# ==============================================================

def build_incident_timeline(source_files):

    timeline = defaultdict(
        lambda: defaultdict(list)
    )

    source_hashes = {}

    for filepath in source_files:

        source_hashes[
            os.path.basename(filepath)
        ] = sha256_file(filepath)

        loaded = load_source_file(filepath)

        if loaded is None:
            continue

        df = loaded["df"]

        id_candidates = [
            "number",
            "incident_id",
            "id",
            "sys_id",
        ]

        timestamp_candidates = [
            "sys_updated_at",
            "updated_at",
            "sys_created_at",
            "opened_at",
            "timestamp",
            "closed_at",
        ]

        id_col = next(
            (
                c for c in id_candidates
                if c in df.columns
            ),
            None,
        )

        timestamp_col = next(
            (
                c for c in timestamp_candidates
                if c in df.columns
            ),
            None,
        )

        if id_col is None:
            print(
                f"UNSUITABLE: "
                f"{loaded['source_file']} has no "
                f"incident ID column."
            )
            continue

        if timestamp_col is None:
            print(
                f"UNSUITABLE: "
                f"{loaded['source_file']} has no "
                f"timestamp column."
            )
            continue

        for _, row in df.iterrows():

            incident_id = clean_value(
                row.get(id_col)
            )

            if not incident_id:
                continue

            timestamp_raw = clean_value(
                row.get(timestamp_col)
            )

            timestamp = parse_timestamp(
                timestamp_raw
            )

            if timestamp is None:
                continue

            for column, fact_type in COLUMN_TO_FACT_TYPE.items():

                if column not in df.columns:
                    continue

                raw_value = clean_value(
                    row.get(column)
                )

                if raw_value is None:
                    continue

                timeline[
                    incident_id
                ][fact_type].append({
                    "raw_value": raw_value,
                    "normalized_value":
                        normalize_for_compare(raw_value),
                    "timestamp": timestamp,
                    "timestamp_str": timestamp_raw,
                    "agent":
                        loaded["agent"],
                    "extraction_type":
                        loaded["extraction_type"],
                    "source_key":
                        loaded["source_key"],
                    "source_file":
                        loaded["source_file"],
                })

    return timeline, source_hashes


# ==============================================================
# REFERENCE VALUE
# ==============================================================

def get_reference_value(
    incident_id,
    fact_type,
    observations,
):
    """
    Latest chronologically recorded value.

    Important:
    If multiple different values have the exact same timestamp,
    temporal ordering is ambiguous. Do not arbitrarily choose one.
    """

    if not observations:
        return None

    timestamp_groups = defaultdict(list)

    for obs in observations:
        timestamp_groups[
            obs["timestamp"]
        ].append(obs)

    for timestamp, same_time_obs in timestamp_groups.items():

        normalized_values = {
            obs["normalized_value"]
            for obs in same_time_obs
        }

        if len(normalized_values) > 1:

            log_exclusion(
                incident_id,
                fact_type,
                "ambiguous_timestamp_tie",
                {
                    "timestamp":
                        timestamp.isoformat(),
                    "values": [
                        obs["raw_value"]
                        for obs in same_time_obs
                    ],
                },
            )

            return None

    latest_timestamp = max(
        timestamp_groups.keys()
    )

    latest_observations = timestamp_groups[
        latest_timestamp
    ]

    reference = latest_observations[0]

    return {
        "reference_value":
            reference["raw_value"],
        "reference_normalized":
            reference["normalized_value"],
        "reference_timestamp":
            reference["timestamp"].isoformat(),
        "reference_source":
            reference["source_key"],
        "reference_agent":
            reference["agent"],
        "reference_method":
            "latest_timestamped_record",
    }


# ==============================================================
# CANDIDATE CREATION
# ==============================================================

def build_candidates(timeline):

    contradictions = []
    corroborations = []

    seen_contradiction = set()
    seen_corroboration = set()

    for incident_id, fact_types in timeline.items():

        for fact_type, observations in fact_types.items():

            agents = {
                obs["agent"]
                for obs in observations
            }

            if len(agents) < 2:
                continue

            normalized_values = {
                obs["normalized_value"]
                for obs in observations
            }

            # --------------------------------------------------
            # CORROBORATION
            # --------------------------------------------------

            if len(normalized_values) == 1:

                signature = (
                    incident_id,
                    fact_type,
                    frozenset(normalized_values),
                )

                if signature in seen_corroboration:
                    continue

                seen_corroboration.add(signature)

                corroborations.append({
                    "incident_id": incident_id,
                    "fact_type": fact_type,
                    "agents": sorted(agents),
                    "turns": [
                        {
                            "turn": i + 1,
                            "entity": incident_id,
                            "fact_type": fact_type,
                            "value": obs["raw_value"],
                            "raw_value": obs["raw_value"],
                            "normalized_value":
                                obs["normalized_value"],
                            "agent": obs["agent"],
                            "extraction_type":
                                obs["extraction_type"],
                            "source_name":
                                obs["source_key"],
                            "source_file":
                                obs["source_file"],
                            "timestamp":
                                obs["timestamp"].isoformat(),
                            "timestamp_str":
                                obs["timestamp_str"],
                        }
                        for i, obs
                        in enumerate(
                            sorted(
                                observations,
                                key=lambda x:
                                x["timestamp"],
                            )
                        )
                    ],
                })

                continue

            # --------------------------------------------------
            # CONTRADICTION
            # --------------------------------------------------

            reference = get_reference_value(
                incident_id,
                fact_type,
                observations,
            )

            if reference is None:
                continue

            observed_normalized = {
                obs["normalized_value"]
                for obs in observations
            }

            if (
                reference["reference_normalized"]
                not in observed_normalized
            ):
                log_exclusion(
                    incident_id,
                    fact_type,
                    "reference_not_observed",
                    {
                        "reference":
                            reference["reference_value"],
                        "observed": [
                            obs["raw_value"]
                            for obs in observations
                        ],
                    },
                )
                continue

            signature = (
                incident_id,
                fact_type,
                frozenset(observed_normalized),
            )

            if signature in seen_contradiction:
                log_exclusion(
                    incident_id,
                    fact_type,
                    "duplicate_scenario",
                )
                continue

            seen_contradiction.add(signature)

            sorted_observations = sorted(
                observations,
                key=lambda x: x["timestamp"],
            )

            turns = []

            for index, obs in enumerate(
                sorted_observations
            ):

                turns.append({
                    "turn": index + 1,
                    "entity": incident_id,
                    "fact_type": fact_type,
                    "value": obs["raw_value"],
                    "raw_value": obs["raw_value"],
                    "normalized_value":
                        obs["normalized_value"],
                    "agent": obs["agent"],
                    "extraction_type":
                        obs["extraction_type"],
                    "source_name":
                        obs["source_key"],
                    "source_file":
                        obs["source_file"],
                    "timestamp":
                        obs["timestamp"].isoformat(),
                    "timestamp_str":
                        obs["timestamp_str"],
                })

            contradictions.append({
                "incident_id": incident_id,
                "fact_type": fact_type,
                "agents": sorted(agents),
                "turns": turns,
                "reference": reference,
                "all_observed_values": [
                    {
                        "value": obs["raw_value"],
                        "normalized":
                            obs["normalized_value"],
                        "agent": obs["agent"],
                        "source":
                            obs["source_key"],
                        "timestamp":
                            obs["timestamp"].isoformat(),
                    }
                    for obs in sorted_observations
                ],
            })

    return contradictions, corroborations


# ==============================================================
# SCENARIO FORMATTING
# ==============================================================

def format_contradiction(
    candidate,
    scenario_id,
):

    reference = candidate["reference"]

    return {
        "scenario_id": scenario_id,
        "incident_id":
            candidate["incident_id"],
        "entity":
            candidate["incident_id"],
        "fact_type":
            candidate["fact_type"],
        "agents":
            candidate["agents"],
        "turns":
            candidate["turns"],

        "ground_truth": {
            "contradiction_expected": True,

            "recorded_reference_value":
                reference["reference_value"],

            "reference_value":
                reference["reference_value"],

            "reference_normalized":
                reference["reference_normalized"],

            "reference_timestamp":
                reference["reference_timestamp"],

            "reference_source":
                reference["reference_source"],

            "reference_agent":
                reference["reference_agent"],

            "reference_method":
                reference["reference_method"],
        },

        "dataset_metadata": {
            "source":
                "ServiceNow UCI ML Repository",
            "doi":
                "10.24432/C5JW3R",

            "reference_definition":
                "Latest non-null chronologically "
                "recorded value for the same "
                "(incident_id, fact_type).",

            "reference_is_objective_truth":
                False,

            "source_independence_claim":
                False,

            "note":
                "Source files are source-attributed "
                "benchmark views. They are not "
                "treated as independent real-world "
                "systems unless provenance separately "
                "establishes independence.",
        },
    }


def format_corroboration(
    candidate,
    scenario_id,
):

    agreed_value = candidate["turns"][0]["raw_value"]

    return {
        "scenario_id": scenario_id,
        "incident_id":
            candidate["incident_id"],
        "entity":
            candidate["incident_id"],
        "fact_type":
            candidate["fact_type"],
        "agents":
            candidate["agents"],
        "turns":
            candidate["turns"],

        "ground_truth": {
            "contradiction_expected": False,
            "agreed_value": agreed_value,
        },

        "dataset_metadata": {
            "source":
                "ServiceNow UCI ML Repository",
            "source_independence_claim":
                False,
        },
    }


# ==============================================================
# VALIDATION
# ==============================================================

def validate_scenarios(
    contradictions,
    corroborations,
):

    print("\nRunning Stage 1 validation...")

    failures = []

    all_ids = set()

    for scenario in (
        contradictions + corroborations
    ):

        scenario_id = scenario["scenario_id"]

        if scenario_id in all_ids:
            failures.append(
                f"Duplicate scenario ID: {scenario_id}"
            )

        all_ids.add(scenario_id)

    # Contradictions
    for scenario in contradictions:

        gt = scenario["ground_truth"]

        if not gt.get(
            "contradiction_expected"
        ):
            failures.append(
                f"{scenario['scenario_id']}: "
                f"not marked contradiction"
            )

        forbidden = [
            "auto_resolve",
            "winning_agent",
            "correct_resolution",
            "resolution_type",
        ]

        for field in forbidden:
            if field in gt:
                failures.append(
                    f"{scenario['scenario_id']}: "
                    f"{field} appears in ground truth"
                )

        reference = gt.get(
            "reference_normalized"
        )

        turn_values = {
            normalize_for_compare(
                turn["value"]
            )
            for turn in scenario["turns"]
        }

        if reference not in turn_values:
            failures.append(
                f"{scenario['scenario_id']}: "
                f"reference not present in observations"
            )

        if len(turn_values) < 2:
            failures.append(
                f"{scenario['scenario_id']}: "
                f"not an actual value conflict"
            )

    # Corroborations
    for scenario in corroborations:

        gt = scenario["ground_truth"]

        if gt.get("contradiction_expected"):
            failures.append(
                f"{scenario['scenario_id']}: "
                f"corroboration marked as contradiction"
            )

        values = {
            normalize_for_compare(
                turn["value"]
            )
            for turn in scenario["turns"]
        }

        if len(values) != 1:
            failures.append(
                f"{scenario['scenario_id']}: "
                f"corroboration has multiple values"
            )

    if failures:

        print(
            f"\nVALIDATION FAILED: "
            f"{len(failures)} issues"
        )

        for failure in failures[:50]:
            print("  FAIL:", failure)

        raise AssertionError(
            "Stage 1 validation failed."
        )

    print("  All Stage 1 assertions passed.")


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print("STAGE 1 — DATASET GENERATION")
    print("=" * 70)

    source_files = (
        select_canonical_source_files()
    )

    timeline, source_hashes = (
        build_incident_timeline(
            source_files
        )
    )

    print(
        f"\nTimeline contains "
        f"{len(timeline)} incidents."
    )

    contradictions, corroborations = (
        build_candidates(timeline)
    )

    print(
        f"\nCandidate statistics:"
    )

    print(
        f"  Contradictions: "
        f"{len(contradictions)}"
    )

    print(
        f"  Corroborations: "
        f"{len(corroborations)}"
    )

    print(
        f"  Exclusions: "
        f"{len(exclusion_log)}"
    )

    contradiction_scenarios = [
        format_contradiction(
            candidate,
            f"GEN_C_{i + 1:05d}",
        )
        for i, candidate
        in enumerate(contradictions)
    ]

    corroboration_scenarios = [
        format_corroboration(
            candidate,
            f"GEN_N_{i + 1:05d}",
        )
        for i, candidate
        in enumerate(corroborations)
    ]

    validate_scenarios(
        contradiction_scenarios,
        corroboration_scenarios,
    )

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    fact_counts = defaultdict(int)

    for scenario in contradiction_scenarios:
        fact_counts[
            scenario["fact_type"]
        ] += 1

    print("\nContradictions by fact type:")

    for fact_type, count in sorted(
        fact_counts.items(),
        key=lambda x: -x[1],
    ):
        print(
            f"  {fact_type:<25} {count:>5}"
        )

    agent_pairs = defaultdict(int)

    for scenario in contradiction_scenarios:

        pair = tuple(
            sorted(
                set(
                    turn["agent"]
                    for turn in scenario["turns"]
                )
            )
        )

        agent_pairs[pair] += 1

    print("\nContradictions by source pair:")

    for pair, count in sorted(
        agent_pairs.items(),
        key=lambda x: -x[1],
    ):
        print(
            f"  {' vs '.join(pair):<40} "
            f"{count:>5}"
        )

    # ----------------------------------------------------------
    # Save all candidates
    # ----------------------------------------------------------

    output = {
        "metadata": {
            "generation_script":
                "dataset/stage1_generate_dataset.py",

            "n_contradiction_candidates":
                len(contradiction_scenarios),

            "n_corroboration_candidates":
                len(corroboration_scenarios),

            "n_excluded":
                len(exclusion_log),

            "source_files":
                [
                    os.path.basename(path)
                    for path in source_files
                ],

            "source_file_sha256":
                source_hashes,

            "ground_truth_method":
                "Latest non-null chronologically "
                "recorded value per "
                "(incident_id, fact_type).",

            "ground_truth_is_objective":
                False,

            "source_independence_claim":
                False,

            "sampling":
                "None. All eligible candidates "
                "are preserved.",
        },

        "contradiction_scenarios":
            contradiction_scenarios,

        "corroboration_scenarios":
            corroboration_scenarios,

        "exclusion_log":
            exclusion_log,
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        "all_candidates.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    exclusion_path = os.path.join(
        OUTPUT_DIR,
        "exclusion_log.json",
    )

    with open(
        exclusion_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            exclusion_log,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETE")
    print("=" * 70)

    print(
        f"Contradictions: "
        f"{len(contradiction_scenarios)}"
    )

    print(
        f"Corroborations: "
        f"{len(corroboration_scenarios)}"
    )

    print(
        f"Excluded: "
        f"{len(exclusion_log)}"
    )

    print(
        f"\nSaved:\n"
        f"  {output_path}\n"
        f"  {exclusion_path}"
    )


if __name__ == "__main__":
    main()