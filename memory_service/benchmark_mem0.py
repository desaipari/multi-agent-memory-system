"""
Mem0 benchmark for VeriMem IT-incident contradiction scenarios.

Smoke test:
    $env:MEM0_API_KEY="your_key"
    $env:BENCHMARK_LIMIT="5"
    python memory_service/benchmark_mem0.py

Run all eligible labelled conflicts:
    $env:BENCHMARK_LIMIT="0"
    python memory_service/benchmark_mem0.py
"""

import json
import os
import re
import time
import uuid
import statistics
from pathlib import Path

import requests

try:
    from mem0 import MemoryClient
except ImportError:
    raise SystemExit(
        "Install Mem0 first: pip install -U mem0ai"
    )


ROOT = Path(__file__).resolve().parents[1]

SCENARIOS_DIR = (
    ROOT / "dataset" / "scenarios"
)

RESULTS_DIR = Path(__file__).with_name("benchmark_runs")
RESULTS_DIR.mkdir(exist_ok=True)


def get_next_output_path():
    existing = list(
        RESULTS_DIR.glob("mem0_benchmark_run_*.json")
    )

    run_numbers = []

    for path in existing:
        match = re.search(
            r"mem0_benchmark_run_(\d+)\.json$",
            path.name
        )
        if match:
            run_numbers.append(int(match.group(1)))

    next_run = max(run_numbers, default=0) + 1

    return RESULTS_DIR / f"mem0_benchmark_run_{next_run:02d}.json"


MEM0_API_KEY = os.getenv(
    "MEM0_API_KEY",
    ""
).strip()

MEM0_API_BASE = os.getenv(
    "MEM0_API_BASE",
    "https://api.mem0.ai"
).rstrip("/")


# 5 = smoke test.
# Set BENCHMARK_LIMIT=0 to test all eligible cases.
BENCHMARK_LIMIT = int(
    os.getenv(
        "BENCHMARK_LIMIT",
        "5"
    )
)

POLL_INTERVAL = float(
    os.getenv(
        "MEM0_POLL_INTERVAL",
        "2"
    )
)

EVENT_TIMEOUT = int(
    os.getenv(
        "MEM0_EVENT_TIMEOUT",
        "90"
    )
)

CLEANUP_AFTER_SCENARIO = (
    os.getenv(
        "BENCHMARK_CLEANUP",
        "1"
    ) != "0"
)


if not MEM0_API_KEY:
    raise SystemExit(
        'Set MEM0_API_KEY first:\n'
        '$env:MEM0_API_KEY="your_key"'
    )


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize(value):
    value = str(
        value or ""
    ).lower().strip()

    return re.sub(
        r"[\s\-_:/.,()]+",
        "",
        value
    )


def value_variants(value):
    """
    Allows simple equivalent forms.

    Example:
        2 - High
        becomes:
        2high
        high
    """

    raw = str(
        value or ""
    ).strip()

    stripped = re.sub(
        r"^\s*\d+\s*[-_:./ ]*\s*",
        "",
        raw
    )

    variants = {
        normalize(raw),
        normalize(stripped)
    }

    return {
        v for v in variants
        if v
    }


def get_candidate_values(turns):
    values = []
    seen = set()

    for turn in turns:
        value = turn.get("value")

        if value in (None, ""):
            continue

        key = normalize(value)

        if key not in seen:
            seen.add(key)
            values.append(
                str(value)
            )

    return values


def match_candidate(
    retrieved_text,
    candidate_values
):
    """
    Map Mem0's free-text result back to one
    of the known candidate values.
    """

    text = normalize(
        retrieved_text
    )

    matches = []

    for value in candidate_values:

        best_length = max(
            (
                len(variant)
                for variant
                in value_variants(value)
                if variant in text
            ),
            default=0
        )

        if best_length > 0:
            matches.append(
                (
                    best_length,
                    value
                )
            )

    if not matches:
        return None

    matches.sort(
        reverse=True
    )

    # Avoid scoring an ambiguous textual match.
    if (
        len(matches) > 1
        and matches[0][0]
        == matches[1][0]
        and normalize(matches[0][1])
        != normalize(matches[1][1])
    ):
        return None

    return matches[0][1]


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

def infer_entity_and_fact_type(
    scenario
):
    entity = scenario.get(
        "entity"
    )

    fact_type = scenario.get(
        "fact_type"
    )

    for turn in scenario.get(
        "turns",
        []
    ):

        entity = (
            entity
            or turn.get("entity")
        )

        fact_type = (
            fact_type
            or turn.get("fact_type")
        )

        if not entity:
            match = re.search(
                r"INC\d+",
                str(
                    turn.get(
                        "input",
                        ""
                    )
                )
            )

            if match:
                entity = match.group(0)

    return (
        entity,
        str(
            fact_type or ""
        ).lower().strip()
    )


def load_benchmark_cases():
    """
    Common benchmark rules:

    INCLUDED:
        contradiction_expected = True
        explicit correct_resolution exists

    EXCLUDED FROM ACCURACY:
        human_review
        contested / ambiguous cases

    The excluded cases are still counted separately.
    """

    if not SCENARIOS_DIR.exists():
        raise FileNotFoundError(
            f"Scenario directory not found: "
            f"{SCENARIOS_DIR}"
        )

    scored_cases = []
    ambiguous_cases = []

    for path in sorted(
        SCENARIOS_DIR.glob(
            "*.json"
        )
    ):

        with open(
            path,
            encoding="utf-8-sig"
        ) as file:
            data = json.load(file)

        for scenario in data.get(
            "scenarios",
            []
        ):

            ground_truth = (
                scenario.get(
                    "ground_truth",
                    {}
                )
            )

            if not ground_truth.get(
                "contradiction_expected",
                False
            ):
                continue

            entity, fact_type = (
                infer_entity_and_fact_type(
                    scenario
                )
            )

            turns = [
                turn
                for turn
                in scenario.get(
                    "turns",
                    []
                )
                if turn.get("value")
                not in (None, "")
            ]

            if (
                not entity
                or not fact_type
                or len(turns) < 2
            ):
                continue

            correct_value = str(
                ground_truth.get(
                    "correct_resolution",
                    ""
                )
            ).strip()

            resolution_type = str(
                ground_truth.get(
                    "resolution_type",
                    ""
                )
            ).lower().strip()

            is_ambiguous = (
                normalize(
                    correct_value
                )
                in {
                    "humanreview",
                    "contested"
                }
                or resolution_type
                in {
                    "human_review",
                    "flag_contested",
                    "contested"
                }
            )

            prepared = {
                "scenario_id":
                    scenario.get(
                        "scenario_id",
                        path.stem
                    ),
                "entity":
                    entity,
                "fact_type":
                    fact_type,
                "turns":
                    turns
            }

            if is_ambiguous:
                ambiguous_cases.append(
                    prepared
                )
                continue

            if not correct_value:
                continue

            latest_value = str(
                turns[-1]["value"]
            )

            prepared[
                "correct_value"
            ] = correct_value

            prepared[
                "latest_value"
            ] = latest_value

            prepared[
                "case_type"
            ] = (
                "newer_correct"
                if normalize(
                    latest_value
                )
                == normalize(
                    correct_value
                )
                else "newer_wrong"
            )

            scored_cases.append(
                prepared
            )

    scored_cases.sort(
        key=lambda row:
        str(
            row["scenario_id"]
        )
    )

    if BENCHMARK_LIMIT > 0:
        scored_cases = (
            scored_cases[
                :BENCHMARK_LIMIT
            ]
        )

    return (
        scored_cases,
        ambiguous_cases
    )


# ---------------------------------------------------------
# MEM0 ASYNC WRITE HANDLING
# ---------------------------------------------------------

def wait_for_event(
    event_id
):
    """
    Mem0 Platform writes are asynchronous.

    Wait until the ADD event succeeds before
    inserting the next report.
    """

    if not event_id:
        return True

    url = (
        f"{MEM0_API_BASE}"
        f"/v1/event/"
        f"{event_id}/"
    )

    headers = {
        "Authorization":
            f"Token {MEM0_API_KEY}",
        "Accept":
            "application/json"
    }

    deadline = (
        time.time()
        + EVENT_TIMEOUT
    )

    while time.time() < deadline:

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        payload = response.json()

        status = str(
            payload.get(
                "status",
                ""
            )
        ).upper()

        if status in {
            "SUCCEEDED",
            "SUCCESS",
            "COMPLETED"
        }:
            return True

        if status in {
            "FAILED",
            "ERROR"
        }:
            return False

        time.sleep(
            POLL_INTERVAL
        )

    return False


def extract_search_results(
    response
):
    if isinstance(
        response,
        dict
    ):
        return (
            response.get(
                "results",
                []
            )
            or []
        )

    if isinstance(
        response,
        list
    ):
        return response

    return []


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def percentile_95(values):
    if not values:
        return None

    values = sorted(
        values
    )

    position = (
        len(values) - 1
    ) * 0.95

    lower = int(
        position
    )

    upper = min(
        lower + 1,
        len(values) - 1
    )

    fraction = (
        position - lower
    )

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


def subgroup_metrics(
    rows,
    case_type
):
    api_success = [
        row
        for row in rows
        if (
            row["case_type"]
            == case_type
            and row[
                "search_api_success"
            ]
        )
    ]

    returned = [
        row
        for row in api_success
        if row[
            "retrieval_returned"
        ]
    ]

    correct = sum(
        bool(
            row["is_correct"]
        )
        for row in returned
    )

    return {
        "api_success":
            len(api_success),

        "returned":
            len(returned),

        "correct":
            correct,

        "retrieval_coverage":
            (
                len(returned)
                / len(api_success)
                if api_success
                else None
            ),

        "accuracy_on_returned":
            (
                correct
                / len(returned)
                if returned
                else None
            ),

        "effective_accuracy":
            (
                correct
                / len(api_success)
                if api_success
                else None
            )
    }


# ---------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------

def run_mem0_benchmark():

    output_path = get_next_output_path()

    scored_cases, ambiguous = (
        load_benchmark_cases()
    )

    if not scored_cases:
        print(
            "No eligible benchmark "
            "cases found."
        )
        return

    client = MemoryClient(
        api_key=MEM0_API_KEY
    )

    run_id = (
        uuid.uuid4().hex[:8]
    )

    print(
        "=" * 72
    )
    print(
        "MEM0 — CURRENT-FACT "
        "RETRIEVAL BENCHMARK"
    )
    print(
        "=" * 72
    )

    print(
        f"Scored cases: "
        f"{len(scored_cases)}"
    )

    print(
        "Ambiguous human-review "
        f"cases excluded: "
        f"{len(ambiguous)}"
    )

    benchmark_results = []
    latencies = []

    for index, case in enumerate(
        scored_cases,
        start=1
    ):

        user_id = (
            f"verimem_mem0_"
            f"{run_id}_"
            f"{index}_"
            f"{uuid.uuid4().hex[:4]}"
        )

        candidates = (
            get_candidate_values(
                case["turns"]
            )
        )

        row = {
            "scenario_id":
                case["scenario_id"],

            "entity":
                case["entity"],

            "fact_type":
                case["fact_type"],

            "case_type":
                case["case_type"],

            "correct_value":
                case["correct_value"],

            "latest_value":
                case["latest_value"],

            "candidate_values":
                candidates,

            "search_api_success":
                False,

            "retrieval_returned":
                False,

            "is_correct":
                False,

            "error":
                None
        }

        print(
            f"\n[{index}/"
            f"{len(scored_cases)}] "
            f"{case['scenario_id']} | "
            f"{case['fact_type']} | "
            f"{case['case_type']}"
        )

        try:

            # ---------------------------------------------
            # Add every conflicting report in order.
            # ---------------------------------------------

            for turn_index, turn in enumerate(
                case["turns"],
                start=1
            ):

                value = str(
                    turn["value"]
                )

                agent = str(
                    turn.get(
                        "agent",
                        "unknown"
                    )
                )

                source = str(
                    turn.get(
                        "source_dataset",
                        turn.get(
                            "source_file",
                            "unknown"
                        )
                    )
                )

                memory_text = (
                    f"Incident "
                    f"{case['entity']}. "
                    f"The "
                    f"{case['fact_type']} "
                    f"is {value}. "
                    f"Reported by {agent}. "
                    f"Source: {source}."
                )

                add_result = (
                    client.add(
                        messages=[
                            {
                                "role":
                                    "user",
                                "content":
                                    memory_text
                            }
                        ],
                        user_id=user_id,
                        metadata={
                            "benchmark":
                                "verimem",

                            "scenario_id":
                                str(
                                    case[
                                        "scenario_id"
                                    ]
                                ),

                            "fact_type":
                                case[
                                    "fact_type"
                                ],

                            "turn_index":
                                turn_index,

                            "source_agent":
                                agent
                        }
                    )
                )

                if isinstance(
                    add_result,
                    dict
                ):
                    event_id = (
                        add_result.get(
                            "event_id"
                        )
                    )
                else:
                    event_id = getattr(
                        add_result,
                        "event_id",
                        None
                    )

                if not wait_for_event(
                    event_id
                ):
                    raise RuntimeError(
                        "Mem0 ADD event "
                        "failed or timed out: "
                        f"{event_id}"
                    )

            # ---------------------------------------------
            # Search for current fact.
            # Only TOP-1 is scored.
            # ---------------------------------------------

            query = (
                f"What is the current "
                f"{case['fact_type']} "
                f"of incident "
                f"{case['entity']}?"
            )

            start = (
                time.perf_counter()
            )

            response = client.search(
                query,
                filters={
                    "user_id":
                        user_id
                },
                top_k=1
            )

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            row[
                "search_latency_ms"
            ] = round(
                latency_ms,
                2
            )

            latencies.append(
                latency_ms
            )

            row[
                "search_api_success"
            ] = True

            hits = (
                extract_search_results(
                    response
                )
            )

            if hits:

                top = hits[0]

                if isinstance(
                    top,
                    dict
                ):
                    retrieved_text = str(
                        top.get(
                            "memory",
                            top.get(
                                "text",
                                ""
                            )
                        )
                    )

                    search_score = (
                        top.get(
                            "score"
                        )
                    )

                else:
                    retrieved_text = str(
                        top
                    )
                    search_score = None

                matched = (
                    match_candidate(
                        retrieved_text,
                        candidates
                    )
                )

                is_correct = (
                    matched is not None
                    and normalize(
                        matched
                    )
                    == normalize(
                        case[
                            "correct_value"
                        ]
                    )
                )

                row.update({
                    "retrieval_returned":
                        True,

                    "retrieved_text":
                        retrieved_text,

                    "matched_candidate":
                        matched,

                    "search_score":
                        search_score,

                    "is_correct":
                        is_correct
                })

                print(
                    f"  Top-1: "
                    f"{retrieved_text[:100]!r}"
                )

                print(
                    f"  Matched: "
                    f"{matched!r}"
                )

                print(
                    f"  Correct: "
                    f"{is_correct}"
                )

            else:
                print(
                    "  Search succeeded "
                    "but returned no memory."
                )

        except Exception as error:

            row["error"] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                f"  ERROR: "
                f"{row['error']}"
            )

        finally:

            benchmark_results.append(
                row
            )

            # Delete temporary benchmark memories.
            if CLEANUP_AFTER_SCENARIO:
                try:
                    client.delete_all(
                        user_id=user_id
                    )
                except Exception as cleanup_error:
                    print(
                        f"  Cleanup warning: "
                        f"{cleanup_error}"
                    )


    # ---------------------------------------------------------
    # FINAL METRICS
    # ---------------------------------------------------------

    attempted = len(
        benchmark_results
    )

    api_success = [
        row
        for row
        in benchmark_results
        if row[
            "search_api_success"
        ]
    ]

    returned = [
        row
        for row in api_success
        if row[
            "retrieval_returned"
        ]
    ]

    correct = sum(
        bool(
            row["is_correct"]
        )
        for row in returned
    )


    api_success_rate = (
        len(api_success)
        / attempted
        if attempted
        else 0
    )

    retrieval_coverage = (
        len(returned)
        / len(api_success)
        if api_success
        else 0
    )

    accuracy_on_returned = (
        correct
        / len(returned)
        if returned
        else 0
    )

    effective_accuracy = (
        correct
        / len(api_success)
        if api_success
        else 0
    )


    summary = {
        "system":
            "Mem0",

        "attempted_cases":
            attempted,

        "api_success_cases":
            len(api_success),

        "returned_cases":
            len(returned),

        "correct_cases":
            correct,

        "api_success_rate":
            api_success_rate,

        "retrieval_coverage":
            retrieval_coverage,

        "top1_accuracy_on_returned":
            accuracy_on_returned,

        "effective_accuracy":
            effective_accuracy,

        "newer_correct":
            subgroup_metrics(
                benchmark_results,
                "newer_correct"
            ),

        "newer_wrong":
            subgroup_metrics(
                benchmark_results,
                "newer_wrong"
            ),

        "latency_ms": {
            "median":
                statistics.median(
                    latencies
                )
                if latencies
                else None,

            "p95":
                percentile_95(
                    latencies
                )
        },

        # N/A, not zero.
        "explicit_contradiction_detection_f1":
            None,

        "explicit_abstention_evaluated":
            False,

        "ambiguous_cases_available":
            len(ambiguous)
    }


    output = {
        "summary":
            summary,

        "scenario_ids": [
            row[
                "scenario_id"
            ]
            for row
            in benchmark_results
        ],

        "results":
            benchmark_results
    }


    with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        "\n" + "=" * 72
    )

    print(
        "MEM0 SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "API success rate:       "
        f"{api_success_rate:.1%}"
    )

    print(
        "Retrieval coverage:     "
        f"{retrieval_coverage:.1%}"
    )

    print(
        "Top-1 accuracy:         "
        f"{accuracy_on_returned:.1%}"
    )

    print(
        "Effective accuracy:     "
        f"{effective_accuracy:.1%}"
    )

    print(
    f"Saved: {output_path}"
    )


if __name__ == "__main__":
    run_mem0_benchmark()