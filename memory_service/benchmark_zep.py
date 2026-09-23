"""
Zep benchmark for VeriMem IT-incident contradiction scenarios.

Smoke test:
    $env:ZEP_API_KEY="your_key"
    $env:BENCHMARK_LIMIT="5"
    python memory_service/benchmark_zep.py

Run all eligible labelled conflicts:
    $env:BENCHMARK_LIMIT="0"
    python memory_service/benchmark_zep.py
"""

import json
import os
import re
import time
import uuid
import statistics

from pathlib import Path
from datetime import (
    datetime,
    timezone,
    timedelta
)

try:
    from zep_cloud.client import Zep
except ImportError:
    raise SystemExit(
        "Install Zep first: "
        "pip install -U zep-cloud"
    )


ROOT = Path(__file__).resolve().parents[1]

SCENARIOS_DIR = (
    ROOT / "dataset" / "scenarios"
)

RESULTS_DIR = Path(__file__).with_name("benchmark_runs")
RESULTS_DIR.mkdir(exist_ok=True)


def get_next_output_path():
    existing = list(
        RESULTS_DIR.glob("zep_benchmark_run_*.json")
    )

    run_numbers = []

    for path in existing:
        match = re.search(
            r"zep_benchmark_run_(\d+)\.json$",
            path.name
        )
        if match:
            run_numbers.append(int(match.group(1)))

    next_run = max(run_numbers, default=0) + 1

    return RESULTS_DIR / f"zep_benchmark_run_{next_run:02d}.json"


ZEP_API_KEY = os.getenv(
    "ZEP_API_KEY",
    ""
).strip()


BENCHMARK_LIMIT = int(
    os.getenv(
        "BENCHMARK_LIMIT",
        "5"
    )
)

POLL_INTERVAL = float(
    os.getenv(
        "ZEP_POLL_INTERVAL",
        "3"
    )
)

PROCESS_TIMEOUT = int(
    os.getenv(
        "ZEP_PROCESS_TIMEOUT",
        "120"
    )
)

CLEANUP_AFTER_SCENARIO = (
    os.getenv(
        "BENCHMARK_CLEANUP",
        "1"
    ) != "0"
)


if not ZEP_API_KEY:
    raise SystemExit(
        'Set ZEP_API_KEY first:\n'
        '$env:ZEP_API_KEY="your_key"'
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

    raw = str(
        value or ""
    ).strip()

    stripped = re.sub(
        r"^\s*\d+\s*[-_:./ ]*\s*",
        "",
        raw
    )

    return {
        v
        for v in {
            normalize(raw),
            normalize(stripped)
        }
        if v
    }


def get_candidate_values(turns):

    values = []
    seen = set()

    for turn in turns:

        value = turn.get(
            "value"
        )

        if value in (
            None,
            ""
        ):
            continue

        key = normalize(
            value
        )

        if key not in seen:
            seen.add(
                key
            )
            values.append(
                str(value)
            )

    return values


def match_candidate(
    retrieved_text,
    candidate_values
):

    text = normalize(
        retrieved_text
    )

    matches = []

    for value in candidate_values:

        best_length = max(
            (
                len(variant)
                for variant
                in value_variants(
                    value
                )
                if variant in text
            ),
            default=0
        )

        if best_length:
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

    if (
        len(matches) > 1
        and matches[0][0]
        == matches[1][0]
        and normalize(
            matches[0][1]
        )
        != normalize(
            matches[1][1]
        )
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
            or turn.get(
                "entity"
            )
        )

        fact_type = (
            fact_type
            or turn.get(
                "fact_type"
            )
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
                entity = (
                    match.group(0)
                )

    return (
        entity,
        str(
            fact_type or ""
        ).lower().strip()
    )


def load_benchmark_cases():

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

            data = json.load(
                file
            )

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
                if turn.get(
                    "value"
                )
                not in (
                    None,
                    ""
                )
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


            prepared.update({
                "correct_value":
                    correct_value,

                "latest_value":
                    latest_value,

                "case_type":
                    (
                        "newer_correct"
                        if normalize(
                            latest_value
                        )
                        == normalize(
                            correct_value
                        )
                        else "newer_wrong"
                    )
            })


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
# ZEP HELPERS
# ---------------------------------------------------------

def get_episode_uuid(
    episode
):

    return (
        getattr(
            episode,
            "uuid_",
            None
        )
        or getattr(
            episode,
            "uuid",
            None
        )
    )


def wait_for_episode(
    client,
    episode_uuid
):
    """
    Wait until Zep has processed the last
    report in this scenario.
    """

    if not episode_uuid:
        return False

    deadline = (
        time.time()
        + PROCESS_TIMEOUT
    )

    while time.time() < deadline:

        episode = (
            client.graph.episode.get(
                uuid_=episode_uuid
            )
        )

        if bool(
            getattr(
                episode,
                "processed",
                False
            )
        ):
            return True

        time.sleep(
            POLL_INTERVAL
        )

    return False


def get_edges(
    result
):

    edges = getattr(
        result,
        "edges",
        None
    )

    if edges is not None:
        return list(
            edges
        )

    if isinstance(
        result,
        dict
    ):
        return list(
            result.get(
                "edges",
                []
            )
            or []
        )

    return []


def edge_field(
    edge,
    field,
    default=None
):

    if isinstance(
        edge,
        dict
    ):
        return edge.get(
            field,
            default
        )

    return getattr(
        edge,
        field,
        default
    )


def edge_fact(edge):

    return str(
        edge_field(
            edge,
            "fact",
            ""
        )
        or ""
    )


def is_current_edge(edge):
    """
    Zep exposes temporal validity.

    A current fact is treated as one that
    is neither invalidated nor expired.
    """

    invalid_at = (
        edge_field(
            edge,
            "invalid_at"
        )
    )

    expired_at = (
        edge_field(
            edge,
            "expired_at"
        )
    )

    return (
        invalid_at in (
            None,
            ""
        )
        and expired_at in (
            None,
            ""
        )
    )


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
    case_type,
    metric_prefix
):

    api_success = [
        row
        for row in rows
        if (
            row[
                "case_type"
            ]
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
            metric_prefix
            + "_returned"
        ]
    ]


    correct = sum(
        bool(
            row[
                metric_prefix
                + "_correct"
            ]
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

def run_zep_benchmark():

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


    client = Zep(
        api_key=ZEP_API_KEY
    )

    run_id = (
        uuid.uuid4().hex[:8]
    )

    benchmark_results = []
    latencies = []


    print(
        "=" * 72
    )

    print(
        "ZEP — CURRENT-FACT "
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

    print(
        "\nTwo Zep metrics will be saved:"
    )

    print(
        "1. raw_top1 = first ranked edge"
    )

    print(
        "2. current_valid = first "
        "non-invalidated edge"
    )


    base_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc
    )


    for index, case in enumerate(
        scored_cases,
        start=1
    ):

        user_id = (
            f"verimem_zep_"
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
                case[
                    "scenario_id"
                ],

            "entity":
                case[
                    "entity"
                ],

            "fact_type":
                case[
                    "fact_type"
                ],

            "case_type":
                case[
                    "case_type"
                ],

            "correct_value":
                case[
                    "correct_value"
                ],

            "latest_value":
                case[
                    "latest_value"
                ],

            "candidate_values":
                candidates,

            "search_api_success":
                False,

            "raw_top1_returned":
                False,

            "raw_top1_correct":
                False,

            "current_valid_returned":
                False,

            "current_valid_correct":
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
            # One unique Zep user per scenario.
            # This prevents graph contamination.
            # ---------------------------------------------

            client.user.add(
                user_id=user_id,
                first_name="Benchmark",
                last_name=(
                    f"Scenario{index}"
                )
            )


            last_episode = None


            # ---------------------------------------------
            # Add reports in their original order.
            # ---------------------------------------------

            for turn_index, turn in enumerate(
                case["turns"],
                start=1
            ):

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


                record = {
                    "incident_id":
                        case[
                            "entity"
                        ],

                    "fact_type":
                        case[
                            "fact_type"
                        ],

                    "value":
                        str(
                            turn[
                                "value"
                            ]
                        ),

                    "source_agent":
                        agent,

                    "source":
                        source,

                    "report_order":
                        turn_index
                }


                # Stable artificial event order.
                event_time = (
                    base_time
                    + timedelta(
                        seconds=
                            turn_index
                    )
                )

                created_at = (
                    event_time
                    .isoformat()
                    .replace(
                        "+00:00",
                        "Z"
                    )
                )


                last_episode = (
                    client.graph.add(
                        user_id=user_id,

                        type="json",

                        data=json.dumps(
                            record
                        ),

                        created_at=
                            created_at,

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

                            "source_agent":
                                agent
                        },

                        source_description=(
                            "VeriMem IT "
                            "incident benchmark "
                            "report"
                        )
                    )
                )


            last_episode_uuid = (
                get_episode_uuid(
                    last_episode
                )
            )


            if not wait_for_episode(
                client,
                last_episode_uuid
            ):

                raise RuntimeError(
                    "Zep processing timed "
                    "out for episode "
                    f"{last_episode_uuid}"
                )


            # ---------------------------------------------
            # Search Zep fact edges.
            # ---------------------------------------------

            query = (
                f"current "
                f"{case['fact_type']} "
                f"for incident "
                f"{case['entity']}"
            )


            start = (
                time.perf_counter()
            )


            search_result = (
                client.graph.search(
                    user_id=user_id,

                    query=query,

                    scope="edges",

                    limit=5,

                    reranker=
                        "cross_encoder"
                )
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


            edges = get_edges(
                search_result
            )


            # ---------------------------------------------
            # METRIC 1:
            # Raw first-ranked edge.
            #
            # This is the closest comparison to
            # Mem0's top-1 retrieval.
            # ---------------------------------------------

            if edges:

                raw_edge = edges[0]

                raw_text = edge_fact(
                    raw_edge
                )

                raw_match = (
                    match_candidate(
                        raw_text,
                        candidates
                    )
                )

                raw_correct = (
                    raw_match is not None
                    and normalize(
                        raw_match
                    )
                    == normalize(
                        case[
                            "correct_value"
                        ]
                    )
                )


                row.update({
                    "raw_top1_returned":
                        True,

                    "raw_top1_text":
                        raw_text,

                    "raw_top1_matched_candidate":
                        raw_match,

                    "raw_top1_correct":
                        raw_correct,

                    "raw_top1_invalid_at":
                        str(
                            edge_field(
                                raw_edge,
                                "invalid_at"
                            )
                            or ""
                        ),

                    "raw_top1_expired_at":
                        str(
                            edge_field(
                                raw_edge,
                                "expired_at"
                            )
                            or ""
                        )
                })


            # ---------------------------------------------
            # METRIC 2:
            # Use Zep's native temporal validity metadata.
            #
            # Historical invalidated facts are skipped.
            # ---------------------------------------------

            current_edges = [
                edge
                for edge in edges
                if is_current_edge(
                    edge
                )
            ]


            if current_edges:

                current_edge = (
                    current_edges[0]
                )

                current_text = (
                    edge_fact(
                        current_edge
                    )
                )

                current_match = (
                    match_candidate(
                        current_text,
                        candidates
                    )
                )

                current_correct = (
                    current_match
                    is not None
                    and normalize(
                        current_match
                    )
                    == normalize(
                        case[
                            "correct_value"
                        ]
                    )
                )


                row.update({
                    "current_valid_returned":
                        True,

                    "current_valid_text":
                        current_text,

                    "current_valid_matched_candidate":
                        current_match,

                    "current_valid_correct":
                        current_correct
                })


            print(
                "  Raw top-1: "
                f"{row.get('raw_top1_text', '')[:100]!r}"
            )

            print(
                "  Raw correct: "
                f"{row['raw_top1_correct']}"
            )

            print(
                "  Current-valid: "
                f"{row.get('current_valid_text', '')[:100]!r}"
            )

            print(
                "  Current-valid correct: "
                f"{row['current_valid_correct']}"
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

            if CLEANUP_AFTER_SCENARIO:

                try:
                    client.user.delete(
                        user_id=user_id
                    )
                except Exception:
                    pass


    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    api_success = [
        row
        for row
        in benchmark_results
        if row[
            "search_api_success"
        ]
    ]


    def calculate_metric(
        prefix
    ):

        returned = [
            row
            for row
            in api_success
            if row[
                prefix
                + "_returned"
            ]
        ]

        correct = sum(
            bool(
                row[
                    prefix
                    + "_correct"
                ]
            )
            for row in returned
        )

        return {
            "api_success_cases":
                len(api_success),

            "returned_cases":
                len(returned),

            "correct_cases":
                correct,

            "retrieval_coverage":
                (
                    len(returned)
                    / len(api_success)
                    if api_success
                    else 0
                ),

            "accuracy_on_returned":
                (
                    correct
                    / len(returned)
                    if returned
                    else 0
                ),

            "effective_accuracy":
                (
                    correct
                    / len(api_success)
                    if api_success
                    else 0
                )
        }


    raw_metric = (
        calculate_metric(
            "raw_top1"
        )
    )

    current_metric = (
        calculate_metric(
            "current_valid"
        )
    )


    summary = {
        "system":
            "Zep",

        "attempted_cases":
            len(
                benchmark_results
            ),

        "api_success_cases":
            len(
                api_success
            ),

        "api_success_rate":
            (
                len(api_success)
                / len(
                    benchmark_results
                )
                if benchmark_results
                else 0
            ),

        # Fair/common retrieval result.
        "raw_top1":
            raw_metric,

        # Zep-native temporal result.
        "current_valid":
            current_metric,

        "raw_top1_newer_correct":
            subgroup_metrics(
                benchmark_results,
                "newer_correct",
                "raw_top1"
            ),

        "raw_top1_newer_wrong":
            subgroup_metrics(
                benchmark_results,
                "newer_wrong",
                "raw_top1"
            ),

        "current_valid_newer_correct":
            subgroup_metrics(
                benchmark_results,
                "newer_correct",
                "current_valid"
            ),

        "current_valid_newer_wrong":
            subgroup_metrics(
                benchmark_results,
                "newer_wrong",
                "current_valid"
            ),

        "latency_ms": {
            "median":
                (
                    statistics.median(
                        latencies
                    )
                    if latencies
                    else None
                ),

            "p95":
                percentile_95(
                    latencies
                )
        },

        "explicit_contradiction_detection_f1":
            None,

        "explicit_abstention_evaluated":
            False,

        "ambiguous_cases_available":
            len(
                ambiguous
            )
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
        "ZEP SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        "API success rate:             "
        f"{summary['api_success_rate']:.1%}"
    )

    print(
        "Raw top-1 accuracy:           "
        f"{raw_metric['accuracy_on_returned']:.1%}"
    )

    print(
        "Raw top-1 effective accuracy: "
        f"{raw_metric['effective_accuracy']:.1%}"
    )

    print(
        "Current-valid accuracy:       "
        f"{current_metric['accuracy_on_returned']:.1%}"
    )

    print(
        "Current-valid effective:      "
        f"{current_metric['effective_accuracy']:.1%}"
    )

    print(
    f"Saved: {output_path}"
)


if __name__ == "__main__":
    run_zep_benchmark()