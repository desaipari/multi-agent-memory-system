"""
VeriMem benchmark on the exact same IT-incident scenarios
used by benchmark_mem0.py and benchmark_zep.py.

This benchmark has two purposes:

COMMON COMPARISON
-----------------
For the 18 labelled contradiction cases:
- Top-ranked candidate accuracy
- Newer-correct accuracy
- Newer-wrong safety

These metrics can be compared with Mem0 and Zep.

VERIMEM-SPECIFIC SELECTIVE RESOLUTION
-------------------------------------
Also reports:
- Auto-resolution coverage
- Auto-resolution accuracy
- Contested rate
- Ambiguous-case contest rate

Important:
- Uses the frozen V2 resolver directly.
- Uses the final calibrated thresholds:
      winner score >= 0.35
      winner margin >= 0.15
- Uses noisy-OR corroboration.
- Uses prior strength m=5.
- Does NOT train trust on benchmark ground truth.
- Each scenario is independent.
"""

import json
import os
import re
import sys

from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MEMORY_SERVICE_DIR = (
    ROOT / "memory_service"
)

sys.path.insert(
    0,
    str(MEMORY_SERVICE_DIR)
)


SCENARIOS_DIR = (
    ROOT / "dataset" / "scenarios"
)

OUTPUT = (
    MEMORY_SERVICE_DIR
    / "verimem_benchmark_results.json"
)


# ---------------------------------------------------------
# VERIMEM V2 IMPORTS
# ---------------------------------------------------------

from verimem_core.types import Observation

from verimem_core.providers import (
    InMemoryTrustProvider
)

from verimem_core.resolver import (
    resolve,
    normalize_value
)

from verimem_core.config import (
    PRIOR_STRENGTH,
    MIN_WINNER_SCORE,
    MIN_WINNER_MARGIN,
)


CORROBORATION_METHOD = "noisy_or"


# ---------------------------------------------------------
# AGENT NAME NORMALIZATION
# ---------------------------------------------------------
#
# Some older scenario files use historical names:
#
#     monitoring_agent
#     field_report_agent
#
# Current VeriMem uses:
#
#     delivery_agent
#     billing_agent
#
# These are role-name aliases, not changes to the evidence.
# ---------------------------------------------------------

AGENT_ALIASES = {
    "monitoring_agent":
        "delivery_agent",

    "field_report_agent":
        "billing_agent",

    "ticket_intake_agent":
        "intake_agent",
}


def canonical_agent(agent_id):

    value = str(
        agent_id or ""
    ).strip()

    return AGENT_ALIASES.get(
        value,
        value
    )


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize(value):

    if value is None:
        return ""

    return normalize_value(
        str(value)
    )


# ---------------------------------------------------------
# TIMESTAMP PARSING
# ---------------------------------------------------------

def parse_timestamp(turn):
    """
    Use an actual observation/event timestamp only when
    the scenario provides one.

    We deliberately DO NOT replace missing observation
    time with ingestion order.

    This matches VeriMem V2:
    missing observation time receives neutral temporal
    evidence inside the resolver.
    """

    candidates = [
        turn.get("observed_at"),
        turn.get("timestamp"),
        turn.get("timestamp_str"),
        turn.get("sys_updated_at"),
    ]

    for raw in candidates:

        if raw in (
            None,
            ""
        ):
            continue

        text = str(
            raw
        ).strip()

        # ISO timestamp first.
        try:

            iso = text.replace(
                "Z",
                "+00:00"
            )

            parsed = (
                datetime.fromisoformat(
                    iso
                )
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=timezone.utc
                )

            return parsed

        except ValueError:
            pass


        # Formats found in the scenario files.
        formats = [
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]


        for fmt in formats:

            try:

                parsed = (
                    datetime.strptime(
                        text,
                        fmt
                    )
                )

                return parsed.replace(
                    tzinfo=timezone.utc
                )

            except ValueError:
                continue


    return None


# ---------------------------------------------------------
# DATASET HELPERS
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
    """
    Uses the same inclusion logic as the Mem0/Zep
    benchmark.

    Scored:
        contradiction_expected = True
        and a concrete correct_resolution exists

    Ambiguous:
        human_review / contested cases

    Therefore the common comparison should contain
    exactly the same 18 scored cases if the same
    dataset/scenarios folder is used.
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
                    turns,

                "source_file":
                    path.name,
            }


            if is_ambiguous:

                ambiguous_cases.append(
                    prepared
                )

                continue


            if not correct_value:
                continue


            latest_value = str(
                turns[-1][
                    "value"
                ]
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
            row[
                "scenario_id"
            ]
        )
    )

    ambiguous_cases.sort(
        key=lambda row:
        str(
            row[
                "scenario_id"
            ]
        )
    )


    return (
        scored_cases,
        ambiguous_cases
    )


# ---------------------------------------------------------
# CONVERT SCENARIO TO VERIMEM OBSERVATIONS
# ---------------------------------------------------------

def to_observations(case):

    observations = []


    for index, turn in enumerate(
        case["turns"],
        start=1
    ):

        agent_id = (
            canonical_agent(
                turn.get(
                    "agent",
                    "unknown"
                )
            )
        )


        observations.append(
            Observation(
                fact_id=(
                    f"{case['scenario_id']}"
                    f"_turn_{index}"
                ),

                agent_id=
                    agent_id,

                fact_type=
                    case[
                        "fact_type"
                    ],

                value=str(
                    turn[
                        "value"
                    ]
                ),

                extraction_type=str(
                    turn.get(
                        "extraction_type",
                        "direct"
                    )
                    or "direct"
                ),

                observed_at=
                    parse_timestamp(
                        turn
                    ),
            )
        )


    return observations


# ---------------------------------------------------------
# ONE SCENARIO
# ---------------------------------------------------------

def evaluate_case(
    case,
    provider
):

    observations = (
        to_observations(
            case
        )
    )


    result = resolve(
        observations=
            observations,

        provider=
            provider,

        prior_strength=
            PRIOR_STRENGTH,

        corroboration_method=
            CORROBORATION_METHOD,

        min_winner_score=
            MIN_WINNER_SCORE,

        min_margin=
            MIN_WINNER_MARGIN,
    )


    winner = result.get(
        "winner"
    )

    runner = result.get(
        "runner_up"
    )


    winner_value = (
        winner.get(
            "value"
        )
        if winner
        else None
    )


    winner_score = (
        winner.get(
            "score"
        )
        if winner
        else None
    )


    runner_value = (
        runner.get(
            "value"
        )
        if runner
        else None
    )


    runner_score = (
        runner.get(
            "score"
        )
        if runner
        else None
    )


    margin = result.get(
        "margin"
    )


    correct = (
        winner_value is not None
        and normalize(
            winner_value
        )
        == normalize(
            case[
                "correct_value"
            ]
        )
    )


    return {
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

        "decision":
            result.get(
                "decision"
            ),

        "winner_value":
            winner_value,

        "winner_score":
            winner_score,

        "runner_up_value":
            runner_value,

        "runner_up_score":
            runner_score,

        "margin":
            margin,

        "is_correct":
            correct,

        "candidate_count":
            len(
                result.get(
                    "candidates",
                    []
                )
            ),

        "observations": [
            {
                "agent_id":
                    observation.agent_id,

                "value":
                    observation.value,

                "extraction_type":
                    observation.extraction_type,

                "observed_at":
                    (
                        observation
                        .observed_at
                        .isoformat()

                        if observation
                        .observed_at

                        else None
                    ),
            }

            for observation
            in observations
        ],
    }


# ---------------------------------------------------------
# AMBIGUOUS CASE
# ---------------------------------------------------------

def evaluate_ambiguous_case(
    case,
    provider
):

    result = resolve(
        observations=
            to_observations(
                case
            ),

        provider=
            provider,

        prior_strength=
            PRIOR_STRENGTH,

        corroboration_method=
            CORROBORATION_METHOD,

        min_winner_score=
            MIN_WINNER_SCORE,

        min_margin=
            MIN_WINNER_MARGIN,
    )


    winner = result.get(
        "winner"
    )


    return {
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

        "decision":
            result.get(
                "decision"
            ),

        "winner_value":
            (
                winner.get(
                    "value"
                )
                if winner
                else None
            ),

        "winner_score":
            (
                winner.get(
                    "score"
                )
                if winner
                else None
            ),

        "margin":
            result.get(
                "margin"
            ),

        "correct_behavior":
            (
                result.get(
                    "decision"
                )
                == "contested"
            ),
    }


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def safe_ratio(
    numerator,
    denominator
):

    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )


def subgroup_metrics(
    rows,
    case_type
):

    group = [
        row
        for row in rows
        if row[
            "case_type"
        ]
        == case_type
    ]


    correct = sum(
        bool(
            row[
                "is_correct"
            ]
        )
        for row in group
    )


    auto = [
        row
        for row in group
        if row[
            "decision"
        ]
        == "auto_resolve"
    ]


    correct_auto = sum(
        bool(
            row[
                "is_correct"
            ]
        )
        for row in auto
    )


    contested = sum(
        row[
            "decision"
        ]
        == "contested"

        for row in group
    )


    return {
        "total":
            len(group),

        "top_ranked_correct":
            correct,

        "top_ranked_accuracy":
            safe_ratio(
                correct,
                len(group)
            ),

        "auto_resolved":
            len(auto),

        "auto_resolution_coverage":
            safe_ratio(
                len(auto),
                len(group)
            ),

        "correct_auto_resolutions":
            correct_auto,

        "auto_resolution_accuracy":
            safe_ratio(
                correct_auto,
                len(auto)
            ),

        "contested":
            contested,

        "contested_rate":
            safe_ratio(
                contested,
                len(group)
            ),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def run_verimem_benchmark():

    scored_cases, ambiguous_cases = (
        load_benchmark_cases()
    )


    print(
        "=" * 72
    )

    print(
        "VERIMEM — CURRENT-FACT "
        "AND SELECTIVE-RESOLUTION BENCHMARK"
    )

    print(
        "=" * 72
    )


    print(
        f"Scored cases: "
        f"{len(scored_cases)}"
    )

    print(
        "Ambiguous human-review cases: "
        f"{len(ambiguous_cases)}"
    )


    print(
        "\nFrozen V2 configuration:"
    )

    print(
        "  Prior strength:        "
        f"{PRIOR_STRENGTH}"
    )

    print(
        "  Corroboration:         "
        f"{CORROBORATION_METHOD}"
    )

    print(
        "  Minimum winner score:  "
        f"{MIN_WINNER_SCORE}"
    )

    print(
        "  Minimum margin:        "
        f"{MIN_WINNER_MARGIN}"
    )


    # -----------------------------------------------------
    # Fresh provider.
    #
    # No benchmark labels are used for adaptive trust.
    # Each scenario therefore starts from the frozen
    # expert priors rather than learning from test cases.
    # -----------------------------------------------------

    provider = (
        InMemoryTrustProvider()
    )


    results = []


    for index, case in enumerate(
        scored_cases,
        start=1
    ):

        row = evaluate_case(
            case,
            provider
        )

        results.append(
            row
        )


        print(
            f"\n[{index}/"
            f"{len(scored_cases)}] "
            f"{row['scenario_id']} | "
            f"{row['fact_type']} | "
            f"{row['case_type']}"
        )


        print(
            "  Winner:   "
            f"{row['winner_value']!r} "
            f"(score="
            f"{row['winner_score']})"
        )


        print(
            "  Correct:  "
            f"{row['is_correct']}"
        )


        print(
            "  Decision: "
            f"{row['decision']}"
        )


        print(
            "  Margin:   "
            f"{row['margin']}"
        )


    # -----------------------------------------------------
    # AMBIGUOUS / HUMAN-REVIEW CASES
    # -----------------------------------------------------

    ambiguous_results = []


    if ambiguous_cases:

        print(
            "\n" + "=" * 72
        )

        print(
            "AMBIGUOUS CASES"
        )

        print(
            "=" * 72
        )


        for case in ambiguous_cases:

            row = (
                evaluate_ambiguous_case(
                    case,
                    provider
                )
            )

            ambiguous_results.append(
                row
            )


            print(
                f"{row['scenario_id']} | "
                f"{row['fact_type']} | "
                f"decision="
                f"{row['decision']} | "
                f"contested="
                f"{row['correct_behavior']}"
            )


    # -----------------------------------------------------
    # COMMON TOP-RANKED RETRIEVAL METRIC
    # -----------------------------------------------------

    total = len(
        results
    )


    correct = sum(
        bool(
            row[
                "is_correct"
            ]
        )
        for row in results
    )


    top_ranked_accuracy = (
        safe_ratio(
            correct,
            total
        )
    )


    # -----------------------------------------------------
    # VERIMEM SELECTIVE RESOLUTION METRICS
    # -----------------------------------------------------

    auto_rows = [
        row
        for row in results
        if row[
            "decision"
        ]
        == "auto_resolve"
    ]


    contested_rows = [
        row
        for row in results
        if row[
            "decision"
        ]
        == "contested"
    ]


    correct_auto = sum(
        bool(
            row[
                "is_correct"
            ]
        )
        for row in auto_rows
    )


    auto_coverage = (
        safe_ratio(
            len(auto_rows),
            total
        )
    )


    auto_accuracy = (
        safe_ratio(
            correct_auto,
            len(auto_rows)
        )
    )


    contested_rate = (
        safe_ratio(
            len(contested_rows),
            total
        )
    )


    ambiguous_correct = sum(
        bool(
            row[
                "correct_behavior"
            ]
        )
        for row
        in ambiguous_results
    )


    ambiguous_contest_rate = (
        safe_ratio(
            ambiguous_correct,
            len(
                ambiguous_results
            )
        )
    )


    newer_correct = (
        subgroup_metrics(
            results,
            "newer_correct"
        )
    )


    newer_wrong = (
        subgroup_metrics(
            results,
            "newer_wrong"
        )
    )


    summary = {
        "system":
            "VeriMem V2",

        "benchmark_protocol":
            (
                "Same labelled contradiction "
                "cases used for Mem0 and Zep"
            ),

        "configuration": {
            "prior_strength":
                PRIOR_STRENGTH,

            "corroboration_method":
                CORROBORATION_METHOD,

            "minimum_winner_score":
                MIN_WINNER_SCORE,

            "minimum_winner_margin":
                MIN_WINNER_MARGIN,

            "benchmark_trust_learning":
                False,
        },

        # ---------------------------------------------
        # COMMON METRIC
        # ---------------------------------------------

        "common_retrieval": {
            "total_cases":
                total,

            "correct_top_ranked":
                correct,

            "top_ranked_accuracy":
                top_ranked_accuracy,
        },

        # ---------------------------------------------
        # BEHAVIOR SPLIT
        # ---------------------------------------------

        "newer_correct":
            newer_correct,

        "newer_wrong":
            newer_wrong,

        # ---------------------------------------------
        # VERIMEM-SPECIFIC SELECTIVE METRICS
        # ---------------------------------------------

        "selective_resolution": {
            "auto_resolved":
                len(auto_rows),

            "auto_resolution_coverage":
                auto_coverage,

            "correct_auto_resolutions":
                correct_auto,

            "auto_resolution_accuracy":
                auto_accuracy,

            "contested":
                len(
                    contested_rows
                ),

            "contested_rate":
                contested_rate,
        },

        "ambiguous_cases": {
            "total":
                len(
                    ambiguous_results
                ),

            "correctly_contested":
                ambiguous_correct,

            "ambiguous_contest_rate":
                ambiguous_contest_rate,
        },
    }


    output = {
        "summary":
            summary,

        "scenario_ids": [
            row[
                "scenario_id"
            ]
            for row in results
        ],

        "results":
            results,

        "ambiguous_results":
            ambiguous_results,
    }


    with open(
        OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )


    # -----------------------------------------------------
    # TERMINAL SUMMARY
    # -----------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "VERIMEM SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        "Top-ranked accuracy:          "
        f"{top_ranked_accuracy:.1%}"
    )


    print(
        "Newer-correct accuracy:       "
        f"{newer_correct['top_ranked_accuracy']:.1%}"
    )


    print(
        "Newer-wrong safety:           "
        f"{newer_wrong['top_ranked_accuracy']:.1%}"
    )


    print(
        "Auto-resolution coverage:     "
        f"{auto_coverage:.1%}"
    )


    if auto_accuracy is not None:

        print(
            "Auto-resolution accuracy:     "
            f"{auto_accuracy:.1%}"
        )

    else:

        print(
            "Auto-resolution accuracy:     "
            "N/A"
        )


    print(
        "Contested rate:               "
        f"{contested_rate:.1%}"
    )


    if (
        ambiguous_contest_rate
        is not None
    ):

        print(
            "Ambiguous contest rate:       "
            f"{ambiguous_contest_rate:.1%}"
        )


    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    run_verimem_benchmark()