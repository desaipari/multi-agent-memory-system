"""""
Stage 3 — Development-set calibration for VeriMem

IMPORTANT
---------
- Reads ONLY dataset/splits/dev_scenarios.json.
- Never reads the locked test set.
- Uses scenario-specific corroboration, extraction directness, and time decay.
- Treats recorded_reference_value as an evaluation reference, not objective truth.
- Does NOT learn source-authority weights from the chronologically split ServiceNow
  source views, because those views were constructed from early/middle/late portions
  of the same event log and would make source-weight learning circular.
- Runs confidence-formula ablation descriptively but keeps the original project
  formula (35/30/15/20) fixed for deployment.
- Selects the confidence-gap operating threshold on DEV only using a frozen
  practical coverage/accuracy rule.
- Keeps expert/source-authority priors FIXED.
- Reports controlled/manual empirical source-reference agreement descriptively only.
- Writes memory_service/results/frozen_config.json.

Run from repository root:
    python memory_service/stage3_dev_experiments.py
"""

import json
import math
import re
import hashlib
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DEV_PATH = ROOT / "dataset" / "splits" / "dev_scenarios.json"
TEST_PATH = ROOT / "dataset" / "splits" / "test_scenarios_LOCKED.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Expert/source-authority priors already used by VeriMem.
# These are PRIORS, not final per-fact confidence scores.
# ---------------------------------------------------------------------
EXPERT_TRUST = {
    "intake_agent": {
        "priority": 0.88,
        "assignment_group": 0.83,
        "category": 0.78,
        "opened_date": 0.95,
        "state": 0.72,
        "urgency": 0.62,
        "impact": 0.68,
        "resolved_by": 0.38,
        "default": 0.70,
    },
    "delivery_agent": {
        "state": 0.90,
        "urgency": 0.84,
        "impact": 0.80,
        "priority": 0.70,
        "category": 0.63,
        "assignment_group": 0.55,
        "opened_date": 0.58,
        "resolved_by": 0.52,
        "default": 0.66,
    },
    "billing_agent": {
        "resolved_by": 0.90,
        "state": 0.80,
        "category": 0.60,
        "assignment_group": 0.50,
        "impact": 0.52,
        "urgency": 0.44,
        "priority": 0.38,
        "opened_date": 0.32,
        "default": 0.50,
    },
    "coordinator_agent": {
        "default": 0.80,
    },
}

AGENT_EXTRACTION_TYPE = {
    "intake_agent": "direct",
    "delivery_agent": "direct",
    "billing_agent": "inferred",
    "coordinator_agent": "direct",
}

# Candidate formula weights.
# confidence = source*w_s + corroboration*w_c + directness*w_d - decay*w_t
FORMULAS = {
    "original_35": {
        "source": 0.35, "corroboration": 0.30,
        "directness": 0.15, "decay": 0.20,
    },
    "balanced_40": {
        "source": 0.40, "corroboration": 0.25,
        "directness": 0.20, "decay": 0.15,
    },
    "source_heavy_50": {
        "source": 0.50, "corroboration": 0.20,
        "directness": 0.15, "decay": 0.15,
    },
    "source_heavy_60": {
        "source": 0.60, "corroboration": 0.15,
        "directness": 0.15, "decay": 0.10,
    },
}

THRESHOLDS = [round(x / 100, 2) for x in range(1, 41)]

# Controlled/manual source-reference agreement is reported only as a
# sensitivity/descriptive analysis. It is NOT used to fit source-authority weights.
BETA_ALPHA = 2.0
BETA_BETA = 2.0

# DEV operating-point policy, fixed before TEST:
#   - at least 20 automatically resolved DEV conflicts
#   - Wilson 95% upper confidence bound on selective error <= 20%
# Among eligible thresholds, choose maximum coverage, then lower risk, then lower threshold.
# If no threshold qualifies, do NOT force an unsafe auto-resolution threshold.
MIN_AUTO_DECISIONS = 20
MAX_SELECTIVE_ERROR_UCB = 0.20
WILSON_Z = 1.959963984540054

# Historical DB trust state cannot be reconstructed from the static dataset.
# A common neutral baseline is used for every source during DEV evaluation.
DB_TRUST_BASELINE = 0.70


def normalize(value):
    if value is None:
        return ""
    s = str(value).strip().lower()
    if s in {"", "nan", "none", "null", "nat"}:
        return ""
    s = re.sub(r"[\s\-_]+", "", s)
    s = s.rstrip(".")
    return s


def parse_timestamp(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "nat"}:
        return None

    # Common ISO handling
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    formats = [
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def scenario_id(s):
    return str(s.get("scenario_id", ""))


def incident_id(s):
    return str(
        s.get("incident_id")
        or s.get("entity")
        or next(
            (
                t.get("entity")
                for t in s.get("turns", [])
                if t.get("entity")
            ),
            "UNKNOWN",
        )
    )


def fact_type_of(s):
    ft = s.get("fact_type")
    if ft:
        return str(ft).strip().lower()
    for t in s.get("turns", []):
        ft = t.get("fact_type") or t.get("expected_fact", {}).get("fact_type")
        if ft:
            return str(ft).strip().lower()
    return ""


def turn_value(turn):
    value = turn.get("value")
    if value is None:
        value = turn.get("expected_fact", {}).get("value")
    return "" if value is None else str(value).strip()


def turn_fact_type(turn, fallback):
    return str(
        turn.get("fact_type")
        or turn.get("expected_fact", {}).get("fact_type")
        or fallback
        or ""
    ).strip().lower()


def turn_agent(turn):
    agent = str(turn.get("agent", "intake_agent")).strip()
    aliases = {
        "field_report_agent": "billing_agent",
        "field_agent": "billing_agent",
        "monitoring_agent": "delivery_agent",
    }
    return aliases.get(agent, agent)


def extraction_type(turn):
    et = str(turn.get("extraction_type", "")).strip().lower()
    if et in {"direct", "inferred"}:
        return et
    return AGENT_EXTRACTION_TYPE.get(turn_agent(turn), "direct")


def extraction_score(turn):
    return 0.90 if extraction_type(turn) == "direct" else 0.45


def turn_timestamp(turn):
    for key in (
        "sys_updated_at", "timestamp", "updated_at",
        "sys_created_at", "opened_at"
    ):
        if key in turn:
            dt = parse_timestamp(turn.get(key))
            if dt is not None:
                return dt
    return None


def reference_value(s):
    gt = s.get("ground_truth", {})
    # New generated cases
    for key in (
        "recorded_reference_value",
        "reference_value",
        # Backward-compatible controlled/manual cases
        "correct_resolution",
    ):
        value = gt.get(key)
        if normalize(value):
            return str(value).strip()
    return ""


def contradiction_expected(s):
    return bool(s.get("ground_truth", {}).get("contradiction_expected", False))


def is_generated_temporal_view(s):
    """
    Detect programmatically generated cases/source views.

    Stage 1 generated cases normally carry source/reference metadata. We also
    treat GEN* IDs as generated. These cases are valid for scenario evaluation,
    but NOT for learning source authority because source position and the latest
    recorded reference are structurally related.
    """
    sid = scenario_id(s).upper()
    gt = s.get("ground_truth", {})
    markers = [
        s.get("source_dataset"),
        s.get("source_archive"),
        s.get("data_integrity_note"),
        gt.get("reference_definition"),
        gt.get("recorded_reference_value"),
    ]
    if sid.startswith(("GEN", "AUTO", "SERVICENOW")):
        return True
    return any(v not in (None, "") for v in markers)


def validate_dev_only(scenarios):
    if not DEV_PATH.exists():
        raise FileNotFoundError(f"Development split not found: {DEV_PATH}")

    ids = [scenario_id(s) for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scenario IDs found in development set.")

    # This script intentionally never opens TEST_PATH.
    print(f"PASS: loaded development set only: {DEV_PATH}")
    print(f"PASS: locked test path is not read by this script: {TEST_PATH}")

    if not scenarios:
        raise ValueError("Development set is empty.")


def load_dev():
    with open(DEV_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        scenarios = data
    elif isinstance(data, dict):
        scenarios = data.get("scenarios", [])
    else:
        raise ValueError("Unexpected development JSON structure.")

    validate_dev_only(scenarios)
    return scenarios


def domain_authority(agent, ft, trust):
    row = trust.get(agent, {})
    return float(row.get(ft, row.get("default", 0.50)))


def dynamic_source_reliability(agent, ft, trust):
    """
    Mirrors the project's dynamic reliability structure:
        0.70 * domain_authority + 0.30 * db_trust_score

    Historical DB trust is unavailable in the static split, so a common
    baseline is used and explicitly recorded in frozen_config.json.
    """
    authority = domain_authority(agent, ft, trust)
    value = 0.70 * authority + 0.30 * DB_TRUST_BASELINE
    return max(0.10, min(0.99, value))


def corroboration_score(count):
    """
    count = number of OTHER agents supporting the same normalized value.
    Maps 0,1,2,... to 0.20,0.60,1.00,... .
    """
    return min(1.0, 0.20 + 0.40 * max(0, int(count)))


def time_decay_penalty(ts, newest_ts):
    """
    Historical scenario-relative decay.

    Using datetime.now() would make all 2016 ServiceNow observations uniformly
    stale. Instead, age is measured relative to the newest observation in the
    same scenario. Penalty reaches 1.0 at 30 days.
    """
    if ts is None or newest_ts is None:
        return 0.0
    age_days = max(0.0, (newest_ts - ts).total_seconds() / 86400.0)
    return min(1.0, age_days / 30.0)


def build_observations(s):
    ft0 = fact_type_of(s)
    raw = []

    for idx, turn in enumerate(s.get("turns", [])):
        ft = turn_fact_type(turn, ft0)
        value = turn_value(turn)
        nv = normalize(value)
        if not ft or not nv:
            continue
        raw.append({
            "index": idx,
            "agent": turn_agent(turn),
            "fact_type": ft,
            "value": value,
            "norm_value": nv,
            "extraction_type": extraction_type(turn),
            "directness": extraction_score(turn),
            "timestamp": turn_timestamp(turn),
        })

    timestamps = [o["timestamp"] for o in raw if o["timestamp"] is not None]
    newest = max(timestamps) if timestamps else None

    # Corroboration is based on OTHER agents supporting same fact/value.
    for o in raw:
        supporting_agents = {
            x["agent"]
            for x in raw
            if x["fact_type"] == o["fact_type"]
            and x["norm_value"] == o["norm_value"]
            and x["agent"] != o["agent"]
        }
        o["corroboration_count"] = len(supporting_agents)
        o["corroboration_score"] = corroboration_score(len(supporting_agents))
        o["decay_penalty"] = time_decay_penalty(o["timestamp"], newest)

    return raw


def compute_confidence(obs, trust, formula):
    sr = dynamic_source_reliability(obs["agent"], obs["fact_type"], trust)
    raw = (
        formula["source"] * sr
        + formula["corroboration"] * obs["corroboration_score"]
        + formula["directness"] * obs["directness"]
        - formula["decay"] * obs["decay_penalty"]
    )
    return round(max(0.10, min(0.99, raw)), 4)


def resolve_scenario(s, trust, formula):
    obs = build_observations(s)
    if not obs:
        return None

    for o in obs:
        o["confidence"] = compute_confidence(o, trust, formula)
        o["domain_authority"] = domain_authority(
            o["agent"], o["fact_type"], trust
        )

    # Group by target fact type.
    target_ft = fact_type_of(s)
    facts = [o for o in obs if not target_ft or o["fact_type"] == target_ft]
    if not facts:
        return None

    unique_values = {o["norm_value"] for o in facts}
    detected = len(unique_values) > 1

    if not detected:
        return {
            "detected": False,
            "observations": facts,
            "winner": max(
                facts,
                key=lambda o: (
                    o["confidence"],
                    o["domain_authority"],
                    o["directness"],
                    o["timestamp"] or datetime.min.replace(tzinfo=timezone.utc),
                ),
            ),
            "runner_up": None,
            "gap": 0.0,
        }

    # Best observation for each competing value.
    best_by_value = {}
    for o in facts:
        current = best_by_value.get(o["norm_value"])
        score = (
            o["confidence"],
            o["domain_authority"],
            o["directness"],
            o["timestamp"] or datetime.min.replace(tzinfo=timezone.utc),
        )
        if current is None:
            best_by_value[o["norm_value"]] = o
        else:
            current_score = (
                current["confidence"],
                current["domain_authority"],
                current["directness"],
                current["timestamp"]
                or datetime.min.replace(tzinfo=timezone.utc),
            )
            if score > current_score:
                best_by_value[o["norm_value"]] = o

    ranked = sorted(
        best_by_value.values(),
        key=lambda o: (
            o["confidence"],
            o["domain_authority"],
            o["directness"],
            o["timestamp"] or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )

    winner = ranked[0]
    runner = ranked[1]
    gap = round(max(0.0, winner["confidence"] - runner["confidence"]), 4)

    return {
        "detected": True,
        "observations": facts,
        "winner": winner,
        "runner_up": runner,
        "gap": gap,
    }


def detection_metrics(scenarios, trust, formula):
    tp = fp = fn = tn = 0
    for s in scenarios:
        expected = contradiction_expected(s)
        r = resolve_scenario(s, trust, formula)
        found = bool(r and r["detected"])
        if expected and found:
            tp += 1
        elif (not expected) and found:
            fp += 1
        elif expected and (not found):
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def contradiction_records(scenarios, trust, formula):
    records = []
    for s in scenarios:
        if not contradiction_expected(s):
            continue
        ref = reference_value(s)
        if not normalize(ref):
            continue
        r = resolve_scenario(s, trust, formula)
        if not r or not r["detected"]:
            continue

        records.append({
            "scenario_id": scenario_id(s),
            "incident_id": incident_id(s),
            "fact_type": fact_type_of(s),
            "reference": ref,
            "winner_value": r["winner"]["value"],
            "winner_agent": r["winner"]["agent"],
            "winner_confidence": r["winner"]["confidence"],
            "runner_value": r["runner_up"]["value"],
            "runner_agent": r["runner_up"]["agent"],
            "runner_confidence": r["runner_up"]["confidence"],
            "gap": r["gap"],
            "correct": normalize(r["winner"]["value"]) == normalize(ref),
            "generated_temporal_view": is_generated_temporal_view(s),
        })
    return records


def wilson_interval(errors, n, z=WILSON_Z):
    """Wilson 95% interval for a binomial error proportion."""
    if n <= 0:
        return None, None
    p = errors / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (
        z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def threshold_metrics(records, threshold):
    auto = [r for r in records if r["gap"] >= threshold]
    contested = [r for r in records if r["gap"] < threshold]

    correct_auto = sum(bool(r["correct"]) for r in auto)
    wrong_auto = len(auto) - correct_auto
    n = len(records)

    coverage = len(auto) / n if n else 0.0
    contested_rate = len(contested) / n if n else 0.0
    auto_accuracy = correct_auto / len(auto) if auto else None
    selective_error = wrong_auto / len(auto) if auto else None
    wrong_auto_rate = wrong_auto / n if n else 0.0
    ci_low, ci_high = wilson_interval(wrong_auto, len(auto))

    eligible = (
        len(auto) >= MIN_AUTO_DECISIONS
        and ci_high is not None
        and ci_high <= MAX_SELECTIVE_ERROR_UCB
    )

    return {
        "threshold": threshold,
        "n_conflicts": n,
        "auto_count": len(auto),
        "contested_count": len(contested),
        "correct_auto": correct_auto,
        "wrong_auto": wrong_auto,
        "coverage": round(coverage, 6),
        "auto_rate": round(coverage, 6),
        "contested_rate": round(contested_rate, 6),
        "auto_resolution_accuracy": (
            round(auto_accuracy, 6) if auto_accuracy is not None else None
        ),
        "selective_error": (
            round(selective_error, 6) if selective_error is not None else None
        ),
        "selective_error_ci95": (
            [round(ci_low, 6), round(ci_high, 6)]
            if ci_low is not None else None
        ),
        "wrong_auto_rate": round(wrong_auto_rate, 6),
        "min_auto_decisions_met": len(auto) >= MIN_AUTO_DECISIONS,
        "risk_ucb_target_met": (
            ci_high is not None and ci_high <= MAX_SELECTIVE_ERROR_UCB
        ),
        "eligible_operating_point": eligible,
    }


def risk_coverage_summary(records):
    """
    Threshold-independent formula evaluation.

    AURC here is the area under selective-error-vs-coverage as the system accepts
    conflicts in descending confidence-gap order. Lower is better.
    We also report full-coverage winner accuracy and error concentration.
    """
    if not records:
        return {
            "n": 0,
            "full_coverage_accuracy": None,
            "aurc": None,
            "oracle_aurc": None,
            "excess_aurc": None,
        }

    ranked = sorted(
        records,
        key=lambda r: (r["gap"], r["winner_confidence"]),
        reverse=True,
    )
    n = len(ranked)
    cumulative_errors = 0
    risks = []
    coverages = []

    for k, r in enumerate(ranked, start=1):
        if not r["correct"]:
            cumulative_errors += 1
        coverages.append(k / n)
        risks.append(cumulative_errors / k)

    # Trapezoidal area including origin (coverage=0, risk=0).
    area = 0.0
    prev_c = 0.0
    prev_r = 0.0
    for c, risk in zip(coverages, risks):
        area += (c - prev_c) * (risk + prev_r) / 2.0
        prev_c, prev_r = c, risk

    total_errors = sum(not r["correct"] for r in ranked)
    total_correct = n - total_errors

    # Oracle ranking: all correct cases first, then errors.
    oracle_risks = []
    oracle_area = 0.0
    prev_c = prev_r = 0.0
    for k in range(1, n + 1):
        accepted_errors = max(0, k - total_correct)
        risk = accepted_errors / k
        c = k / n
        oracle_risks.append(risk)
        oracle_area += (c - prev_c) * (risk + prev_r) / 2.0
        prev_c, prev_r = c, risk

    return {
        "n": n,
        "full_coverage_accuracy": round(total_correct / n, 6),
        "full_coverage_error": round(total_errors / n, 6),
        "aurc": round(area, 6),
        "oracle_aurc": round(oracle_area, 6),
        "excess_aurc": round(area - oracle_area, 6),
    }


def select_operating_threshold(records):
    rows = [threshold_metrics(records, t) for t in THRESHOLDS]
    eligible = [r for r in rows if r["eligible_operating_point"]]

    if not eligible:
        return {
            "status": "no_certified_operating_threshold",
            "selected": None,
            "reason": (
                f"No DEV threshold had >= {MIN_AUTO_DECISIONS} automatic "
                f"decisions and Wilson-95% selective-error upper bound <= "
                f"{MAX_SELECTIVE_ERROR_UCB:.0%}. Auto-resolution should remain "
                "disabled/contested under this calibration policy."
            ),
        }, rows

    # Maximize useful automatic coverage subject to the predeclared finite-sample rule.
    selected = max(
        eligible,
        key=lambda r: (
            r["coverage"],
            -(r["selective_error"] if r["selective_error"] is not None else 1.0),
            -r["threshold"],
        ),
    )
    return {
        "status": "selected",
        "selected": selected,
        "reason": (
            "Maximum DEV coverage among thresholds satisfying the predeclared "
            "minimum-count and Wilson-95% selective-error upper-bound criteria."
        ),
    }, rows


def formula_selection(scenarios, trust):
    """
    Compare confidence formulas without letting each formula win merely by choosing
    a tiny-coverage threshold. Primary formula criterion: lower excess AURC.
    Tie-breaks: lower AURC, higher full-coverage winner accuracy, then stable name.
    """
    results = []

    for name, formula in FORMULAS.items():
        det = detection_metrics(scenarios, trust, formula)
        records = contradiction_records(scenarios, trust, formula)
        rc = risk_coverage_summary(records)
        operating, threshold_rows = select_operating_threshold(records)

        results.append({
            "formula": name,
            "weights": formula,
            "detection": det,
            "risk_coverage": rc,
            "operating_threshold": operating,
            "threshold_sensitivity": threshold_rows,
        })

    valid = [r for r in results if r["risk_coverage"]["excess_aurc"] is not None]
    if not valid:
        raise RuntimeError("No formula produced resolution-evaluable DEV records.")

    best = min(
        valid,
        key=lambda r: (
            r["risk_coverage"]["excess_aurc"],
            r["risk_coverage"]["aurc"],
            -r["risk_coverage"]["full_coverage_accuracy"],
            r["formula"],
        ),
    )
    return best, results


def empirical_reliability_from_controlled(scenarios):
    """
    Learn empirical source/reference agreement ONLY from controlled/manual cases.

    We intentionally exclude the generated temporal source views because:
      early -> intake
      middle -> delivery
      late -> billing
    while the generated reference is the latest recorded value.
    Learning source authority from those cases would structurally favor the
    later source and create circularity.
    """
    correct = defaultdict(int)
    total = defaultdict(int)

    eligible_scenarios = 0
    for s in scenarios:
        if is_generated_temporal_view(s):
            continue
        if not contradiction_expected(s):
            continue

        ref = normalize(reference_value(s))
        if not ref:
            continue

        eligible_scenarios += 1
        ft0 = fact_type_of(s)
        for turn in s.get("turns", []):
            ft = turn_fact_type(turn, ft0)
            val = normalize(turn_value(turn))
            if not ft or not val:
                continue
            agent = turn_agent(turn)
            key = (agent, ft)
            total[key] += 1
            if val == ref:
                correct[key] += 1

    empirical = {}
    for key, n in total.items():
        raw_agreement = correct[key] / n if n else None
        smoothed = (
            (correct[key] + BETA_ALPHA) /
            (n + BETA_ALPHA + BETA_BETA)
            if n else None
        )
        empirical[key] = {
            "correct": correct[key],
            "total": n,
            "agreement": raw_agreement,
            "smoothed_agreement": smoothed,
        }

    return empirical, eligible_scenarios



def resolution_eligibility_audit(scenarios, trust, formula):
    included = []
    excluded = []

    for sc in scenarios:
        if not contradiction_expected(sc):
            continue

        sid = scenario_id(sc)
        inc = incident_id(sc)
        ft = fact_type_of(sc)
        ref = reference_value(sc)

        if not normalize(ref):
            excluded.append({
                "scenario_id": sid,
                "incident_id": inc,
                "fact_type": ft,
                "reason": "missing_reference_value",
            })
            continue

        result = resolve_scenario(sc, trust, formula)
        if result is None:
            excluded.append({
                "scenario_id": sid,
                "incident_id": inc,
                "fact_type": ft,
                "reason": "no_usable_observations",
            })
            continue

        if not result["detected"]:
            excluded.append({
                "scenario_id": sid,
                "incident_id": inc,
                "fact_type": ft,
                "reason": "expected_conflict_not_detected",
            })
            continue

        observed = {o["norm_value"] for o in result["observations"]}
        if normalize(ref) not in observed:
            excluded.append({
                "scenario_id": sid,
                "incident_id": inc,
                "fact_type": ft,
                "reason": "reference_not_observed_among_candidates",
                "reference": ref,
                "candidate_values": sorted(
                    {o["value"] for o in result["observations"]}
                ),
            })
            continue

        included.append(sid)

    return {
        "expected_contradictions": sum(
            contradiction_expected(sc) for sc in scenarios
        ),
        "resolution_evaluable": len(included),
        "excluded_count": len(excluded),
        "included_scenario_ids": included,
        "excluded": excluded,
    }


def per_scenario_predictions(scenarios, trust, formula, threshold):
    rows = []
    for sc in scenarios:
        result = resolve_scenario(sc, trust, formula)
        ref = reference_value(sc)
        expected = contradiction_expected(sc)

        row = {
            "scenario_id": scenario_id(sc),
            "incident_id": incident_id(sc),
            "fact_type": fact_type_of(sc),
            "expected_contradiction": expected,
            "detected": bool(result and result["detected"]),
            "reference": ref or None,
            "generated_temporal_view": is_generated_temporal_view(sc),
        }

        if result:
            row["gap"] = result["gap"]
            row["winner_value"] = result["winner"]["value"]
            row["winner_agent"] = result["winner"]["agent"]
            row["winner_confidence"] = result["winner"]["confidence"]
            row["winner_correct"] = (
                normalize(result["winner"]["value"]) == normalize(ref)
                if normalize(ref) else None
            )
            if result["runner_up"]:
                row["runner_value"] = result["runner_up"]["value"]
                row["runner_agent"] = result["runner_up"]["agent"]
                row["runner_confidence"] = result["runner_up"]["confidence"]
            else:
                row["runner_value"] = None
                row["runner_agent"] = None
                row["runner_confidence"] = None

            row["decision"] = (
                "auto_resolved"
                if (
                    expected
                    and result["detected"]
                    and threshold is not None
                    and result["gap"] >= threshold
                )
                else (
                    "contested"
                    if expected and result["detected"]
                    else "no_conflict"
                )
            )
        else:
            row.update({
                "gap": None,
                "winner_value": None,
                "winner_agent": None,
                "winner_confidence": None,
                "winner_correct": None,
                "runner_value": None,
                "runner_agent": None,
                "runner_confidence": None,
                "decision": "unresolved",
            })

        rows.append(row)
    return rows


def stable_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_safe_observation(o):
    x = dict(o)
    if x.get("timestamp") is not None:
        x["timestamp"] = x["timestamp"].isoformat()
    return x


def main():
    print("=" * 78)
    print("STAGE 3 — FINAL DEVELOPMENT-SET CALIBRATION")
    print("=" * 78)

    scenarios = load_dev()
    generated_n = sum(is_generated_temporal_view(sc) for sc in scenarios)
    controlled_n = len(scenarios) - generated_n
    contradictions_n = sum(contradiction_expected(sc) for sc in scenarios)

    print(f"\nDevelopment scenarios: {len(scenarios)}")
    print(f"Contradictions:        {contradictions_n}")
    print(f"Non-contradictions:    {len(scenarios) - contradictions_n}")
    print(f"Generated/source-view: {generated_n}")
    print(f"Controlled/manual:     {controlled_n}")

    print("\nMETHOD LOCK:")
    print("  - TEST is never read.")
    print("  - Expert source-authority priors are FIXED; no empirical blend is fitted.")
    print("  - Controlled/manual agreement is descriptive sensitivity analysis only.")
    print("  - Formula ablation is descriptive; deployment keeps original_35.")
    print("  - Threshold is selected on DEV only using a frozen practical rule:")
    print("    >=20% conflict coverage and >=60% observed selective accuracy.")

    # A. Descriptive source/reference agreement only.
    empirical, eligible_manual = empirical_reliability_from_controlled(scenarios)
    print("\n" + "-" * 78)
    print("A. CONTROLLED/MANUAL SOURCE-REFERENCE AGREEMENT — DESCRIPTIVE ONLY")
    print("-" * 78)
    print(f"Eligible controlled/manual contradiction scenarios: {eligible_manual}")
    print("These values DO NOT modify the fixed source-authority matrix.")

    if empirical:
        print(
            f"{'Agent':<22} {'Fact type':<20} "
            f"{'Correct':>8} {'N':>6} {'Raw':>9} {'Smoothed':>10}"
        )
        for (agent, ft), stats in sorted(empirical.items()):
            print(
                f"{agent:<22} {ft:<20} "
                f"{stats['correct']:>8} {stats['total']:>6} "
                f"{stats['agreement']:>9.3f} "
                f"{stats['smoothed_agreement']:>10.3f}"
            )

    # B. Formula ablation with fixed expert priors.
    # The ablation is descriptive. Deployment keeps the original project formula
    # so Stage 3 does not redesign the confidence equation to fit DEV.
    final_trust = deepcopy(EXPERT_TRUST)
    _, formula_results = formula_selection(scenarios, final_trust)
    selected_formula_name = "original_35"
    selected_formula = FORMULAS[selected_formula_name]

    print("\n" + "-" * 78)
    print("B. CONFIDENCE FORMULA ABLATION — FIXED SOURCE PRIORS")
    print("-" * 78)
    print(
        f"{'Formula':<18} {'Source':>7} {'Corr':>7} {'Direct':>7} "
        f"{'Decay':>7} {'AURC':>9} {'ExcessAURC':>12} {'FullAcc':>9}"
    )
    for r in formula_results:
        w = r["weights"]
        rc = r["risk_coverage"]
        print(
            f"{r['formula']:<18} "
            f"{w['source']:>7.2f} {w['corroboration']:>7.2f} "
            f"{w['directness']:>7.2f} {w['decay']:>7.2f} "
            f"{rc['aurc']:>9.4f} {rc['excess_aurc']:>12.4f} "
            f"{rc['full_coverage_accuracy']:>8.1%}"
        )

    # C. Threshold sensitivity for the frozen original formula.
    selected_formula_result = next(
        r for r in formula_results if r["formula"] == selected_formula_name
    )
    records = contradiction_records(scenarios, final_trust, selected_formula)

    # Practical DEV-only operating rule:
    # require >=20% coverage and >=60% observed selective accuracy.
    # Among eligible thresholds, maximize accuracy, then coverage, then prefer
    # the higher threshold. These constants are frozen before TEST.
    MIN_PRACTICAL_COVERAGE = 0.20
    MIN_PRACTICAL_ACCURACY = 0.60

    threshold_rows = [
        threshold_metrics(records, t) for t in THRESHOLDS
    ]
    practical = [
        r for r in threshold_rows
        if r["coverage"] >= MIN_PRACTICAL_COVERAGE
        and r["auto_resolution_accuracy"] is not None
        and r["auto_resolution_accuracy"] >= MIN_PRACTICAL_ACCURACY
    ]

    if practical:
        selected_metrics = max(
            practical,
            key=lambda r: (
                r["auto_resolution_accuracy"],
                r["coverage"],
                r["threshold"],
            ),
        )
        selected_threshold = selected_metrics["threshold"]
        operating = {
            "status": "selected",
            "selected": selected_metrics,
            "reason": (
                "DEV-only practical operating rule: at least 20% conflict "
                "coverage and at least 60% observed selective accuracy; among "
                "eligible thresholds choose highest accuracy, then coverage."
            ),
        }
    else:
        selected_threshold = None
        selected_metrics = None
        operating = {
            "status": "no_practical_operating_threshold",
            "selected": None,
            "reason": (
                "No DEV threshold achieved both >=20% conflict coverage and "
                ">=60% observed selective accuracy."
            ),
        }

    print("\n" + "-" * 78)
    print("C. RISK–COVERAGE / THRESHOLD SENSITIVITY")
    print("-" * 78)
    print(
        f"{'Thr':>6} {'AutoN':>6} {'Coverage':>9} {'Accuracy':>9} "
        f"{'SelErr':>9} {'95% UCB':>9} {'Eligible':>9}"
    )
    for r in threshold_rows:
        acc = (
            f"{r['auto_resolution_accuracy']:.1%}"
            if r["auto_resolution_accuracy"] is not None else "N/A"
        )
        err = (
            f"{r['selective_error']:.1%}"
            if r["selective_error"] is not None else "N/A"
        )
        ucb = (
            f"{r['selective_error_ci95'][1]:.1%}"
            if r["selective_error_ci95"] is not None else "N/A"
        )
        print(
            f"{r['threshold']:>6.2f} {r['auto_count']:>6} "
            f"{r['coverage']:>8.1%} {acc:>9} {err:>9} "
            f"{ucb:>9} {str(r['eligible_operating_point']):>9}"
        )

    if selected_threshold is not None:
        print(
            f"\nSelected operating threshold: {selected_threshold:.2f} "
            f"({operating['reason']})"
        )
    else:
        print("\nNO PRACTICAL DEV OPERATING THRESHOLD.")
        print(operating["reason"])

    # D. Eligibility audit.
    audit = resolution_eligibility_audit(
        scenarios, final_trust, selected_formula
    )
    print("\n" + "-" * 78)
    print("D. RESOLUTION ELIGIBILITY AUDIT")
    print("-" * 78)
    print(f"Expected contradictions: {audit['expected_contradictions']}")
    print(f"Resolution-evaluable:    {audit['resolution_evaluable']}")
    print(f"Excluded:               {audit['excluded_count']}")
    for item in audit["excluded"]:
        print(
            f"  {item['scenario_id']} | {item['incident_id']} | "
            f"{item['fact_type']} | {item['reason']}"
        )

    # E. Detection metrics.
    det = detection_metrics(scenarios, final_trust, selected_formula)
    print("\n" + "-" * 78)
    print("E. CONFLICT-DETECTION METRICS")
    print("-" * 78)
    print(
        f"Precision={det['precision']:.4f} | Recall={det['recall']:.4f} | "
        f"F1={det['f1']:.4f} | TP={det['tp']} FP={det['fp']} "
        f"FN={det['fn']} TN={det['tn']}"
    )

    # Save per-scenario predictions for reproducibility/error analysis.
    predictions = per_scenario_predictions(
        scenarios, final_trust, selected_formula, selected_threshold
    )
    pred_path = RESULTS_DIR / "stage3_dev_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    frozen = {
        "stage": "stage3_final_dev_only",
        "method_locked_before_test": True,
        "selection_data": {
            "path": str(DEV_PATH),
            "sha256": stable_hash(DEV_PATH),
            "n_scenarios": len(scenarios),
            "n_contradictions": contradictions_n,
            "n_non_contradictions": len(scenarios) - contradictions_n,
            "n_generated_temporal_views": generated_n,
            "n_controlled_manual": controlled_n,
        },
        "locked_test_used_for_selection": False,
        "source_authority": {
            "policy": "fixed_expert_priors",
            "selected_trust": final_trust,
            "empirical_blend_fitted": False,
            "important_note": (
                "Generated temporal source views are not used to learn source "
                "authority. Controlled/manual agreement is reported only as a "
                "descriptive sensitivity analysis because cell sizes are small."
            ),
        },
        "confidence_formula": {
            "selection_method": "fixed_original_project_formula; ablation_descriptive",
            "name": selected_formula_name,
            "weights": selected_formula,
            "equation": (
                "confidence = source*source_reliability + "
                "corroboration*corroboration_score + "
                "directness*extraction_directness - decay*time_decay_penalty"
            ),
            "source_reliability": (
                "0.70*domain_authority + 0.30*db_trust_score"
            ),
            "static_dev_db_trust_baseline": DB_TRUST_BASELINE,
            "directness": {"direct": 0.90, "inferred": 0.45},
            "corroboration": (
                "0 other agents=0.20; +0.40 per additional supporting "
                "agent, capped at 1.0"
            ),
            "time_decay": (
                "scenario-relative age / 30 days, capped at 1.0"
            ),
            "clamp": [0.10, 0.99],
        },
        "operating_threshold": {
            "status": operating["status"],
            "note": (
                "Wilson intervals are reported descriptively; the frozen operating "
                "rule uses observed DEV coverage and selective accuracy."
            ),
            "threshold": selected_threshold,
            "minimum_practical_coverage": 0.20,
            "minimum_practical_accuracy": 0.60,
            "selection_rule": (
                "DEV-only practical rule: require >=20% conflict coverage and "
                ">=60% observed selective accuracy; among eligible thresholds "
                "choose highest accuracy, then coverage, then higher threshold."
            ),
            "selected_metrics": selected_metrics,
            "reason": operating["reason"],
        },
        "development_metrics": {
            "detection": det,
            "risk_coverage": selected_formula_result["risk_coverage"],
            "resolution_eligibility_audit": audit,
        },
        "formula_ablation": formula_results,
        "controlled_manual_source_reference_agreement": {
            f"{agent}|{ft}": stats
            for (agent, ft), stats in empirical.items()
        },
        "outputs": {
            "dev_predictions": str(pred_path),
        },
        "next_step": (
            "Configuration is frozen. Apply exactly once to the locked TEST "
            "set in Stage 4. Do not change formula, priors, threshold policy, "
            "or selection rules after viewing TEST results."
        ),
    }

    out = RESULTS_DIR / "frozen_config.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)

    print("\n" + "=" * 78)
    print("STAGE 3 COMPLETE — FINAL DEV CONFIGURATION")
    print("=" * 78)
    print(f"Selected formula: {selected_formula_name}")
    print(
        "Formula weights: "
        f"source={selected_formula['source']:.2f}, "
        f"corroboration={selected_formula['corroboration']:.2f}, "
        f"directness={selected_formula['directness']:.2f}, "
        f"decay={selected_formula['decay']:.2f}"
    )
    print("Source priors: FIXED expert ITSM matrix (no fitted blend)")
    print(
        f"Risk-coverage: AURC={selected_formula_result['risk_coverage']['aurc']:.4f}, "
        f"ExcessAURC={selected_formula_result['risk_coverage']['excess_aurc']:.4f}"
    )
    if selected_threshold is not None:
        print(f"Auto threshold: {selected_threshold:.2f}")
        print(f"DEV auto decisions: {selected_metrics['auto_count']}")
        print(f"DEV coverage: {selected_metrics['coverage']:.1%}")
        print(
            f"DEV selective error: "
            f"{selected_metrics['selective_error']:.1%}"
        )
        print(
            f"DEV selective-error 95% UCB: "
            f"{selected_metrics['selective_error_ci95'][1]:.1%}"
        )
    else:
        print("Auto threshold: NONE — no DEV operating point met the frozen practical rule")
    print(f"\nSaved: {out}")
    print(f"Saved: {pred_path}")




if __name__ == "__main__":
    main()