"""
Stage 4 — Final held-out TEST evaluation for VeriMem

Purpose
-------
- Reads ONLY the locked TEST split for evaluation.
- Loads the frozen Stage 3 configuration.
- NEVER tunes formula weights, source priors, or threshold on TEST.
- Applies the frozen configuration exactly once.
- Reports:
    * contradiction detection precision / recall / F1
    * resolution eligibility audit
    * winner/reference accuracy
    * frozen-policy auto-resolution coverage and selective accuracy/error
    * contested rate
    * risk-coverage AURC / excess AURC
    * breakdown by fact type
    * breakdown by source combination
    * generated vs controlled/manual breakdown
    * incident-cluster bootstrap 95% confidence intervals
    * complete per-scenario predictions
    * complete error analysis
- Saves reproducible JSON outputs.

Run from repository root:
    python memory_service/stage4_final_verimem_test.py
"""

import json
import math
import re
import hashlib
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
BOOTSTRAP_REPS = 2000

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "dataset" / "splits" / "test_scenarios_LOCKED.json"
FROZEN_PATH = Path(__file__).resolve().parent / "results" / "frozen_config.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "stage4_test_predictions.json"
RESULTS_PATH = RESULTS_DIR / "stage4_test_results.json"
ERRORS_PATH = RESULTS_DIR / "stage4_error_analysis.json"

AGENT_EXTRACTION_TYPE = {
    "intake_agent": "direct",
    "delivery_agent": "direct",
    "billing_agent": "inferred",
    "coordinator_agent": "direct",
}


def normalize(value):
    if value is None:
        return ""
    s = str(value).strip().lower()
    if s in {"", "nan", "none", "null", "nat"}:
        return ""
    s = re.sub(r"[\s\-_]+", "", s)
    return s.rstrip(".")


def parse_timestamp(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "nat"}:
        return None

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


def stable_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def scenario_id(s):
    return str(s.get("scenario_id", ""))


def incident_id(s):
    return str(
        s.get("incident_id")
        or s.get("entity")
        or next(
            (t.get("entity") for t in s.get("turns", []) if t.get("entity")),
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


def extraction_score(turn, directness_cfg):
    et = extraction_type(turn)
    return float(directness_cfg.get(et, 0.90 if et == "direct" else 0.45))


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
    for key in (
        "recorded_reference_value",
        "reference_value",
        "correct_resolution",
    ):
        value = gt.get(key)
        if normalize(value):
            return str(value).strip()
    return ""


def contradiction_expected(s):
    return bool(s.get("ground_truth", {}).get("contradiction_expected", False))


def is_generated_temporal_view(s):
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


def source_combination(s):
    agents = sorted(
        {
            turn_agent(t)
            for t in s.get("turns", [])
            if normalize(turn_value(t))
        }
    )
    return " + ".join(agents) if agents else "unknown"


def load_json_scenarios(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("scenarios", [])
    raise ValueError(f"Unexpected JSON structure in {path}")


def load_frozen_config():
    if not FROZEN_PATH.exists():
        raise FileNotFoundError(
            f"Frozen Stage 3 configuration not found: {FROZEN_PATH}"
        )
    with open(FROZEN_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    if cfg.get("stage") != "stage3_final_dev_only":
        raise ValueError(
            "frozen_config.json is not marked as the final DEV-only Stage 3 config."
        )
    if cfg.get("locked_test_used_for_selection") is not False:
        raise ValueError(
            "Frozen config does not certify that TEST was unused for selection."
        )
    if cfg.get("method_locked_before_test") is not True:
        raise ValueError("Frozen config is not marked as locked before TEST.")

    return cfg


def validate_test(scenarios, frozen):
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Locked TEST split not found: {TEST_PATH}")
    if not scenarios:
        raise ValueError("Locked TEST set is empty.")

    ids = [scenario_id(s) for s in scenarios]
    if any(not x for x in ids):
        raise ValueError("At least one TEST scenario has no scenario_id.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scenario IDs found in locked TEST.")

    dev_path = Path(frozen["selection_data"]["path"])
    if dev_path.resolve() == TEST_PATH.resolve():
        raise ValueError("Frozen selection path points to TEST. Aborting.")

    expected_test_count = 841
    if len(scenarios) != expected_test_count:
        raise ValueError(
            f"Expected locked TEST to contain {expected_test_count} scenarios, "
            f"but found {len(scenarios)}. Do not continue until verified."
        )

    print(f"PASS: loaded LOCKED TEST only for final evaluation: {TEST_PATH}")
    print(f"PASS: frozen config came from DEV: {dev_path}")
    print("PASS: no parameter selection is performed by Stage 4.")


def domain_authority(agent, ft, trust):
    row = trust.get(agent, {})
    return float(row.get(ft, row.get("default", 0.50)))


def source_reliability(agent, ft, trust, db_trust_baseline):
    authority = domain_authority(agent, ft, trust)
    value = 0.70 * authority + 0.30 * db_trust_baseline
    return max(0.10, min(0.99, value))


def corroboration_score(count):
    return min(1.0, 0.20 + 0.40 * max(0, int(count)))


def time_decay_penalty(ts, newest_ts):
    if ts is None or newest_ts is None:
        return 0.0
    age_days = max(0.0, (newest_ts - ts).total_seconds() / 86400.0)
    return min(1.0, age_days / 30.0)


def build_observations(s, directness_cfg):
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
            "directness": extraction_score(turn, directness_cfg),
            "timestamp": turn_timestamp(turn),
            "source_dataset": turn.get("source_dataset"),
        })

    timestamps = [o["timestamp"] for o in raw if o["timestamp"] is not None]
    newest = max(timestamps) if timestamps else None

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


def compute_confidence(obs, trust, formula, db_trust_baseline):
    sr = source_reliability(
        obs["agent"], obs["fact_type"], trust, db_trust_baseline
    )
    raw = (
        formula["source"] * sr
        + formula["corroboration"] * obs["corroboration_score"]
        + formula["directness"] * obs["directness"]
        - formula["decay"] * obs["decay_penalty"]
    )
    return round(max(0.10, min(0.99, raw)), 4)


def resolve_scenario(s, trust, formula, directness_cfg, db_trust_baseline):
    obs = build_observations(s, directness_cfg)
    if not obs:
        return None

    for o in obs:
        o["confidence"] = compute_confidence(
            o, trust, formula, db_trust_baseline
        )
        o["domain_authority"] = domain_authority(
            o["agent"], o["fact_type"], trust
        )

    target_ft = fact_type_of(s)
    facts = [o for o in obs if not target_ft or o["fact_type"] == target_ft]
    if not facts:
        return None

    unique_values = {o["norm_value"] for o in facts}
    detected = len(unique_values) > 1

    def ranking_key(o):
        return (
            o["confidence"],
            o["domain_authority"],
            o["directness"],
            o["timestamp"] or datetime.min.replace(tzinfo=timezone.utc),
        )

    if not detected:
        return {
            "detected": False,
            "observations": facts,
            "winner": max(facts, key=ranking_key),
            "runner_up": None,
            "gap": 0.0,
        }

    best_by_value = {}
    for o in facts:
        current = best_by_value.get(o["norm_value"])
        if current is None or ranking_key(o) > ranking_key(current):
            best_by_value[o["norm_value"]] = o

    ranked = sorted(best_by_value.values(), key=ranking_key, reverse=True)
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


def detection_counts(predictions):
    tp = fp = fn = tn = 0
    for p in predictions:
        expected = p["expected_contradiction"]
        found = p["detected"]
        if expected and found:
            tp += 1
        elif (not expected) and found:
            fp += 1
        elif expected and (not found):
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def detection_metrics_from_predictions(predictions):
    tp, fp, fn, tn = detection_counts(predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (tp + tn) / (tp + fp + fn + tn) if predictions else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def resolution_eligibility_reason(p):
    if not p["expected_contradiction"]:
        return "not_expected_contradiction"
    if not normalize(p.get("reference")):
        return "missing_reference_value"
    if not p.get("has_usable_observations"):
        return "no_usable_observations"
    if not p["detected"]:
        return "expected_conflict_not_detected"
    if not p.get("reference_observed_among_candidates"):
        return "reference_not_observed_among_candidates"
    return None


def make_predictions(
    scenarios, trust, formula, directness_cfg, db_trust_baseline, threshold
):
    rows = []

    for sc in scenarios:
        result = resolve_scenario(
            sc, trust, formula, directness_cfg, db_trust_baseline
        )
        ref = reference_value(sc)
        expected = contradiction_expected(sc)

        row = {
            "scenario_id": scenario_id(sc),
            "incident_id": incident_id(sc),
            "fact_type": fact_type_of(sc),
            "source_combination": source_combination(sc),
            "generated_temporal_view": is_generated_temporal_view(sc),
            "expected_contradiction": expected,
            "reference": ref or None,
            "detected": bool(result and result["detected"]),
            "has_usable_observations": result is not None,
        }

        if result:
            observed_norm = {o["norm_value"] for o in result["observations"]}
            row["candidate_values"] = sorted(
                {o["value"] for o in result["observations"]}
            )
            row["reference_observed_among_candidates"] = (
                normalize(ref) in observed_norm if normalize(ref) else False
            )
            row["gap"] = result["gap"]
            row["winner_value"] = result["winner"]["value"]
            row["winner_agent"] = result["winner"]["agent"]
            row["winner_confidence"] = result["winner"]["confidence"]
            row["winner_correct"] = (
                normalize(result["winner"]["value"]) == normalize(ref)
                if normalize(ref) else None
            )

            if result["runner_up"] is not None:
                row["runner_value"] = result["runner_up"]["value"]
                row["runner_agent"] = result["runner_up"]["agent"]
                row["runner_confidence"] = result["runner_up"]["confidence"]
            else:
                row["runner_value"] = None
                row["runner_agent"] = None
                row["runner_confidence"] = None
        else:
            row.update({
                "candidate_values": [],
                "reference_observed_among_candidates": False,
                "gap": None,
                "winner_value": None,
                "winner_agent": None,
                "winner_confidence": None,
                "winner_correct": None,
                "runner_value": None,
                "runner_agent": None,
                "runner_confidence": None,
            })

        reason = resolution_eligibility_reason(row)
        row["resolution_evaluable"] = reason is None
        row["resolution_exclusion_reason"] = reason

        if expected and row["detected"]:
            if threshold is not None and row["gap"] >= threshold:
                row["decision"] = "auto_resolved"
            else:
                row["decision"] = "contested"
        elif row["detected"]:
            row["decision"] = "false_positive_conflict"
        else:
            row["decision"] = "no_conflict"

        rows.append(row)

    return rows


def risk_coverage_summary(records):
    if not records:
        return {
            "n": 0,
            "full_coverage_accuracy": None,
            "full_coverage_error": None,
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
    area = 0.0
    prev_c = 0.0
    prev_r = 0.0

    for k, r in enumerate(ranked, start=1):
        if not r["winner_correct"]:
            cumulative_errors += 1
        c = k / n
        risk = cumulative_errors / k
        area += (c - prev_c) * (risk + prev_r) / 2.0
        prev_c, prev_r = c, risk

    total_errors = sum(not r["winner_correct"] for r in ranked)
    total_correct = n - total_errors

    oracle_area = 0.0
    prev_c = 0.0
    prev_r = 0.0
    for k in range(1, n + 1):
        accepted_errors = max(0, k - total_correct)
        risk = accepted_errors / k
        c = k / n
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


def resolution_metrics(predictions, threshold):
    expected = [p for p in predictions if p["expected_contradiction"]]
    evaluable = [p for p in expected if p["resolution_evaluable"]]
    excluded = [p for p in expected if not p["resolution_evaluable"]]

    correct_winners = sum(bool(p["winner_correct"]) for p in evaluable)
    winner_accuracy = (
        correct_winners / len(evaluable) if evaluable else None
    )

    auto = [
        p for p in evaluable
        if threshold is not None and p["gap"] >= threshold
    ]
    contested = [
        p for p in evaluable
        if threshold is None or p["gap"] < threshold
    ]

    correct_auto = sum(bool(p["winner_correct"]) for p in auto)
    wrong_auto = len(auto) - correct_auto
    auto_accuracy = correct_auto / len(auto) if auto else None
    selective_error = wrong_auto / len(auto) if auto else None

    return {
        "expected_contradictions": len(expected),
        "resolution_evaluable": len(evaluable),
        "excluded_count": len(excluded),
        "exclusion_reasons": dict(Counter(
            p["resolution_exclusion_reason"] for p in excluded
        )),
        "winner_reference_accuracy": (
            round(winner_accuracy, 6) if winner_accuracy is not None else None
        ),
        "winner_correct": correct_winners,
        "winner_total": len(evaluable),
        "frozen_threshold": threshold,
        "auto_count": len(auto),
        "correct_auto": correct_auto,
        "wrong_auto": wrong_auto,
        "auto_resolution_coverage": (
            round(len(auto) / len(evaluable), 6) if evaluable else None
        ),
        "auto_resolution_accuracy": (
            round(auto_accuracy, 6) if auto_accuracy is not None else None
        ),
        "selective_error": (
            round(selective_error, 6) if selective_error is not None else None
        ),
        "contested_count": len(contested),
        "contested_rate": (
            round(len(contested) / len(evaluable), 6) if evaluable else None
        ),
        "risk_coverage": risk_coverage_summary(evaluable),
    }


def subset_metrics(rows):
    det = detection_metrics_from_predictions(rows)
    res = resolution_metrics(rows, threshold=None)
    return {
        "n_scenarios": len(rows),
        "detection": det,
        "resolution_evaluable": res["resolution_evaluable"],
        "winner_reference_accuracy": res["winner_reference_accuracy"],
    }


def grouped_breakdown(predictions, field):
    groups = defaultdict(list)
    for p in predictions:
        groups[str(p.get(field, "unknown"))].append(p)
    return {
        key: subset_metrics(rows)
        for key, rows in sorted(groups.items())
    }


def bootstrap_metric_by_incident(predictions, metric_fn, reps=BOOTSTRAP_REPS):
    groups = defaultdict(list)
    for p in predictions:
        groups[p["incident_id"]].append(p)

    incident_ids = list(groups)
    if len(incident_ids) < 2:
        return None

    rng = random.Random(SEED)
    values = []

    for _ in range(reps):
        sampled = [rng.choice(incident_ids) for _ in incident_ids]
        rows = []
        for new_cluster_idx, inc in enumerate(sampled):
            # A sampled incident may appear multiple times in a cluster bootstrap.
            # Duplicate its rows and give the duplicate a temporary cluster label.
            for p in groups[inc]:
                q = dict(p)
                q["_bootstrap_cluster"] = new_cluster_idx
                rows.append(q)

        value = metric_fn(rows)
        if value is not None and math.isfinite(value):
            values.append(float(value))

    if not values:
        return None

    values.sort()

    def percentile(p):
        if len(values) == 1:
            return values[0]
        x = (len(values) - 1) * p
        lo = int(math.floor(x))
        hi = int(math.ceil(x))
        if lo == hi:
            return values[lo]
        return values[lo] + (x - lo) * (values[hi] - values[lo])

    return {
        "method": "incident-cluster nonparametric bootstrap percentile interval",
        "seed": SEED,
        "repetitions": reps,
        "n_incidents": len(incident_ids),
        "lower_95": round(percentile(0.025), 6),
        "upper_95": round(percentile(0.975), 6),
    }


def detection_f1_value(rows):
    return detection_metrics_from_predictions(rows)["f1"]


def detection_precision_value(rows):
    return detection_metrics_from_predictions(rows)["precision"]


def detection_recall_value(rows):
    return detection_metrics_from_predictions(rows)["recall"]


def winner_accuracy_value(rows):
    evaluable = [
        p for p in rows
        if p["expected_contradiction"] and p["resolution_evaluable"]
    ]
    if not evaluable:
        return None
    return sum(bool(p["winner_correct"]) for p in evaluable) / len(evaluable)


def auto_accuracy_value(rows, threshold):
    evaluable = [
        p for p in rows
        if p["expected_contradiction"]
        and p["resolution_evaluable"]
        and threshold is not None
        and p["gap"] >= threshold
    ]
    if not evaluable:
        return None
    return sum(bool(p["winner_correct"]) for p in evaluable) / len(evaluable)


def auto_coverage_value(rows, threshold):
    evaluable = [
        p for p in rows
        if p["expected_contradiction"] and p["resolution_evaluable"]
    ]
    if not evaluable:
        return None
    auto = [
        p for p in evaluable
        if threshold is not None and p["gap"] >= threshold
    ]
    return len(auto) / len(evaluable)


def build_error_analysis(predictions, threshold):
    failures = []

    for p in predictions:
        reasons = []

        if p["expected_contradiction"] and not p["detected"]:
            reasons.append("false_negative_conflict_detection")
        if (not p["expected_contradiction"]) and p["detected"]:
            reasons.append("false_positive_conflict_detection")
        if p["expected_contradiction"] and not p["resolution_evaluable"]:
            reasons.append(
                f"resolution_excluded:{p['resolution_exclusion_reason']}"
            )
        if (
            p["expected_contradiction"]
            and p["resolution_evaluable"]
            and p["winner_correct"] is False
        ):
            reasons.append("incorrect_top_ranked_value")
        if (
            p["expected_contradiction"]
            and p["resolution_evaluable"]
            and threshold is not None
            and p["gap"] >= threshold
            and p["winner_correct"] is False
        ):
            reasons.append("incorrect_auto_resolution")

        if reasons:
            failures.append({
                "scenario_id": p["scenario_id"],
                "incident_id": p["incident_id"],
                "fact_type": p["fact_type"],
                "source_combination": p["source_combination"],
                "generated_temporal_view": p["generated_temporal_view"],
                "failure_reasons": reasons,
                "expected_contradiction": p["expected_contradiction"],
                "detected": p["detected"],
                "reference": p["reference"],
                "candidate_values": p["candidate_values"],
                "winner_value": p["winner_value"],
                "winner_agent": p["winner_agent"],
                "winner_confidence": p["winner_confidence"],
                "runner_value": p["runner_value"],
                "runner_agent": p["runner_agent"],
                "runner_confidence": p["runner_confidence"],
                "confidence_gap": p["gap"],
                "decision": p["decision"],
            })

    return {
        "n_failure_records": len(failures),
        "failure_type_counts": dict(Counter(
            reason
            for item in failures
            for reason in item["failure_reasons"]
        )),
        "failures": failures,
    }


def main():
    print("=" * 78)
    print("STAGE 4 — FINAL LOCKED-TEST EVALUATION")
    print("=" * 78)

    frozen = load_frozen_config()
    scenarios = load_json_scenarios(TEST_PATH)
    validate_test(scenarios, frozen)

    formula_cfg = frozen["confidence_formula"]
    formula = formula_cfg["weights"]
    trust = frozen["source_authority"]["selected_trust"]
    directness_cfg = formula_cfg.get(
        "directness", {"direct": 0.90, "inferred": 0.45}
    )
    db_trust_baseline = float(
        formula_cfg.get("static_dev_db_trust_baseline", 0.70)
    )
    threshold = frozen["operating_threshold"].get("threshold")

    if threshold is not None:
        threshold = float(threshold)

    print("\nFROZEN CONFIGURATION")
    print(f"  Formula:  {formula_cfg['name']}")
    print(
        "  Weights:  "
        f"source={formula['source']:.2f}, "
        f"corroboration={formula['corroboration']:.2f}, "
        f"directness={formula['directness']:.2f}, "
        f"decay={formula['decay']:.2f}"
    )
    print("  Source authority: fixed Stage 3 expert priors")
    print(f"  DB trust baseline: {db_trust_baseline:.2f}")
    print(
        f"  Auto threshold: {'NONE' if threshold is None else f'{threshold:.2f}'}"
    )
    print("  TEST parameters will NOT be tuned.")

    predictions = make_predictions(
        scenarios,
        trust,
        formula,
        directness_cfg,
        db_trust_baseline,
        threshold,
    )

    det = detection_metrics_from_predictions(predictions)
    res = resolution_metrics(predictions, threshold)

    print("\n" + "-" * 78)
    print("A. LOCKED TEST COMPOSITION")
    print("-" * 78)
    print(f"Total TEST scenarios: {len(predictions)}")
    print(
        "Expected contradictions: "
        f"{sum(p['expected_contradiction'] for p in predictions)}"
    )
    print(
        "Expected non-contradictions: "
        f"{sum(not p['expected_contradiction'] for p in predictions)}"
    )
    print(
        "Unique incidents: "
        f"{len({p['incident_id'] for p in predictions})}"
    )

    print("\n" + "-" * 78)
    print("B. CONTRADICTION-DETECTION METRICS")
    print("-" * 78)
    print(
        f"Precision={det['precision']:.4f} | "
        f"Recall={det['recall']:.4f} | "
        f"F1={det['f1']:.4f} | "
        f"Accuracy={det['accuracy']:.4f}"
    )
    print(
        f"TP={det['tp']} FP={det['fp']} "
        f"FN={det['fn']} TN={det['tn']}"
    )

    print("\n" + "-" * 78)
    print("C. RESOLUTION / SELECTIVE POLICY")
    print("-" * 78)
    print(f"Resolution-evaluable: {res['resolution_evaluable']}")
    print(f"Excluded:              {res['excluded_count']}")
    if res["exclusion_reasons"]:
        for reason, n in sorted(res["exclusion_reasons"].items()):
            print(f"  {reason}: {n}")

    if res["winner_reference_accuracy"] is not None:
        print(
            f"Top-ranked/reference accuracy: "
            f"{res['winner_reference_accuracy']:.1%} "
            f"({res['winner_correct']}/{res['winner_total']})"
        )

    if threshold is not None:
        print(f"Frozen threshold:       {threshold:.2f}")
        print(
            f"Auto decisions:         {res['auto_count']} "
            f"({res['auto_resolution_coverage']:.1%} coverage)"
        )
        if res["auto_resolution_accuracy"] is not None:
            print(
                f"Auto accuracy:          "
                f"{res['auto_resolution_accuracy']:.1%}"
            )
            print(
                f"Selective error:        "
                f"{res['selective_error']:.1%}"
            )
        print(
            f"Contested:              {res['contested_count']} "
            f"({res['contested_rate']:.1%})"
        )
    else:
        print("Auto-resolution disabled by frozen Stage 3 policy.")

    rc = res["risk_coverage"]
    print(
        f"AURC={rc['aurc']} | OracleAURC={rc['oracle_aurc']} | "
        f"ExcessAURC={rc['excess_aurc']}"
    )

    print("\n" + "-" * 78)
    print("D. INCIDENT-CLUSTER BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("-" * 78)

    cis = {
        "detection_precision": bootstrap_metric_by_incident(
            predictions, detection_precision_value
        ),
        "detection_recall": bootstrap_metric_by_incident(
            predictions, detection_recall_value
        ),
        "detection_f1": bootstrap_metric_by_incident(
            predictions, detection_f1_value
        ),
        "winner_reference_accuracy": bootstrap_metric_by_incident(
            predictions, winner_accuracy_value
        ),
        "auto_resolution_coverage": bootstrap_metric_by_incident(
            predictions, lambda rows: auto_coverage_value(rows, threshold)
        ),
        "auto_resolution_accuracy": bootstrap_metric_by_incident(
            predictions, lambda rows: auto_accuracy_value(rows, threshold)
        ),
    }

    for name, ci in cis.items():
        if ci is None:
            print(f"{name}: N/A")
        else:
            print(
                f"{name}: [{ci['lower_95']:.4f}, "
                f"{ci['upper_95']:.4f}]"
            )

    fact_breakdown = grouped_breakdown(predictions, "fact_type")
    source_breakdown = grouped_breakdown(predictions, "source_combination")

    generated_rows = [
        p for p in predictions if p["generated_temporal_view"]
    ]
    controlled_rows = [
        p for p in predictions if not p["generated_temporal_view"]
    ]
    provenance_breakdown = {
        "generated_temporal_views": subset_metrics(generated_rows),
        "controlled_manual": subset_metrics(controlled_rows),
    }

    print("\n" + "-" * 78)
    print("E. FACT-TYPE BREAKDOWN")
    print("-" * 78)
    for ft, m in fact_breakdown.items():
        d = m["detection"]
        wa = m["winner_reference_accuracy"]
        wa_s = "N/A" if wa is None else f"{wa:.1%}"
        print(
            f"{ft:<20} n={m['n_scenarios']:<4} "
            f"DetF1={d['f1']:.3f} "
            f"ResN={m['resolution_evaluable']:<4} "
            f"WinnerAcc={wa_s}"
        )

    error_analysis = build_error_analysis(predictions, threshold)

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)

    final_output = {
        "stage": "stage4_final_locked_test",
        "test_used_for_parameter_selection": False,
        "configuration_frozen_before_test": True,
        "test_set": {
            "path": str(TEST_PATH),
            "sha256": stable_hash(TEST_PATH),
            "n_scenarios": len(predictions),
            "n_unique_incidents": len(
                {p["incident_id"] for p in predictions}
            ),
            "n_expected_contradictions": sum(
                p["expected_contradiction"] for p in predictions
            ),
            "n_expected_non_contradictions": sum(
                not p["expected_contradiction"] for p in predictions
            ),
        },
        "frozen_config": {
            "path": str(FROZEN_PATH),
            "sha256": stable_hash(FROZEN_PATH),
            "formula_name": formula_cfg["name"],
            "formula_weights": formula,
            "source_authority_policy": frozen["source_authority"]["policy"],
            "db_trust_baseline": db_trust_baseline,
            "auto_resolution_threshold": threshold,
        },
        "detection": det,
        "resolution": res,
        "confidence_intervals_95": cis,
        "breakdown_by_fact_type": fact_breakdown,
        "breakdown_by_source_combination": source_breakdown,
        "breakdown_by_provenance": provenance_breakdown,
        "error_analysis_summary": {
            "n_failure_records": error_analysis["n_failure_records"],
            "failure_type_counts": error_analysis["failure_type_counts"],
        },
        "outputs": {
            "predictions": str(PREDICTIONS_PATH),
            "error_analysis": str(ERRORS_PATH),
        },
        "method_note": (
            "All parameters were loaded from the frozen DEV-selected Stage 3 "
            "configuration. No TEST-set tuning, threshold search, formula "
            "selection, or source-authority fitting was performed."
        ),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("\n" + "=" * 78)
    print("STAGE 4 COMPLETE — LOCKED TEST HAS NOW BEEN EVALUATED")
    print("=" * 78)
    print("DO NOT tune Stage 3 parameters after viewing these TEST results.")
    print(f"Saved: {RESULTS_PATH}")
    print(f"Saved: {PREDICTIONS_PATH}")
    print(f"Saved: {ERRORS_PATH}")


if __name__ == "__main__":
    main()
