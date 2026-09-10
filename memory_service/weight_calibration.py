"""
Weight Calibration — Ablation Study and Threshold Sensitivity Analysis
For: Trusted Shared Memory for Multi-Agent AI Systems
"""

import requests
import json
import os
import sys
import re

BASE = "http://127.0.0.1:8000"

CONFIGURATIONS = {
    "uniform": {
        "intake_agent":      {"default": 0.70},
        "delivery_agent":    {"default": 0.70},
        "billing_agent":     {"default": 0.70},
        "coordinator_agent": {"default": 0.70}
    },
    "global_rank": {
        "intake_agent":      {"default": 0.85},
        "delivery_agent":    {"default": 0.75},
        "billing_agent":     {"default": 0.45},
        "coordinator_agent": {"default": 0.80}
    },
    "itsm_domain": {
        "intake_agent": {
            "priority": 0.88, "assignment_group": 0.83,
            "category": 0.78, "opened_date": 0.95,
            "state": 0.72, "urgency": 0.62,
            "impact": 0.68, "resolved_by": 0.38, "default": 0.70
        },
        "delivery_agent": {
            "state": 0.90, "urgency": 0.84, "impact": 0.80,
            "priority": 0.70, "category": 0.63,
            "assignment_group": 0.55, "opened_date": 0.58,
            "resolved_by": 0.52, "default": 0.66
        },
        "billing_agent": {
            "resolved_by": 0.90, "state": 0.80, "category": 0.60,
            "assignment_group": 0.50, "impact": 0.52,
            "urgency": 0.44, "priority": 0.38,
            "opened_date": 0.32, "default": 0.50
        },
        "coordinator_agent": {"default": 0.80}
    },
    "itsm_domain_conservative": {
        "intake_agent": {
            "priority": 0.80, "assignment_group": 0.75,
            "category": 0.70, "opened_date": 0.88,
            "state": 0.65, "urgency": 0.58,
            "impact": 0.62, "resolved_by": 0.42, "default": 0.65
        },
        "delivery_agent": {
            "state": 0.82, "urgency": 0.78, "impact": 0.74,
            "priority": 0.65, "category": 0.60,
            "assignment_group": 0.52, "opened_date": 0.55,
            "resolved_by": 0.50, "default": 0.62
        },
        "billing_agent": {
            "resolved_by": 0.82, "state": 0.72, "category": 0.58,
            "assignment_group": 0.48, "impact": 0.50,
            "urgency": 0.42, "priority": 0.42,
            "opened_date": 0.38, "default": 0.52
        },
        "coordinator_agent": {"default": 0.78}
    }
}

AGENT_EXTRACTION_TYPE = {
    "intake_agent":      "direct",
    "delivery_agent":    "direct",
    "billing_agent":     "inferred",
    "coordinator_agent": "direct"
}


def compute_confidence_local(agent_id, fact_type,
                              extraction_type, config):
    agent_weights = config.get(agent_id, {})
    domain_authority = agent_weights.get(
        fact_type, agent_weights.get("default", 0.50)
    )
    directness = 0.90 if extraction_type == "direct" else 0.45
    raw = (
        0.50 * domain_authority +   # higher domain weight
        0.20 * 0.20 +               # corroboration
        0.15 * directness -
        0.15 * 0.02                 # decay
    )
    return round(max(0.10, min(0.99, raw)), 4)


def get_domain_weight(agent_id, fact_type, config):
    """Get just the domain weight for an agent+fact_type."""
    agent_weights = config.get(agent_id, {})
    return agent_weights.get(
        fact_type,
        agent_weights.get("default", 0.50)
    )


def get_extraction_type(turn):
    """Get extraction type from turn, falling back to agent-based default."""
    et = turn.get("extraction_type")
    if et and et in ("direct", "inferred"):
        return et
    agent = turn.get("agent", "intake_agent")
    return AGENT_EXTRACTION_TYPE.get(agent, "direct")


def get_turn_value(turn, scenario_fact_type):
    """Extract value and fact_type from turn."""
    fact_type = (
        turn.get("fact_type") or
        turn.get("expected_fact", {}).get("fact_type") or
        scenario_fact_type or ""
    ).lower().strip()

    value = (
        turn.get("value") or
        turn.get("expected_fact", {}).get("value") or ""
    ).strip().lower()

    return fact_type, value


def find_conflict(scenario, config):
    scenario_fact_type = scenario.get("fact_type", "")
    turns = scenario.get("turns", [])

    facts = []
    for turn in turns:
        ft, value = get_turn_value(turn, scenario_fact_type)
        if not ft or not value:
            continue
        agent = turn.get("agent", "intake_agent")
        extraction = get_extraction_type(turn)
        conf = compute_confidence_local(agent, ft, extraction, config)
        domain_w = get_domain_weight(agent, ft, config)
        facts.append({
            "fact_type": ft,
            "value": value,
            "agent": agent,
            "extraction": extraction,
            "confidence": conf,
            "domain_weight": domain_w
        })

    conflicts = []
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            a, b = facts[i], facts[j]
            if (a["fact_type"] == b["fact_type"] and
                    a["value"] != b["value"] and
                    a["agent"] != b["agent"]):

                conf_gap = abs(a["confidence"] - b["confidence"])
                domain_gap = abs(
                    a["domain_weight"] - b["domain_weight"]
                )

                # Winner: higher confidence
                # Tiebreak: higher domain weight
                # Tiebreak 2: higher extraction directness
                def score(f):
                    ext_score = 1.0 if f["extraction"] == "direct" else 0.0
                    return (f["confidence"], f["domain_weight"], ext_score)

                if score(a) >= score(b):
                    winner, loser = a, b
                else:
                    winner, loser = b, a

                conflicts.append({
                    "gap": conf_gap,
                    "domain_gap": domain_gap,
                    "winner": winner,
                    "loser": loser
                })

    if not conflicts:
        return None
    return max(conflicts, key=lambda x: (x["gap"], x["domain_gap"]))


def normalize(s):
    if not s:
        return ""
    s = str(s).lower().strip()
    # Remove all whitespace, dashes, underscores
    s = re.sub(r'[\s\-_]+', '', s)
    # Remove trailing dots
    s = s.rstrip('.')
    return s

def resolution_correct(conflict, ground_truth):
    correct_raw = ground_truth.get("correct_resolution", "")
    if not correct_raw:
        return False
    correct_n = normalize(correct_raw)
    winner_n = normalize(conflict["winner"]["value"])
    return winner_n == correct_n


def load_scenarios():
    possible_dirs = [
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "dataset", "scenarios"
        ),
        os.path.join("..", "dataset", "scenarios"),
    ]
    scenarios_dir = None
    for d in possible_dirs:
        resolved = os.path.abspath(d)
        if os.path.exists(resolved):
            scenarios_dir = resolved
            break

    if not scenarios_dir:
        print("Scenarios directory not found.")
        return []

    print(f"Found scenarios at: {scenarios_dir}\n")
    scenarios = []
    files_found = 0

    for filename in sorted(os.listdir(scenarios_dir)):
        if not filename.endswith(".json"):
            continue
        files_found += 1
        with open(
            os.path.join(scenarios_dir, filename),
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        category = data.get("category", "?")
        category_name = data.get("category_name", "")
        loaded = 0

        for scenario in data.get("scenarios", []):
            scenario["category"] = category
            scenario["category_name"] = category_name
            turns = scenario.get("turns", [])

            # Extract entity
            if "entity" not in scenario:
                for turn in turns:
                    entity = (
                        turn.get("entity") or
                        turn.get("expected_fact", {}).get("entity")
                    )
                    if not entity:
                        m = re.search(r'INC\d+',
                                      turn.get("input", ""))
                        if m:
                            entity = m.group(0)
                    if entity:
                        scenario["entity"] = entity
                        break

            # Extract fact_type
            if "fact_type" not in scenario:
                for turn in turns:
                    ft = (
                        turn.get("fact_type") or
                        turn.get("expected_fact", {}).get("fact_type")
                    )
                    if ft:
                        scenario["fact_type"] = ft.lower().strip()
                        break

            missing = [
                f for f in
                ["entity", "fact_type", "turns", "ground_truth"]
                if f not in scenario
            ]
            if missing:
                continue

            scenarios.append(scenario)
            loaded += 1

        print(f"  {filename}: loaded {loaded}/"
              f"{len(data.get('scenarios', []))}")

    print(f"\nTotal: {len(scenarios)} valid scenarios "
          f"from {files_found} files\n")
    return scenarios


def evaluate_config(config_name, config, scenarios, verbose=False):
    tp = fp = fn = tn = 0
    correct_res = 0
    total_res = 0

    for scenario in scenarios:
        gt = scenario.get("ground_truth", {})
        expected = gt.get("contradiction_expected", False)
        correct_val_raw = gt.get("correct_resolution", "")
        correct_val = normalize(correct_val_raw)

        conflict = find_conflict(scenario, config)
        found = conflict is not None

        if expected and found:
            tp += 1
            # Count this scenario for resolution accuracy
            # if it has ANY correct_resolution value
            if correct_val:
                total_res += 1
                winner_val = normalize(conflict["winner"]["value"])
                is_correct = resolution_correct(conflict, gt)

                if verbose:
                    print(f"    {scenario.get('scenario_id')}: "
                          f"winner='{winner_val}' "
                          f"correct='{correct_val}' "
                          f"match={is_correct} "
                          f"gap={conflict['gap']:.4f}")

                if is_correct:
                    correct_res += 1

        elif not expected and found:
            fp += 1
        elif expected and not found:
            fn += 1
        else:
            tn += 1

    total = len(scenarios)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    res_acc = correct_res / total_res if total_res > 0 else 0

    return {
        "config": config_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round((tp + tn) / total, 4),
        "f1": round(f1, 4),
        "resolution_accuracy": round(res_acc, 4),
        "correct_res": correct_res,
        "total_res": total_res,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def threshold_sensitivity(scenarios):
    """
    Test AUTO_RESOLVE_THRESHOLD values using itsm_domain config.
    Uses SAME logic as evaluate_config for consistency.
    """
    config = CONFIGURATIONS["itsm_domain"]
    # Finer granularity in the range where your gaps actually fall
    thresholds = [0.03, 0.05, 0.07, 0.09, 0.10,
              0.11, 0.13, 0.15, 0.17, 0.19, 0.20]

    # Pre-compute all conflicts
    all_conflicts = []
    for scenario in scenarios:
        gt = scenario.get("ground_truth", {})
        if not gt.get("contradiction_expected"):
            continue
        c = find_conflict(scenario, config)
        if c:
            correct_val = gt.get("correct_resolution", "")
            all_conflicts.append({
                "gap": c["gap"],
                "is_correct": resolution_correct(c, gt),
                "correct_val": normalize(correct_val),
                "winner_val": normalize(c["winner"]["value"]),
                "scenario_id": scenario.get("scenario_id")
            })

    print("\n" + "=" * 72)
    print("THRESHOLD SENSITIVITY ANALYSIS (itsm_domain config)")
    print(f"Total contradiction scenarios: {len(all_conflicts)}")
    if all_conflicts:
        gaps = [c["gap"] for c in all_conflicts]
        correct_count = sum(
            1 for c in all_conflicts if c["is_correct"]
        )
        print(f"Gap range: {min(gaps):.4f} to {max(gaps):.4f} "
              f"| Mean: {sum(gaps)/len(gaps):.4f}")
        print(f"Correct resolutions regardless of threshold: "
              f"{correct_count}/{len(all_conflicts)} = "
              f"{correct_count/len(all_conflicts):.1%}")
    print("=" * 72)
    print(f"{'Threshold':>11} {'AutoRes':>9} {'Contested':>11} "
          f"{'ResAcc':>8} {'Notes':>20}")
    print("-" * 72)

    results = []
    best = None

    for threshold in thresholds:
        # Count how many conflicts exceed threshold
        auto = [c for c in all_conflicts if c["gap"] >= threshold]
        contested = [
            c for c in all_conflicts if c["gap"] < threshold
        ]

        auto_pct = len(auto) / len(all_conflicts) if all_conflicts else 0
        contested_pct = 1.0 - auto_pct

        # Resolution accuracy: only auto-resolved conflicts get resolved
        # Contested conflicts are flagged for human review
        # ResAcc = fraction of auto-resolved that are correct
        correct_auto = sum(1 for c in auto if c["is_correct"])
        res_acc = correct_auto / len(auto) if len(auto) > 0 else 0

        # F1 stays same (detection unchanged by threshold)
        # We track it for completeness
        total_contradiction_scenarios = len(all_conflicts)
        auto_count = len(auto)
        contested_count = len(contested)

        note = ""
        if auto_pct == 0:
            note = "all contested"
        elif auto_pct == 1:
            note = "all auto"
        elif 0.4 <= auto_pct <= 0.7:
            note = "good balance"

        r = {
            "threshold": threshold,
            "auto_count": len(auto),
            "contested_count": len(contested),
            "auto_resolve_pct": round(auto_pct, 4),
            "contested_pct": round(contested_pct, 4),
            "resolution_accuracy": round(res_acc, 4),
            "correct_auto": correct_auto
        }
        results.append(r)

        print(f"  {threshold:>9.2f} {auto_pct:>8.1%} "
              f"{contested_pct:>10.1%} {res_acc:>8.3f}  {note:>20}")

        if best is None:
            best = r
        elif res_acc > best["resolution_accuracy"]:
            best = r
        elif (res_acc == best["resolution_accuracy"] and
              abs(auto_pct - 0.50) < abs(
                  best["auto_resolve_pct"] - 0.50)):
            best = r

    print("-" * 72)

    # Gap distribution bar chart
    print("\nGap distribution across contradiction scenarios:")
    for t in [0.03, 0.05, 0.07, 0.10, 0.13, 0.15, 0.20]:
        above = sum(1 for c in all_conflicts if c["gap"] >= t)
        total = len(all_conflicts)
        bar = "=" * above + "-" * (total - above)
        pct = above / total if total > 0 else 0
        print(f"  >= {t:.2f}: {above:>2}/{total} "
              f"[{bar}] {pct:.0%}")

    # Show each conflict's gap and correctness
    print("\nPer-scenario conflict details:")
    print(f"  {'Scenario':<12} {'Gap':>8} {'Winner':>20} "
          f"{'Correct':>8} {'Auto@0.07':>10}")
    for c in sorted(all_conflicts, key=lambda x: x["gap"],
                    reverse=True):
        auto_at_7 = "YES" if c["gap"] >= 0.07 else "no"
        print(f"  {c['scenario_id']:<12} {c['gap']:>8.4f} "
              f"{c['winner_val']:>20} {str(c['is_correct']):>8} "
              f"{auto_at_7:>10}")

    if best:
        print(f"\nOptimal threshold: {best['threshold']:.2f} "
              f"(ResAcc={best['resolution_accuracy']:.3f}, "
              f"Auto={best['auto_resolve_pct']:.1%})")

    with open("threshold_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to threshold_sensitivity.json")

    return best["threshold"] if best else 0.07

FORMULA_CONFIGURATIONS = {
    "equal_weights": {
        "domain": 0.25, "corroboration": 0.25,
        "directness": 0.25, "decay": 0.25
    },
    "original_35": {
        "domain": 0.35, "corroboration": 0.30,
        "directness": 0.15, "decay": 0.20
    },
    "domain_heavy_40": {
        "domain": 0.40, "corroboration": 0.25,
        "directness": 0.20, "decay": 0.15
    },
    "domain_heavy_50": {
        "domain": 0.50, "corroboration": 0.20,
        "directness": 0.15, "decay": 0.15
    },
    "domain_heavy_60": {
        "domain": 0.60, "corroboration": 0.15,
        "directness": 0.15, "decay": 0.10
    }
}

def compute_confidence_with_formula(agent_id, fact_type,
                                     extraction_type,
                                     formula_config,
                                     trust_config):
    agent_weights = trust_config.get(agent_id, {})
    domain_authority = agent_weights.get(
        fact_type, agent_weights.get("default", 0.50)
    )
    directness = 0.90 if extraction_type == "direct" else 0.45
    raw = (
        formula_config["domain"] * domain_authority +
        formula_config["corroboration"] * 0.20 +
        formula_config["directness"] * directness -
        formula_config["decay"] * 0.02
    )
    return round(max(0.10, min(0.99, raw)), 4)

def run_formula_ablation(scenarios):
    """
    Test different formula weight distributions.
    Uses itsm_domain SOURCE_CONDITIONAL_TRUST throughout.
    Measures: max gap achieved, ResAcc at threshold 0.20.
    """
    trust_config = CONFIGURATIONS["itsm_domain"]
    threshold = 0.20

    print("\n" + "=" * 70)
    print("FORMULA WEIGHT ABLATION")
    print("(itsm_domain trust config, threshold=0.20)")
    print("=" * 70)
    print(f"{'Formula':<22} {'MaxGap':>8} {'MeanGap':>9} "
          f"{'AutoRes':>9} {'ResAcc':>8}")
    print("-" * 70)

    results = []
    for formula_name, formula in FORMULA_CONFIGURATIONS.items():
        gaps = []
        correct_auto = 0
        total_auto = 0

        for scenario in scenarios:
            gt = scenario.get("ground_truth", {})
            if not gt.get("contradiction_expected"):
                continue

            fact_type = scenario.get("fact_type", "")
            turns = scenario.get("turns", [])
            facts = []

            for turn in turns:
                ft, value = get_turn_value(turn, fact_type)
                if not ft or not value:
                    continue
                agent = turn.get("agent", "intake_agent")
                extraction = get_extraction_type(turn)
                conf = compute_confidence_with_formula(
                    agent, ft, extraction,
                    formula, trust_config
                )
                facts.append({
                    "fact_type": ft, "value": value,
                    "agent": agent, "confidence": conf
                })

            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    a, b = facts[i], facts[j]
                    if (a["fact_type"] == b["fact_type"] and
                            a["value"] != b["value"] and
                            a["agent"] != b["agent"]):
                        gap = abs(a["confidence"] - b["confidence"])
                        gaps.append(gap)

                        if gap >= threshold:
                            total_auto += 1
                            winner = a if a["confidence"] >= b["confidence"] else b
                            correct_val = normalize(
                                gt.get("correct_resolution", "")
                            )
                            if normalize(winner["value"]) == correct_val:
                                correct_auto += 1
                        break

        max_gap = max(gaps) if gaps else 0
        mean_gap = sum(gaps) / len(gaps) if gaps else 0
        auto_pct = sum(1 for g in gaps if g >= threshold) / len(gaps) if gaps else 0
        res_acc = correct_auto / total_auto if total_auto > 0 else 0

        results.append({
            "formula": formula_name,
            "max_gap": round(max_gap, 4),
            "mean_gap": round(mean_gap, 4),
            "auto_pct": round(auto_pct, 4),
            "res_acc": round(res_acc, 4),
            "weights": formula
        })

        print(f"  {formula_name:<22} {max_gap:>8.4f} {mean_gap:>9.4f} "
              f"{auto_pct:>8.1%} {res_acc:>8.3f}")

    best = max(results, key=lambda x: (x["res_acc"], x["auto_pct"]))
    print("-" * 70)
    print(f"\nBest formula: {best['formula']}")
    print(f"  domain={best['weights']['domain']} "
          f"corroboration={best['weights']['corroboration']} "
          f"directness={best['weights']['directness']} "
          f"decay={best['weights']['decay']}")
    print(f"  MaxGap={best['max_gap']:.4f} "
          f"ResAcc={best['res_acc']:.3f} "
          f"AutoRes={best['auto_pct']:.1%}")

    with open("formula_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to formula_ablation_results.json")

    return best

def main():
    try:
        r = requests.get(f"{BASE}/memory/health", timeout=5)
        print(f"Server: {r.json().get('status', 'ok')}\n")
    except Exception:
        print(f"Cannot connect to {BASE}")
        sys.exit(1)

    scenarios = load_scenarios()
    if not scenarios:
        print("No scenarios loaded.")
        sys.exit(1)

    # Debug: show first contradiction scenario with confidence breakdown
    print("=== DEBUG: First contradiction scenario ===")
    for s in scenarios:
        gt = s.get("ground_truth", {})
        if not gt.get("contradiction_expected"):
            continue
        print(f"Scenario: {s.get('scenario_id')} | "
              f"entity={s.get('entity')} | "
              f"fact_type={s.get('fact_type')}")
        print(f"correct_resolution: "
              f"'{gt.get('correct_resolution')}'")
        turns = s.get("turns", [])
        for i, t in enumerate(turns):
            ft, val = get_turn_value(t, s.get("fact_type", ""))
            agent = t.get("agent", "?")
            ext = get_extraction_type(t)
            print(f"\n  Turn {i+1}: agent={agent} | "
                  f"fact_type={ft} | value={val} | "
                  f"extraction={ext}")
            for cfg_name, cfg in CONFIGURATIONS.items():
                conf = compute_confidence_local(
                    agent, ft, ext, cfg
                )
                dw = get_domain_weight(agent, ft, cfg)
                print(f"    [{cfg_name:<28}] "
                      f"domain_w={dw:.2f} conf={conf:.4f}")

        # Show conflict results per config
        print(f"\n  Conflict detection per config:")
        for cfg_name, cfg in CONFIGURATIONS.items():
            c = find_conflict(s, cfg)
            if c:
                correct = resolution_correct(c, gt)
                print(f"    [{cfg_name:<28}] "
                      f"gap={c['gap']:.4f} "
                      f"winner={c['winner']['value']} "
                      f"correct={correct}")
            else:
                print(f"    [{cfg_name:<28}] no conflict found")
        break
    print("=" * 45 + "\n")

    # ── Ablation study ─────────────────────────────────────────
    print("=" * 72)
    print("ABLATION STUDY — Weight Configuration Comparison")
    print("=" * 72)
    print(f"{'Configuration':<30} {'Prec':>6} {'Recall':>8} "
          f"{'F1':>8} {'ResAcc':>8} "
          f"{'Correct':>8} {'Total':>7}")
    print("-" * 72)

    results = []
    for cfg_name, cfg in CONFIGURATIONS.items():
        m = evaluate_config(cfg_name, cfg, scenarios)
        results.append(m)
        print(f"  {cfg_name:<28} {m['precision']:>6.3f} "
              f"{m['recall']:>8.3f} {m['f1']:>8.3f} "
              f"{m['resolution_accuracy']:>8.3f} "
              f"{m['correct_res']:>8} {m['total_res']:>7}")

    print("=" * 72)
    best = max(
        results,
        key=lambda x: (x["f1"], x["resolution_accuracy"])
    )
    print(f"\nBest: {best['config']} "
          f"(F1={best['f1']:.3f}, "
          f"ResAcc={best['resolution_accuracy']:.3f})")

    with open("calibration_results.json", "w") as f:
        json.dump({
            "ablation_results": results,
            "selected_config": best["config"],
            "selection_criterion":
                "highest F1, then resolution_accuracy",
            "dataset_size": len(scenarios)
        }, f, indent=2)
    print("Saved to calibration_results.json\n")

    # ── Threshold sensitivity ──────────────────────────────────
    optimal = threshold_sensitivity(scenarios)

    print(f"\n{'='*72}")
    print("FINAL RECOMMENDATIONS")
    print(f"  Best weight config:   {best['config']}")
    print(f"  Optimal threshold:    {optimal:.2f}")
    print(f"\nUpdate confidence_scorer.py:")
    print(f"  AUTO_RESOLVE_THRESHOLD = {optimal:.2f}")
    if best["config"] == "itsm_domain":
        print("  SOURCE_CONDITIONAL_TRUST: no change (already itsm_domain)")
    else:
        print(f"  Consider updating SOURCE_CONDITIONAL_TRUST "
              f"to {best['config']} weights")
    print(f"{'='*72}")


    # First pass verbose to see resolution matching
    print("=== VERBOSE RESOLUTION CHECK (uniform config) ===")
    evaluate_config("uniform", CONFIGURATIONS["uniform"],
                scenarios, verbose=True)
    print("=== END VERBOSE ===\n")

    print("\n--- Running formula weight ablation ---")
    best_formula = run_formula_ablation(scenarios)
if __name__ == "__main__":
    main()