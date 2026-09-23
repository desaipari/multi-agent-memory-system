"""
Bootstrap confidence intervals for evaluation metrics.
Required for statistical validity on small datasets.

Run after weight_calibration.py has produced calibration_results.json
"""

import json
import os
import sys
import re
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def normalize(s):
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[\s\-_]+', '', s)
    return s.rstrip('.')


def load_scenarios():
    possible_dirs = [
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "dataset", "scenarios"
        ),
    ]
    scenarios_dir = None
    for d in possible_dirs:
        resolved = os.path.abspath(d)
        if os.path.exists(resolved):
            scenarios_dir = resolved
            break
    if not scenarios_dir:
        return []

    scenarios = []
    for filename in sorted(os.listdir(scenarios_dir)):
        if not filename.endswith(".json"):
            continue
        with open(
            os.path.join(scenarios_dir, filename),
            encoding="utf-8"
        ) as f:
            data = json.load(f)
        for s in data.get("scenarios", []):
            turns = s.get("turns", [])
            if "entity" not in s:
                for t in turns:
                    entity = t.get("entity")
                    if not entity:
                        m = re.search(
                            r'INC\d+', t.get("input", "")
                        )
                        if m:
                            entity = m.group(0)
                    if entity:
                        s["entity"] = entity
                        break
            if "fact_type" not in s:
                for t in turns:
                    ft = t.get("fact_type")
                    if ft:
                        s["fact_type"] = ft.lower()
                        break
            if all(f in s for f in ["entity","fact_type",
                                      "turns","ground_truth"]):
                scenarios.append(s)
    return scenarios


def compute_metrics_on_sample(sample, config, threshold=0.20):
    """Compute detection and resolution metrics on a sample."""
    from weight_calibration import (
        find_conflict, resolution_correct,
        CONFIGURATIONS
    )

    tp = fp = fn = tn = 0
    correct_res = 0
    total_res = 0

    for scenario in sample:
        gt = scenario.get("ground_truth", {})
        expected = gt.get("contradiction_expected", False)

        conflict = find_conflict(scenario, config)
        found = conflict is not None

        if expected and found:
            tp += 1
            correct_val = gt.get("correct_resolution", "")
            if correct_val:
                total_res += 1
                if resolution_correct(conflict, gt):
                    correct_res += 1
        elif not expected and found:
            fp += 1
        elif expected and not found:
            fn += 1
        else:
            tn += 1

    total = len(sample)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    res_acc = correct_res / total_res if total_res > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "resolution_accuracy": res_acc
    }


def bootstrap_ci(scenarios, config, n_bootstrap=1000,
                 ci_level=0.95, threshold=0.20, random_seed=42):
    """
    Bootstrap confidence intervals via sampling with replacement.
    Standard method for CI estimation on small datasets.
    Reference: Efron & Tibshirani (1993) — An Introduction to
    the Bootstrap, Chapman & Hall.
    """
    random.seed(random_seed)
    n = len(scenarios)

    metrics_samples = {
        "f1": [], "precision": [], "recall": [],
        "accuracy": [], "resolution_accuracy": []
    }

    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = [
            scenarios[random.randint(0, n-1)]
            for _ in range(n)
        ]
        m = compute_metrics_on_sample(sample, config, threshold)
        for key in metrics_samples:
            metrics_samples[key].append(m[key])

    alpha = (1 - ci_level) / 2
    ci_results = {}
    for metric, values in metrics_samples.items():
        values_sorted = sorted(values)
        lower_idx = int(alpha * n_bootstrap)
        upper_idx = int((1 - alpha) * n_bootstrap)
        mean_val = sum(values) / len(values)
        ci_results[metric] = {
            "mean": round(mean_val, 4),
            "lower": round(values_sorted[lower_idx], 4),
            "upper": round(values_sorted[upper_idx], 4),
            "ci_level": ci_level
        }

    return ci_results


def run_error_analysis(scenarios, config, threshold=0.20):
    """
    Analyse false positives, false negatives, and wrong resolutions.
    Required by reviewers — need to explain what the system gets wrong.
    """
    from weight_calibration import (
        find_conflict, resolution_correct
    )

    false_positives = []
    false_negatives = []
    wrong_resolutions = []

    for scenario in scenarios:
        gt = scenario.get("ground_truth", {})
        expected = gt.get("contradiction_expected", False)
        sid = scenario.get("scenario_id", "?")
        entity = scenario.get("entity", "?")
        fact_type = scenario.get("fact_type", "?")

        conflict = find_conflict(scenario, config)
        found = conflict is not None

        if not expected and found:
            # False positive — detected contradiction where none expected
            false_positives.append({
                "scenario_id": sid,
                "entity": entity,
                "fact_type": fact_type,
                "category": scenario.get("category_name", ""),
                "detected_gap": round(conflict["gap"], 4),
                "value_a": conflict["winner"]["value"],
                "value_b": conflict["loser"]["value"],
                "reason": gt.get("reason", ""),
                "analysis": (
                    "System detected value difference as contradiction "
                    "but ground truth labels this as corroboration or "
                    "legitimate update. Likely a close-value case where "
                    "string normalization treats different formats of the "
                    "same value as distinct."
                )
            })

        elif expected and found:
            correct_val = gt.get("correct_resolution", "")
            if correct_val and not resolution_correct(conflict, gt):
                # Wrong resolution — detected but picked wrong winner
                wrong_resolutions.append({
                    "scenario_id": sid,
                    "entity": entity,
                    "fact_type": fact_type,
                    "category": scenario.get("category_name", ""),
                    "correct_value": correct_val,
                    "predicted_winner": conflict["winner"]["value"],
                    "predicted_agent": conflict["winner"]["agent"],
                    "correct_agent": gt.get("winning_agent", "?"),
                    "confidence_gap": round(conflict["gap"], 4),
                    "analysis": (
                        f"System selected {conflict['winner']['agent']} "
                        f"but ground truth says "
                        f"{gt.get('winning_agent','?')} should win. "
                        f"Gap={conflict['gap']:.4f}. May indicate that "
                        f"domain weight for this agent/fact_type pair "
                        f"needs adjustment."
                    )
                })

        elif expected and not found:
            # False negative — missed a real contradiction
            turns = scenario.get("turns", [])
            values = [
                t.get("value", "") for t in turns if t.get("value")
            ]
            false_negatives.append({
                "scenario_id": sid,
                "entity": entity,
                "fact_type": fact_type,
                "category": scenario.get("category_name", ""),
                "turn_values": values,
                "analysis": (
                    "System did not detect contradiction. "
                    "Possible causes: turns have same agent, "
                    "values normalized to same string, or "
                    "fact_type extracted incorrectly."
                )
            })

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "wrong_resolutions": wrong_resolutions,
        "summary": {
            "n_fp": len(false_positives),
            "n_fn": len(false_negatives),
            "n_wrong_resolution": len(wrong_resolutions)
        }
    }


def main():
    from weight_calibration import CONFIGURATIONS, split_scenarios

    scenarios = load_scenarios()
    if not scenarios:
        return

    # Use SAME split as weight_calibration.py
    # CRITICAL: CI must be computed on held-out test set only
    # Computing on full dataset inflates results
    dev_scenarios, test_scenarios = split_scenarios(
        scenarios, test_fraction=0.30, seed=42
    )

    print(f"Computing CIs on HELD-OUT TEST SET only")
    print(f"Test set: {len(test_scenarios)} scenarios")
    print(f"Dev set:  {len(dev_scenarios)} scenarios (NOT used here)")
    print(f"This matches the split used in weight_calibration.py\n")

    config = CONFIGURATIONS["itsm_domain_conservative"]

    print("Running bootstrap CI (n=1000 resamples)...")
    # Bootstrap on TEST SET ONLY
    ci = bootstrap_ci(
        test_scenarios, config, n_bootstrap=1000
    )
    # ... rest of function unchanged

    print("\n" + "=" * 65)
    print("CONFIDENCE INTERVALS (95%, bootstrap, n=1000)")
    print("=" * 65)
    print(f"{'Metric':<25} {'Mean':>8} {'Lower':>8} {'Upper':>8}")
    print("-" * 65)
    for metric, vals in ci.items():
        print(
            f"  {metric:<23} {vals['mean']:>8.4f} "
            f"{vals['lower']:>8.4f} {vals['upper']:>8.4f}"
        )
    print("=" * 65)

    print("\nRunning error analysis...")
    errors = run_error_analysis(scenarios, config)

    print(f"\nError Analysis Summary:")
    print(f"  False Positives:     {errors['summary']['n_fp']}")
    print(f"  False Negatives:     {errors['summary']['n_fn']}")
    print(f"  Wrong Resolutions:   "
          f"{errors['summary']['n_wrong_resolution']}")

    if errors["false_positives"]:
        print(f"\nFalse Positive Cases:")
        for fp in errors["false_positives"]:
            print(f"  {fp['scenario_id']}: "
                  f"{fp['fact_type']} | "
                  f"gap={fp['detected_gap']:.4f} | "
                  f"'{fp['value_a']}' vs '{fp['value_b']}'")
            print(f"    → {fp['analysis'][:100]}")

    if errors["wrong_resolutions"]:
        print(f"\nWrong Resolution Cases:")
        for wr in errors["wrong_resolutions"]:
            print(f"  {wr['scenario_id']}: "
                  f"predicted='{wr['predicted_winner']}' "
                  f"correct='{wr['correct_value']}'")
            print(f"    → {wr['analysis'][:100]}")

    # Save all results
    output = {
        "confidence_intervals": ci,
        "n_scenarios": len(scenarios),
        "n_bootstrap": 1000,
        "ci_level": 0.95,
        "config_used": "itsm_domain_conservative",
        "threshold": 0.20,
        "error_analysis": errors,
        "note": (
            "Bootstrap CI computed via sampling with replacement "
            "following Efron & Tibshirani (1993). "
            "Recommended minimum for small-dataset evaluation "
            "in ML/NLP conference papers."
        )
    }
    with open("confidence_intervals.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to confidence_intervals.json")

    # Print paper-ready numbers
    f1_ci = ci["f1"]
    resacc_ci = ci["resolution_accuracy"]
    prec_ci = ci["precision"]
    rec_ci = ci["recall"]

    print(f"\n{'='*65}")
    print("PAPER-READY NUMBERS")
    print(f"{'='*65}")
    print(f"  Detection F1:  {f1_ci['mean']:.3f} "
          f"[{f1_ci['lower']:.3f}, {f1_ci['upper']:.3f}]")
    print(f"  Precision:     {prec_ci['mean']:.3f} "
          f"[{prec_ci['lower']:.3f}, {prec_ci['upper']:.3f}]")
    print(f"  Recall:        {rec_ci['mean']:.3f} "
          f"[{rec_ci['lower']:.3f}, {rec_ci['upper']:.3f}]")
    print(f"  Resolution Acc:{resacc_ci['mean']:.3f} "
          f"[{resacc_ci['lower']:.3f}, {resacc_ci['upper']:.3f}]")
    print(f"{'='*65}")
    print("Use these in your paper as: metric [CI lower, CI upper]")


if __name__ == "__main__":
    main()