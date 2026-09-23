# Save as: dataset/fix_extraction_types.py
# Run from project root: python dataset/fix_extraction_types.py

import json
import os

SCENARIOS_DIR = os.path.dirname(os.path.abspath(__file__))

# Define which agents use direct vs inferred extraction
# Based on ITSM source-of-record principles:
# - intake_agent reads from primary ticket system → direct
# - delivery_agent reads from monitoring logs → direct
# - billing_agent reads from transferred/old records → inferred
# - coordinator_agent resolves conflicts → direct
AGENT_EXTRACTION_TYPE = {
    "intake_agent":      "direct",
    "delivery_agent":    "direct",
    "billing_agent":     "inferred",   # KEY: stale/transferred source
    "coordinator_agent": "direct"
}

updated = 0
for filename in sorted(os.listdir(SCENARIOS_DIR)):
    if not filename.endswith(".json"):
        continue
    filepath = os.path.join(SCENARIOS_DIR, filename)
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for scenario in data.get("scenarios", []):
        for turn in scenario.get("turns", []):
            agent = turn.get("agent", "intake_agent")
            correct_type = AGENT_EXTRACTION_TYPE.get(agent, "direct")
            if turn.get("extraction_type") != correct_type:
                turn["extraction_type"] = correct_type
                changed = True

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        updated += 1
        print(f"Updated: {filename}")

print(f"\nDone. Updated {updated} files.")
print("Now rerun: python weight_calibration.py")