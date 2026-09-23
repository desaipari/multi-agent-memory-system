"""
Fast CSV upload — bypasses LLM extraction.
Reads CSV columns directly as structured facts and writes
to memory service without calling Groq API.

Place at: agent_pipeline/fast_csv_upload.py

Usage:
    python fast_csv_upload.py ../demo_data/ticket_intake_clean.csv intake_agent
    python fast_csv_upload.py ../demo_data/monitoring_logs_clean.csv delivery_agent
    python fast_csv_upload.py ../demo_data/field_reports_clean.csv billing_agent
"""

import pandas as pd
import requests
import sys
import os
import time

BASE = "http://127.0.0.1:8000"

COLUMN_TO_FACT_TYPE = {
    "priority":         "priority",
    "incident_state":   "state",
    "assignment_group": "assignment_group",
    "category":         "category",
    "urgency":          "urgency",
    "impact":           "impact",
    "opened_at":        "opened_date",
    "resolved_by":      "resolved_by",
    "assigned_to":      "resolved_by",
}

AGENT_EXTRACTION_TYPE = {
    "intake_agent":      "direct",
    "delivery_agent":    "direct",
    "billing_agent":     "inferred",
    "coordinator_agent": "direct"
}


def upload_csv(file_path: str, agent_id: str):
    print(f"\nFast CSV Upload")
    print(f"File:  {file_path}")
    print(f"Agent: {agent_id}")
    print("=" * 50)

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    id_col = None
    for col in ["number", "incident_id", "id"]:
        if col in df.columns:
            id_col = col
            break

    if not id_col:
        print(f"ERROR: No incident ID column found")
        print(f"Available columns: {list(df.columns)}")
        return

    extraction_type = AGENT_EXTRACTION_TYPE.get(agent_id, "direct")
    source_file     = os.path.basename(file_path)

    written        = 0
    contradictions = 0
    errors         = 0
    skipped        = 0

    start_total = time.time()

    for _, row in df.iterrows():
        incident_id = str(row[id_col]).strip()
        if not incident_id or incident_id == "nan":
            skipped += 1
            continue

        for col, fact_type in COLUMN_TO_FACT_TYPE.items():
            if col not in df.columns:
                continue
            value = str(row.get(col, "")).strip()
            if not value or value in ["nan", "NaN", "", "None"]:
                continue

            try:
                response = requests.post(
                    f"{BASE}/memory/write",
                    json={
                        "entity":         incident_id,
                        "fact_type":      fact_type,
                        "value":          value,
                        "agent_id":       agent_id,
                        "extraction_type": extraction_type,
                        "source_file":    source_file
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    written += 1
                    if data.get("contradiction_detected"):
                        contradictions += 1
                        resolution = data.get(
                            "contradiction", {}
                        ).get("resolution", "")
                        print(f"  CONFLICT: {incident_id}/"
                              f"{fact_type} → {resolution}")
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"  Error: {incident_id}/{fact_type}: {e}")

    elapsed = time.time() - start_total
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Facts written:        {written}")
    print(f"  Contradictions found: {contradictions}")
    print(f"  Errors:               {errors}")
    print(f"  Skipped rows:         {skipped}")
    if elapsed > 0:
        print(f"  Throughput:           "
              f"{written/elapsed:.0f} facts/second")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fast_csv_upload.py <csv_file> <agent_id>")
        print("Example:")
        print("  python fast_csv_upload.py "
              "../demo_data/ticket_intake_clean.csv intake_agent")
        sys.exit(1)
    upload_csv(sys.argv[1], sys.argv[2])