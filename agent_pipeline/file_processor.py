"""
Handles PDF and CSV file uploads.
Extracts text/rows and passes to the appropriate agent.
"""

import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.intake_agent import extract_and_store
from agents.billing_agent import extract_and_store_billing


def process_csv(file_path: str, agent_id: str = "intake_agent") -> dict:
    print(f"\nProcessing CSV: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    except Exception as e:
        return {"success": False, "error": f"Cannot read CSV: {e}"}

    column_to_fact_type = {
    "incident_state": "state",
    "priority": "priority",
    "urgency": "urgency",
    "impact": "impact",
    "assignment_group": "assignment_group",
    "category": "category",
    "resolved_by": "resolved_by",
    "opened_at": "opened_date",
    "closed_at": "opened_date"
}

    id_col = None
    for col in ["number", "incident_id", "id"]:
        if col in df.columns:
            id_col = col
            break

    if not id_col:
        return {"success": False, "error": f"No incident ID column found. Columns: {list(df.columns)}"}

    results, errors, contradictions_found = [], [], 0
    source_file = os.path.basename(file_path)
    use_billing = "field_report" in source_file.lower()

    for _, row in df.iterrows():
        incident_id = str(row[id_col]).strip()
        if not incident_id or incident_id == "nan":
            continue

        for col, fact_type in column_to_fact_type.items():
            if col not in df.columns:
                continue
            value = str(row.get(col, "")).strip()
            if not value or value == "nan":
                continue

            sentence = f"{incident_id} has {fact_type} {value} according to {source_file}."

            result = (extract_and_store_billing(sentence, source_file) if use_billing
                      else extract_and_store(sentence, agent_id, source_file))

            if result.get("success"):
                results.append({"incident_id": incident_id, "fact_type": fact_type, "value": value})
                if result.get("contradiction_detected"):
                    contradictions_found += 1
            else:
                errors.append({"incident_id": incident_id, "error": result.get("error", "Unknown error")})

    return {
        "success": True, "file": source_file, "rows_processed": len(results),
        "contradictions_found": contradictions_found, "errors": len(errors),
        "results": results[:10]
    }


def process_pdf(file_path: str, agent_id: str = "intake_agent") -> dict:
    try:
        import fitz
    except ImportError:
        return {"success": False, "error": "PyMuPDF not installed. Run: pip install pymupdf"}

    print(f"\nProcessing PDF: {file_path}")
    try:
        doc = fitz.open(file_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        return {"success": False, "error": f"Cannot read PDF: {e}"}

    if not full_text.strip():
        return {"success": False, "error": "PDF appears to be empty"}

    print(f"Extracted {len(full_text)} characters from PDF")

    sentences = [s.strip() for s in full_text.split(".")
                 if len(s.strip()) > 15 and any(w in s.upper() for w in ["INC", "INCIDENT"])]

    print(f"Found {len(sentences)} incident-related sentences")

    results, contradictions_found = [], 0
    source_file = os.path.basename(file_path)

    for sentence in sentences[:20]:
        result = extract_and_store(sentence, agent_id, source_file)
        if result.get("success"):
            results.append(result["fact"])
            if result.get("contradiction_detected"):
                contradictions_found += 1

    return {
        "success": True, "file": source_file, "sentences_processed": len(results),
        "contradictions_found": contradictions_found, "results": results
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_processor.py <path_to_csv_or_pdf>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = process_csv(file_path) if file_path.endswith(".csv") else \
              process_pdf(file_path) if file_path.endswith(".pdf") else None

    if result is None:
        print("Unsupported file type. Use .csv or .pdf")
        sys.exit(1)

    print("\nResult:")
    print(f"  Success: {result['success']}")
    if result['success']:
        print(f"  Processed: {result.get('rows_processed') or result.get('sentences_processed')}")
        print(f"  Contradictions found: {result.get('contradictions_found', 0)}")
        print(f"  Errors: {result.get('errors', 0)}")