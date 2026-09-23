from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_client import write_memory

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

BILLING_PROMPT = """You are a Billing/Ops agent reading from
transferred or older IT incident records.

Your job is to extract exactly ONE structured fact from the text.
You are reading from field reports and transferred logs which may
contain outdated information.

Return ONLY a valid JSON with these fields:
- entity: incident ID (e.g. "INC0000001")
- fact_type: one of: priority, state, assignment_group,
  category, opened_date, resolved_by, urgency, impact
- value: the specific value
- extraction_type: "direct" or "inferred"
- confidence: 0.5 to 0.95

Return ONLY the JSON object. No explanation. No backticks.
"""

def extract_and_store_billing(text: str, source_file: str = "field_reports.csv") -> dict:
    messages = [
        SystemMessage(content=BILLING_PROMPT),
        HumanMessage(content=f"Extract the fact from this field report: {text}")
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = re.sub(r"```json|```", "", raw).strip()

        fact = json.loads(raw)

        required = ["entity", "fact_type", "value", "extraction_type", "confidence"]
        for field in required:
            if field not in fact:
                return {"success": False, "error": f"Missing: {field}"}

        # Billing agent's writes are always tagged inferred —
        # field reports are treated as secondary, lower-trust sources
        fact["extraction_type"] = "inferred"

        memory_response = write_memory(
            entity=fact["entity"],
            fact_type=fact["fact_type"],
            value=fact["value"],
            agent_id="billing_agent",
            extraction_type="inferred",
            confidence=fact["confidence"],
            source_file=source_file
        )

        contradiction = (
            memory_response.get("contradiction_detected", False)
            if memory_response else False
        )

        return {
            "success": memory_response is not None,
            "fact": fact,
            "memory_response": memory_response,
            "contradiction_detected": contradiction
        }

    except json.JSONDecodeError:
        return {"success": False, "error": "JSON parse failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("Testing Billing/Ops Agent")
    print("=" * 60)

    # Deliberately conflicts with what Intake already wrote for INC0000001
    test = "Old field log shows INC0000001 priority was 3-Medium."
    print(f"Input: {test}")
    result = extract_and_store_billing(test)

    if result["success"]:
        print(f"Extracted: {json.dumps(result['fact'], indent=2)}")
        if result["contradiction_detected"]:
            print("*** CONTRADICTION DETECTED — check /memory/conflicts ***")
        else:
            print("Stored — no contradiction flagged by memory service")
    else:
        print(f"Failed: {result['error']}")