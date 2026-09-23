from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json
import re
import requests

MEMORY_SERVICE_URL = "http://10.125.5.158:8000"

load_dotenv()

# Using openai/gpt-oss-20b — open source model on Groq
# Free tier, replaces deprecated llama-3.1-8b-instant
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

EXTRACTION_PROMPT = """You are an IT incident fact extractor for an
IT service management system.

Your job is to read a sentence about an IT incident and extract
exactly ONE structured fact from it.

You must return ONLY a valid JSON object with these exact fields:
- entity: the incident ID (e.g. "INC0000001", "INC0000045")
- fact_type: the category of fact. Use ONLY these values:
  priority, state, assignment_group, category, opened_date,
  resolved_by, urgency, impact
- value: the specific value (e.g. "2-High", "Resolved",
  "Network Team", "Software")
- extraction_type: "direct" if explicitly stated,
  "inferred" if you interpreted it
- confidence: number between 0.5 and 0.95 — how clearly
  this fact was stated. Direct statements from official
  records get higher scores. Inferred or second-hand
  information gets lower scores.

Return ONLY the JSON object.
No explanation. No extra text. No markdown backticks.

Examples:
Input: "INC0000001 has priority 2-High according to the ticket system."
Output: {"entity": "INC0000001", "fact_type": "priority", "value": "2-High", "extraction_type": "direct", "confidence": 0.88}

Input: "Monitoring logs show INC0000001 was reassigned to Desktop Support."
Output: {"entity": "INC0000001", "fact_type": "assignment_group", "value": "Desktop Support", "extraction_type": "direct", "confidence": 0.75}

Input: "Field report indicates INC0000001 might have been resolved."
Output: {"entity": "INC0000001", "fact_type": "state", "value": "Resolved", "extraction_type": "inferred", "confidence": 0.62}
"""

def extract_fact(text: str) -> dict:
    """
    Takes a plain text sentence about an IT incident.
    Returns structured fact dictionary or None if extraction fails.
    """
    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=f"Extract the fact from this text: {text}")
    ]

    try:
        response = llm.invoke(messages)
        raw_output = response.content.strip()

        # Remove markdown formatting if model adds it
        if raw_output.startswith("```"):
            raw_output = re.sub(r"```json|```", "", raw_output).strip()

        fact = json.loads(raw_output)

        # Validate all required fields present
        required = ["entity", "fact_type", "value",
                    "extraction_type", "confidence"]
        for field in required:
            if field not in fact:
                print(f"Missing field: {field}")
                return None

        # Validate fact_type is one of the allowed values
        allowed_fact_types = [
            "priority", "state", "assignment_group",
            "category", "opened_date", "resolved_by",
            "urgency", "impact"
        ]
        if fact["fact_type"] not in allowed_fact_types:
            print(f"Invalid fact_type: {fact['fact_type']} — using state as fallback")
            fact["fact_type"] = "state"

        return fact

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw output was: {response.content}")
        return None
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

def write_to_memory(fact: dict, agent_id: str = "intake_agent", source_file: str = None) -> dict:
    """
    Sends an extracted fact to the memory service's /memory/write endpoint.
    Returns the response JSON, or None if the request fails.
    """
    payload = {
        "entity": fact["entity"],
        "fact_type": fact["fact_type"],
        "value": fact["value"],
        "agent_id": agent_id,
        "extraction_type": fact["extraction_type"],
        "confidence": fact["confidence"],
        "source_file": source_file
    }

    try:
        response = requests.post(
            f"{MEMORY_SERVICE_URL}/memory/write",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to write to memory service: {e}")
        return None

if __name__ == "__main__":
    test_sentences = [
        "INC0000001 has priority 2-High according to the ticket system.",
        "Monitoring logs show INC0000001 was reassigned to Desktop Support team.",
        "Field report indicates INC0000001 state is Resolved as of this morning.",
        "INC0000045 was opened with urgency level 1-High due to server outage.",
        "Transferred log entry shows INC0000001 priority as 3-Medium."
    ]

    print("Testing IT Incident Intake Agent — Extract + Write to Memory")
    print("Model: openai/gpt-oss-20b via Groq")
    print("=" * 60)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"Input:  {sentence}")

        fact = extract_fact(sentence)
        if not fact:
            print("Extraction FAILED — skipping write")
            continue

        print(f"Extracted: {json.dumps(fact, indent=2)}")

        write_result = write_to_memory(fact, source_file="manual_test")
        if write_result:
            print(f"Written to memory: fact_id={write_result.get('fact_id')}, "
                  f"entity_hash={write_result.get('entity_hash')}")
        else:
            print("Write to memory FAILED")

        print("-" * 60)