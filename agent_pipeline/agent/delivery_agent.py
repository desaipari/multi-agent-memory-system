from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json
import re
import requests

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

MEMORY_SERVICE_URL = "http://10.125.5.158:8000"

RECOMMENDATION_PROMPT = """You are an IT incident resolution advisor.

You will be given the current known facts about an IT incident.
Based on these facts, generate ONE recommended next action or resolution step.

You must return ONLY a valid JSON object with these exact fields:
- entity: the incident ID, same as given
- fact_type: always use "resolution_action" for your output
- value: your recommended action (e.g. "Escalate to Network Team", 
  "Restart affected service", "Downgrade priority pending review")
- extraction_type: always "inferred" since this is your own derived recommendation
- confidence: number between 0.5 and 0.85 — inferred recommendations 
  should never exceed 0.85, since they are not directly stated facts

Return ONLY the JSON object.
No explanation. No extra text. No markdown backticks.

Example:
Input facts: entity=INC0000001, priority=2-High, state=New, assignment_group=Network Team
Output: {"entity": "INC0000001", "fact_type": "resolution_action", "value": "Escalate to Network Team immediately due to high priority", "extraction_type": "inferred", "confidence": 0.72}
"""


def read_from_memory(entity: str) -> list:
    """
    Reads all active facts for a given entity from the memory service.
    Returns a list of fact dicts, or empty list if none found / request fails.
    """
    try:
        response = requests.get(
            f"{MEMORY_SERVICE_URL}/memory/read",
            params={"entity": entity},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return data.get("facts", [])
    except requests.exceptions.RequestException as e:
        print(f"Failed to read from memory service: {e}")
        return []


def generate_recommendation(entity: str, facts: list) -> dict:
    """
    Takes current facts for an entity and generates a derived recommendation.
    Returns structured fact dictionary or None if generation fails.
    """
    if not facts:
        print(f"No facts found for {entity} — cannot generate recommendation")
        return None

    facts_summary = ", ".join(
        f"{f['fact_type']}={f['value']}" for f in facts
    )
    input_text = f"entity={entity}, {facts_summary}"

    messages = [
        SystemMessage(content=RECOMMENDATION_PROMPT),
        HumanMessage(content=f"Input facts: {input_text}")
    ]

    try:
        response = llm.invoke(messages)
        raw_output = response.content.strip()

        if raw_output.startswith("```"):
            raw_output = re.sub(r"```json|```", "", raw_output).strip()

        fact = json.loads(raw_output)

        required = ["entity", "fact_type", "value", "extraction_type", "confidence"]
        for field in required:
            if field not in fact:
                print(f"Missing field: {field}")
                return None

        return fact

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw output was: {raw_output}")
        return None
    except Exception as e:
        print(f"Recommendation generation error: {e}")
        return None


def write_to_memory(fact: dict, agent_id: str = "delivery_agent", source_file: str = None) -> dict:
    """
    Sends a derived fact to the memory service's /memory/write endpoint.
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
    test_entity = "INC0000001"

    print("Testing Delivery Agent — Read + Recommend + Write")
    print("Model: openai/gpt-oss-20b via Groq")
    print("=" * 60)

    print(f"\nReading current facts for {test_entity}...")
    facts = read_from_memory(test_entity)
    print(f"Found {len(facts)} facts:")
    for f in facts:
        print(f"  - {f['fact_type']}: {f['value']} (agent: {f['agent_id']}, confidence: {f['confidence']})")

    if facts:
        print(f"\nGenerating recommendation...")
        recommendation = generate_recommendation(test_entity, facts)

        if recommendation:
            print(f"Recommendation: {json.dumps(recommendation, indent=2)}")

            write_result = write_to_memory(recommendation, source_file="delivery_agent_inference")
            if write_result:
                print(f"\nWritten to memory: fact_id={write_result.get('fact_id')}")
            else:
                print("\nWrite to memory FAILED")
        else:
            print("Recommendation generation FAILED")