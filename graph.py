from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agents.intake_agent import extract_fact
import json

class AgentState(TypedDict):
    input_text: str
    extracted_fact: Optional[dict]
    agent_id: str
    source_file: Optional[str]
    status: str
    error: Optional[str]

def intake_node(state: AgentState) -> AgentState:
    """Intake Agent node — reads incident text and extracts one fact."""
    print(f"\nIntake Agent processing: {state['input_text']}")

    fact = extract_fact(state["input_text"])

    if fact:
        state["extracted_fact"] = fact
        state["status"] = "extracted"
        print(f"Extracted: {json.dumps(fact, indent=2)}")
    else:
        state["extracted_fact"] = None
        state["status"] = "extraction_failed"
        state["error"] = "Could not extract fact from input"
        print("Extraction failed")

    return state

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.set_entry_point("intake")
    graph.add_edge("intake", END)
    return graph.compile()

if __name__ == "__main__":
    graph = build_graph()

    test_input = {
        "input_text": "INC0000001 has priority 2-High according to the ticket system.",
        "agent_id": "intake_agent",
        "source_file": "ticket_intake.csv",
        "status": "pending",
        "extracted_fact": None,
        "error": None
    }

    print("Running LangGraph pipeline:")
    print("Model: openai/gpt-oss-20b via Groq")
    print("=" * 60)
    result = graph.invoke(test_input)
    print(f"\nFinal status: {result['status']}")
    print(f"Extracted fact: {json.dumps(result['extracted_fact'], indent=2)}")