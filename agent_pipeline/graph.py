from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agents.intake_agent import extract_fact, write_to_memory as intake_write
from agents.delivery_agent import read_from_memory, generate_recommendation, write_to_memory as delivery_write
import json

class AgentState(TypedDict):
    input_text: str
    entity: Optional[str]
    intake_fact: Optional[dict]
    delivery_fact: Optional[dict]
    status: str
    error: Optional[str]


def intake_node(state: AgentState) -> AgentState:
    """Intake Agent: extract a fact from text and write it to memory."""
    print(f"\n[Intake Agent] Processing: {state['input_text']}")

    fact = extract_fact(state["input_text"])

    if not fact:
        state["status"] = "intake_failed"
        state["error"] = "Intake extraction failed"
        print("[Intake Agent] Extraction failed")
        return state

    print(f"[Intake Agent] Extracted: {json.dumps(fact, indent=2)}")

    write_result = intake_write(fact, source_file="chat_input")
    if not write_result:
        state["status"] = "intake_write_failed"
        state["error"] = "Failed to write intake fact to memory"
        print("[Intake Agent] Write to memory failed")
        return state

    state["entity"] = fact["entity"]
    state["intake_fact"] = fact
    state["status"] = "intake_complete"
    print(f"[Intake Agent] Written to memory: fact_id={write_result.get('fact_id')}")
    return state


def delivery_node(state: AgentState) -> AgentState:
    """Delivery Agent: read current facts, generate a recommendation, write it back."""
    entity = state.get("entity")
    if not entity:
        state["status"] = "delivery_skipped"
        state["error"] = "No entity available for Delivery Agent"
        print("[Delivery Agent] Skipped — no entity")
        return state

    print(f"\n[Delivery Agent] Reading facts for {entity}...")
    facts = read_from_memory(entity)
    print(f"[Delivery Agent] Found {len(facts)} facts")

    if not facts:
        state["status"] = "delivery_no_facts"
        state["error"] = "No facts found to base recommendation on"
        return state

    recommendation = generate_recommendation(entity, facts)
    if not recommendation:
        state["status"] = "delivery_generation_failed"
        state["error"] = "Delivery Agent failed to generate recommendation"
        return state

    print(f"[Delivery Agent] Recommendation: {json.dumps(recommendation, indent=2)}")

    write_result = delivery_write(recommendation, source_file="delivery_agent_inference")
    if not write_result:
        state["status"] = "delivery_write_failed"
        state["error"] = "Failed to write delivery recommendation to memory"
        return state

    state["delivery_fact"] = recommendation
    state["status"] = "pipeline_complete"
    print(f"[Delivery Agent] Written to memory: fact_id={write_result.get('fact_id')}")
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("delivery", delivery_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "delivery")
    graph.add_edge("delivery", END)

    return graph.compile()


if __name__ == "__main__":
    graph = build_graph()

    test_input = {
        "input_text": "INC0000045 has priority 1-High according to the ticket system.",
        "entity": None,
        "intake_fact": None,
        "delivery_fact": None,
        "status": "pending",
        "error": None
    }

    print("Running full Intake -> Delivery LangGraph pipeline")
    print("=" * 60)

    result = graph.invoke(test_input)

    print("\n" + "=" * 60)
    print(f"Final status: {result['status']}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    print(f"\nIntake fact: {json.dumps(result.get('intake_fact'), indent=2)}")
    print(f"\nDelivery fact: {json.dumps(result.get('delivery_fact'), indent=2)}")