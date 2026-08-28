from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agents.intake_agent import extract_and_store
from agents.delivery_agent import process_incident
from agents.billing_agent import extract_and_store_billing
from agents.coordinator_agent import process_flagged_conflicts
import json
import os
import requests

MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://10.125.5.158:8000")


class AgentState(TypedDict):
    input_text: str
    source_file: Optional[str]
    agent_role: str
    extracted_fact: Optional[dict]
    incident_id: Optional[str]
    memory_response: Optional[dict]
    recommendation: Optional[str]
    contradiction_detected: bool
    contradiction_details: Optional[dict]
    conflicts_resolved: int
    status: str
    error: Optional[str]


def intake_node(state: AgentState) -> AgentState:
    print(f"\n[INTAKE] Processing: {state['input_text'][:80]}...")
    result = extract_and_store(state["input_text"], "intake_agent", state.get("source_file"))
    if result["success"]:
        state["extracted_fact"] = result["fact"]
        state["memory_response"] = result["memory_response"]
        state["incident_id"] = result["fact"]["entity"]
        state["contradiction_detected"] = result["contradiction_detected"]
        state["status"] = "intake_complete"
    else:
        state["status"] = "intake_failed"
        state["error"] = result.get("error", "Intake failed")
    return state


def billing_node(state: AgentState) -> AgentState:
    print(f"\n[BILLING] Processing: {state['input_text'][:80]}...")
    result = extract_and_store_billing(state["input_text"], state.get("source_file", "field_reports.csv"))
    if result["success"]:
        state["extracted_fact"] = result["fact"]
        state["memory_response"] = result["memory_response"]
        state["incident_id"] = result["fact"]["entity"]
        state["contradiction_detected"] = result["contradiction_detected"]
        state["status"] = "billing_complete"
    else:
        state["status"] = "billing_failed"
        state["error"] = result.get("error", "Billing failed")
    return state


def delivery_node(state: AgentState) -> AgentState:
    if state["status"] in ["intake_failed", "billing_failed"]:
        print("[DELIVERY] Skipping — previous agent failed")
        return state

    incident_id = state.get("incident_id")
    if not incident_id:
        print("[DELIVERY] No incident ID — skipping")
        return state

    print(f"\n[DELIVERY] Generating recommendation for {incident_id}")
    result = process_incident(incident_id)

    if result["success"]:
        state["recommendation"] = result["recommendation"]
        state["status"] = "delivery_complete"
    else:
        state["status"] = "delivery_failed"
        state["error"] = result.get("error", "Delivery failed")
    return state


def coordinator_node(state: AgentState) -> AgentState:
    print("\n[COORDINATOR] Checking for conflicts to resolve...")
    result = process_flagged_conflicts()
    state["conflicts_resolved"] = result.get("resolved", 0)
    state["status"] = "coordinator_complete"
    print(f"[COORDINATOR] Resolved {result['resolved']} conflicts")
    return state


def route_after_intake(state: AgentState) -> str:
    if state["status"] in ["intake_failed", "billing_failed"]:
        return "end"
    if state.get("contradiction_detected"):
        print("[ROUTER] Contradiction detected — routing to Coordinator")
        return "coordinator"
    return "delivery"


def route_after_coordinator(state: AgentState) -> str:
    return "delivery" if state.get("incident_id") else "end"


def build_graph(agent_role: str = "intake"):
    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("billing", billing_node)
    graph.add_node("delivery", delivery_node)
    graph.add_node("coordinator", coordinator_node)

    entry_node = "billing" if agent_role == "billing" else "intake"
    graph.set_entry_point(entry_node)

    graph.add_conditional_edges(
        entry_node, route_after_intake,
        {"delivery": "delivery", "coordinator": "coordinator", "end": END}
    )
    graph.add_edge("delivery", END)
    graph.add_conditional_edges(
        "coordinator", route_after_coordinator,
        {"delivery": "delivery", "end": END}
    )

    return graph.compile()


def _base_state(text, source_file, agent_role):
    return {
        "input_text": text, "source_file": source_file, "agent_role": agent_role,
        "extracted_fact": None, "incident_id": None, "memory_response": None,
        "recommendation": None, "contradiction_detected": False,
        "contradiction_details": None, "conflicts_resolved": 0,
        "status": "pending", "error": None
    }


def run_intake(text: str, source_file: str = None) -> dict:
    return build_graph("intake").invoke(_base_state(text, source_file, "intake"))


def run_billing(text: str, source_file: str = "field_reports.csv") -> dict:
    return build_graph("billing").invoke(_base_state(text, source_file, "billing"))


if __name__ == "__main__":
    print("Full Four-Agent Pipeline Test")
    print("=" * 60)

    print("\n--- Test 1: Intake Agent ---")
    r1 = run_intake("INC0000099 has priority 1-Critical due to full outage.", "ticket_intake.csv")
    print(f"Status: {r1['status']}")
    print(f"Recommendation: {r1.get('recommendation')}")

    print("\n--- Test 2: Billing Agent (should conflict) ---")
    r2 = run_billing("Old log shows INC0000099 priority as 3-Medium.", "field_reports.csv")
    print(f"Status: {r2['status']}")
    print(f"Contradiction detected: {r2.get('contradiction_detected', False)}")

    print("\n--- Check conflicts at: http://10.125.5.158:8000/memory/conflicts ---")