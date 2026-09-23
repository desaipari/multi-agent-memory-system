import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import hashlib
from datetime import datetime, timezone, date, time, timedelta
from contextlib import asynccontextmanager
from verimem_core.config import (
    PRIOR_STRENGTH,
    get_expert_prior,
    MIN_WINNER_SCORE,
    MIN_WINNER_MARGIN,
)
from access_control import get_default_access, filter_facts_for_agent
from database import engine, get_db, Base
from models import (
    Fact, Agent, Conflict, AuditLog,
    HashMap, ActionGateLog, ContextTrust
)
from vector_store import (
    ensure_collection_exists,
    store_embedding,
)

# VeriMem V2 pure scoring core.
# The dashboard and agent pipeline continue to use the same REST API;
# main.py adapts database rows into the pure resolver's Observation objects.
from verimem_core.types import Observation
from verimem_core.providers import InMemoryTrustProvider
from verimem_core.resolver import resolve, normalize_value
# ---------------------------------------------------------------------------
# Frozen V2 runtime settings.
# ---------------------------------------------------------------------------
# PRIOR_STRENGTH, MIN_WINNER_SCORE and MIN_WINNER_MARGIN are imported from
# verimem_core.config so the runtime has one source of truth for calibrated
# parameters. Corroboration is frozen to noisy-OR from the V2 experiments.
V2_CORROBORATION_METHOD = "noisy_or"


Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    ensure_collection_exists()
    print("Warming up sentence transformer model...")
    from vector_store import get_encoder
    encoder = get_encoder()
    encoder.encode("warmup test", show_progress_bar=False)
    print("Ready to serve requests")
    yield

app = FastAPI(
    title="Multi-Agent Incident Memory Service",
    version="5.0.0",
    description="VeriMem V2 integrated memory service",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper Functions ───────────────────────────────────────────

def hash_value(text: str) -> str:
    return hashlib.sha256(
        text.strip().lower().encode()
    ).hexdigest()


def canonical_fact_type(fact_type: str) -> str:
    """
    Normalize equivalent fact-type names before storage/scoring.

    This prevents the same skill from being split into separate trust buckets,
    e.g. "incident_state" and "state".
    """
    value = (fact_type or "").lower().strip()
    aliases = {
        "incident_state": "state",
        "incident state": "state",
        "incident-state": "state",
        "assignment group": "assignment_group",
        "assignment-group": "assignment_group",
        "opened date": "opened_date",
        "opened-date": "opened_date",
        "resolved by": "resolved_by",
        "resolved-by": "resolved_by",
    }
    return aliases.get(value, value)


def store_hash_mapping(db: Session, original: str, hashed: str):
    existing = db.query(HashMap).filter(
        HashMap.hash_value == hashed
    ).first()
    if not existing:
        db.add(HashMap(hash_value=hashed, original_value=original))
        db.flush()


def ensure_agent_exists(db: Session, agent_id: str):
    """
    Keep the legacy Agent row for the existing dashboard scoreboard.

    V2 decision-making does NOT use this global score. Actual resolution trust
    is contextual R(agent, fact_type) and comes from ContextTrust + expert prior.
    """
    agent = db.query(Agent).filter(
        Agent.agent_id == agent_id
    ).first()

    if not agent:
        initial_trust = get_expert_prior(agent_id, "__default__")
        db.add(Agent(
            agent_id=agent_id,
            trust_score=initial_trust,
            reliability_score=initial_trust,
        ))
        db.flush()


def build_trust_provider(db: Session) -> InMemoryTrustProvider:
    """
    Snapshot verified contextual trust from PostgreSQL into memory.

    The resolver itself remains pure and performs no database queries.
    V2 learns R(agent, fact_type); extraction directness is a separate
    confidence component, so any historical extraction-specific rows are
    aggregated into the same agent/fact bucket.
    """
    provider = InMemoryTrustProvider()
    aggregate = {}

    for row in db.query(ContextTrust).all():
        fact_type = canonical_fact_type(row.fact_type)
        key = (row.agent_id, fact_type)
        if key not in aggregate:
            aggregate[key] = [0, 0]
        aggregate[key][0] += row.successes or 0
        aggregate[key][1] += row.failures or 0

    for (agent_id, fact_type), (successes, failures) in aggregate.items():
        provider.set_stats(
            agent_id=agent_id,
            fact_type=fact_type,
            successes=successes,
            failures=failures,
        )

    return provider


def fact_to_observation(fact: Fact) -> Observation:
    """Convert a database Fact into the pure V2 resolver input type."""
    return Observation(
        fact_id=fact.fact_id,
        agent_id=fact.agent_id,
        fact_type=canonical_fact_type(fact.fact_type),
        value=fact.raw_value or "",
        extraction_type=fact.extraction_type or "direct",
        # observed_at is event time. We deliberately do not substitute the
        # ingestion timestamp when it is missing; V2 uses a neutral time
        # penalty for missing observation time.
        observed_at=fact.observed_at,
    )


def candidate_fact_id(candidate: Optional[dict]) -> Optional[str]:
    if not candidate:
        return None
    evidence = candidate.get("evidence") or []
    return evidence[0].get("fact_id") if evidence else None


def apply_candidate_scores(facts, result: dict):
    """
    Store the candidate-level V2 score back on each supporting Fact.

    This preserves the existing dashboard contract: the dashboard can continue
    displaying Fact.confidence without needing any frontend changes.
    """
    candidate_by_value = {
        candidate["normalized_value"]: candidate
        for candidate in result.get("candidates", [])
    }

    for fact in facts:
        candidate = candidate_by_value.get(
            normalize_value(fact.raw_value)
        )
        if candidate:
            fact.confidence = candidate["score"]
            fact.corroboration_count = candidate["distinct_agent_count"]


def get_context_row(db: Session, agent_id: str, fact_type: str) -> ContextTrust:
    """
    Return/create the single V2 contextual-trust row for an agent/fact pair.

    extraction_type='*' is used for compatibility with the existing table
    schema because V2 learns R(agent,fact), not R(agent,fact,extraction).
    """
    fact_type = canonical_fact_type(fact_type)
    row = db.query(ContextTrust).filter(
        ContextTrust.agent_id == agent_id,
        ContextTrust.fact_type == fact_type,
        ContextTrust.extraction_type == "*",
    ).first()

    if not row:
        row = ContextTrust(
            agent_id=agent_id,
            fact_type=fact_type,
            extraction_type="*",
            successes=0,
            failures=0,
        )
        db.add(row)
        db.flush()

    return row


def refresh_dashboard_agent_score(db: Session, agent_id: str):
    """
    Update the legacy global Agent score for display only.

    The dashboard expects one trust_score per agent. We keep that API stable by
    showing the mean of the agent's contextual posterior estimates. Resolution
    logic itself never uses this aggregate value.
    """
    agent = db.query(Agent).filter(
        Agent.agent_id == agent_id
    ).first()
    if not agent:
        return

    provider = build_trust_provider(db)
    fact_types = {
        canonical_fact_type(row.fact_type)
        for row in db.query(ContextTrust).filter(
            ContextTrust.agent_id == agent_id
        ).all()
    }

    if fact_types:
        values = [
            provider.get_trust(
                agent_id=agent_id,
                fact_type=fact_type,
                prior_strength=PRIOR_STRENGTH,
            )["probability"]
            for fact_type in fact_types
        ]
        score = sum(values) / len(values)
    else:
        score = get_expert_prior(agent_id, "__default__")

    agent.trust_score = score
    agent.reliability_score = score


def record_verified_outcome(
    db: Session,
    agent_id: str,
    fact_type: str,
    correct: bool,
):
    """
    Update contextual trust only from an externally verified resolution.

    Auto-resolutions never train trust. This prevents a self-reinforcing loop
    where the system rewards whichever source it already preferred.
    """
    row = get_context_row(db, agent_id, fact_type)

    if correct:
        row.successes = (row.successes or 0) + 1
    else:
        row.failures = (row.failures or 0) + 1

    agent = db.query(Agent).filter(
        Agent.agent_id == agent_id
    ).first()
    if agent:
        if correct:
            agent.correct_writes = (agent.correct_writes or 0) + 1
        else:
            agent.overturned_writes = (agent.overturned_writes or 0) + 1

    db.flush()
    refresh_dashboard_agent_score(db, agent_id)


def is_verified_human(resolved_by: str) -> bool:
    """
    Coordinator/system resolutions may still use the existing endpoint for
    compatibility, but they do not become trust-training labels.
    """
    value = (resolved_by or "").strip().lower()
    return value not in {
        "coordinator",
        "coordinator_agent",
        "system",
        "auto",
        "automatic",
    }


def _resolve_current_candidates(
    db: Session,
    facts,
    entity_hash: str,
    fact_type: str,
    detection_method: str = "database_candidate_grouping",
) -> Optional[dict]:
    """
    Score all current candidate values for one entity/fact type using V2.

    All active/contested observations are supplied together, so corroboration
    is computed from additional independent supporters instead of from a simple
    counter. The current Conflict table is pairwise, therefore the database
    record stores the top two candidate representatives while the resolver can
    still consider more than two candidates internally.
    """
    if not facts:
        return None

    provider = build_trust_provider(db)
    observations = [fact_to_observation(fact) for fact in facts]

    result = resolve(
    observations=observations,
    provider=provider,
    prior_strength=PRIOR_STRENGTH,
    corroboration_method=V2_CORROBORATION_METHOD,
    min_winner_score=MIN_WINNER_SCORE,
    min_margin=MIN_WINNER_MARGIN,
)

    apply_candidate_scores(facts, result)

    # One candidate value means corroboration/no conflict.
    if result["decision"] == "no_conflict":
        return None

    winner_candidate = result.get("winner")
    runner_up_candidate = result.get("runner_up")
    winner_id = candidate_fact_id(winner_candidate)
    runner_up_id = candidate_fact_id(runner_up_candidate)

    if not winner_id or not runner_up_id:
        return None

    winner_fact = next(f for f in facts if f.fact_id == winner_id)
    runner_up_fact = next(f for f in facts if f.fact_id == runner_up_id)
    margin = result.get("margin") or 0.0

    conflict = Conflict(
        fact_id_a=winner_fact.fact_id,
        fact_id_b=runner_up_fact.fact_id,
        entity_hash=entity_hash,
        fact_type=fact_type,
        status="flagged",
    )
    db.add(conflict)
    db.flush()

    print(f"\nCONTRADICTION DETECTED ({detection_method})")
    print(f"  fact_type: {fact_type}")
    print(
        f"  Winner candidate: {winner_candidate['value']} "
        f"(score:{winner_candidate['score']:.3f})"
    )
    print(
        f"  Runner-up: {runner_up_candidate['value']} "
        f"(score:{runner_up_candidate['score']:.3f})"
    )

    if result["decision"] == "auto_resolve":
        winning_value = winner_candidate["normalized_value"]

        # All observations supporting the winning candidate remain active.
        # Every other candidate is superseded only after V2 resolution.
        for fact in facts:
            if normalize_value(fact.raw_value) == winning_value:
                fact.status = "active"
                fact.conflict_id = None
            else:
                fact.status = "superseded"
                fact.superseded_by = winner_fact.fact_id
                fact.conflict_id = None

        conflict.status = "auto_resolved"
        conflict.resolved_winner = winner_fact.fact_id
        conflict.resolution_type = "auto_resolve"
        conflict.resolution_reason = (
            f"V2 auto-resolution: winner score {winner_candidate['score']:.3f} "
            f"(required >= {MIN_WINNER_SCORE:.2f}), margin {margin:.3f} "
            f"(required >= {MIN_WINNER_MARGIN:.2f}). "
            f"No trust learning is performed from automatic decisions."
        )
        conflict.resolved_at = datetime.now(timezone.utc)

        db.add(AuditLog(
            event_type="auto_resolved",
            fact_id=winner_fact.fact_id,
            agent_id=winner_fact.agent_id,
            description=(
                f"V2 auto-resolved {fact_type}: "
                f"'{winner_candidate['value']}' won with score "
                f"{winner_candidate['score']:.3f} and margin {margin:.3f}. "
                f"Automatic resolution did not update contextual trust."
            ),
        ))

        return {
            "conflict_id": conflict.conflict_id,
            "resolution": "auto_resolved",
            "winner_value": winner_candidate["value"],
            "winner_agent": winner_fact.agent_id,
            "loser_value": runner_up_candidate["value"],
            "loser_agent": runner_up_fact.agent_id,
            "winner_score": round(winner_candidate["score"], 4),
            "runner_up_score": round(runner_up_candidate["score"], 4),
            "confidence_gap": round(margin, 4),
            "detection_method": detection_method,
        }

    # Uncertain evidence is retained rather than overwritten.
    for fact in facts:
        fact.status = "contested"
        fact.conflict_id = conflict.conflict_id

    conflict.status = "flagged"
    conflict.resolution_type = "human_review"
    conflict.resolution_reason = (
        f"V2 contested: winner score {winner_candidate['score']:.3f} "
        f"(required >= {MIN_WINNER_SCORE:.2f}), runner-up "
        f"{runner_up_candidate['score']:.3f}, margin {margin:.3f} "
        f"(required >= {MIN_WINNER_MARGIN:.2f}). Human review required."
    )

    db.add(AuditLog(
        event_type="conflict_detected",
        fact_id=winner_fact.fact_id,
        agent_id=winner_fact.agent_id,
        description=(
            f"V2 contested {fact_type}: "
            f"'{winner_candidate['value']}' ({winner_candidate['score']:.3f}) "
            f"vs '{runner_up_candidate['value']}' "
            f"({runner_up_candidate['score']:.3f}); "
            f"margin {margin:.3f}. Human review required."
        ),
    ))

    return {
        "conflict_id": conflict.conflict_id,
        "resolution": "contested",
        "value_a": winner_candidate["value"],
        "agent_a": winner_fact.agent_id,
        "confidence_a": winner_candidate["score"],
        "value_b": runner_up_candidate["value"],
        "agent_b": runner_up_fact.agent_id,
        "confidence_b": runner_up_candidate["score"],
        "confidence_gap": round(margin, 4),
        "detection_method": detection_method,
        "message": "Flagged for human review",
    }


# ── Request Models ─────────────────────────────────────────────

class MemoryWriteRequest(BaseModel):
    entity: str
    fact_type: str
    value: str
    agent_id: str
    extraction_type: Optional[str] = "direct"
    # Kept for backward compatibility with the current agent pipeline.
    # V2 does not trust caller-supplied confidence as the governed score.
    confidence: Optional[float] = None
    source_file: Optional[str] = None
    # Event/observation time. If omitted, V2 intentionally uses a neutral
    # temporal penalty rather than pretending ingestion time is observation time.
    observed_at: Optional[datetime] = None


class ResolveRequest(BaseModel):
    conflict_id: str
    winning_fact_id: str
    resolved_by: str = "human"
    reason: Optional[str] = None


class ActionCheckRequest(BaseModel):
    agent_id: str
    entity: str
    fact_type: str
    action_attempted: Optional[str] = None
    confidence_threshold: Optional[float] = 0.60


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/memory/health")
def health_check():
    return {
        "status": "running",
        "service": "Multi-Agent Incident Memory Service",
        "version": "5.0.0",
        "features": [
            "SHA-256 hashing",
            "Qdrant semantic search",
            "Confidence scoring",
            "Contradiction detection",
            "Auto-resolution",
            "Contested state",
            "Action gating"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/memory/write")
def write_memory(
    request: MemoryWriteRequest,
    db: Session = Depends(get_db)
):
    """
    Store a fact and score the current candidate set with VeriMem V2.

    Runtime invariants:
    1. Caller-supplied confidence never bypasses governed scoring.
    2. R(agent,fact_type) comes from expert prior + verified ContextTrust history.
    3. Corroboration uses independent supporters through noisy-OR.
    4. Newer values are evidence, not automatic truth.
    5. Auto-resolution never trains trust; human-verified resolution does.
    """
    fact_type = canonical_fact_type(request.fact_type)
    ensure_agent_exists(db, request.agent_id)

    entity_hash = hash_value(request.entity)
    value_hash = hash_value(request.value)

    store_hash_mapping(db, request.entity, entity_hash)
    store_hash_mapping(db, request.value, value_hash)

    # Was the same candidate already supported before this write?
    prior_same_value = db.query(Fact).filter(
        Fact.entity_hash == entity_hash,
        Fact.fact_type == fact_type,
        Fact.status.in_(["active", "contested"]),
    ).all()
    prior_same_value = any(
        normalize_value(f.raw_value) == normalize_value(request.value)
        and f.agent_id != request.agent_id
        for f in prior_same_value
    )

    # Confidence starts as a placeholder. After db.flush(), the complete
    # candidate set is scored by the pure V2 resolver and this value is replaced.
    fact = Fact(
        entity_hash=entity_hash,
        fact_type=fact_type,
        value_hash=value_hash,
        raw_value=request.value,
        agent_id=request.agent_id,
        confidence=0.0,
        status="active",
        extraction_type=request.extraction_type or "direct",
        source_file=request.source_file,
        corroboration_count=1,
        readable_by=get_default_access(fact_type),
        observed_at=request.observed_at,
    )
    db.add(fact)
    db.flush()  # ensures fact_id exists before Qdrant/audit use

    agent = db.query(Agent).filter(
        Agent.agent_id == request.agent_id
    ).first()
    if agent:
        agent.total_writes = (agent.total_writes or 0) + 1

    # Keep Qdrant populated for semantic retrieval without using it as a
    # substitute for candidate-level contradiction reasoning.
    store_embedding(
        fact_id=fact.fact_id,
        entity_hash=entity_hash,
        fact_type=fact_type,
        value=request.value,
        agent_id=request.agent_id,
    )

    current_facts = db.query(Fact).filter(
        Fact.entity_hash == entity_hash,
        Fact.fact_type == fact_type,
        Fact.status.in_(["active", "contested"]),
    ).all()

    # Score all current candidates together. If there is more than one value,
    # this also creates/updates the Conflict state while preserving the existing
    # dashboard response schema.
    contradiction_result = _resolve_current_candidates(
        db=db,
        facts=current_facts,
        entity_hash=entity_hash,
        fact_type=fact_type,
    )

    # If there is only one candidate, run V2 once to obtain its governed score.
    # _resolve_current_candidates already applied scores before returning None.
    computed_confidence = fact.confidence

    if prior_same_value:
        db.add(AuditLog(
            event_type="corroboration",
            fact_id=fact.fact_id,
            agent_id=request.agent_id,
            description=(
                f"{request.agent_id} independently corroborated "
                f"{fact_type}={request.value} for {request.entity}. "
                f"V2 candidate confidence is {computed_confidence:.3f}."
            ),
        ))

    db.add(AuditLog(
        event_type="write",
        fact_id=fact.fact_id,
        agent_id=request.agent_id,
        description=(
            f"{request.agent_id} wrote {fact_type}={request.value} "
            f"for {request.entity} (V2 confidence:{computed_confidence:.3f}) "
            f"from {request.source_file or 'text input'}"
        ),
    ))

    db.commit()
    db.refresh(fact)

    return {
        "fact_id": fact.fact_id,
        "entity_hash": entity_hash,
        "fact_type": fact.fact_type,
        "value_hash": value_hash,
        "agent_id": request.agent_id,
        "confidence": fact.confidence,
        "status": fact.status,
        "corroboration": prior_same_value,
        "contradiction_detected": contradiction_result is not None,
        "contradiction": contradiction_result,
        "message": (
            "Corroboration — candidate confidence updated"
            if prior_same_value and contradiction_result is None
            else "Contradiction detected — flagged for human review"
            if contradiction_result
            and contradiction_result.get("resolution") == "contested"
            else "Contradiction detected and auto-resolved"
            if contradiction_result
            and contradiction_result.get("resolution") == "auto_resolved"
            else "Fact stored successfully"
        ),
    }


@app.get("/memory/read")
def read_memory(
    entity: str,
    fact_type: Optional[str] = None,
    agent_id: Optional[str] = None,  # who is asking
    confidence_threshold: Optional[float] = None,
    db: Session = Depends(get_db)
):
    entity_hash = hash_value(entity)

    query = db.query(Fact).filter(
        Fact.entity_hash == entity_hash
    )
    if fact_type:
        query = query.filter(
            Fact.fact_type == canonical_fact_type(fact_type)
        )

    facts = query.filter(
        Fact.status.in_(["active", "contested"])
    ).all()

    if not facts:
        raise HTTPException(
            status_code=404,
            detail=f"No facts found for: {entity}"
        )

    # Apply role-based access filter
    if agent_id:
        facts = filter_facts_for_agent(facts, agent_id)

    results = []
    blocked = []
    access_denied = []

    for fact in facts:
        fact_data = {
            "fact_id": fact.fact_id,
            "entity": entity,
            "fact_type": fact.fact_type,
            "value": fact.raw_value,
            "agent_id": fact.agent_id,
            "confidence": fact.confidence,
            "status": fact.status,
            "extraction_type": fact.extraction_type,
            "source_file": fact.source_file,
            "corroboration_count": fact.corroboration_count,
            "readable_by": fact.readable_by,
            "timestamp": fact.timestamp.isoformat(),
            "observed_at": fact.observed_at.isoformat() if fact.observed_at else None
        }

        if fact.status == "contested":
            blocked.append({
                **fact_data,
                "blocked_reason": "Fact contested",
                "action": "BLOCKED"
            })
            if agent_id:
                db.add(ActionGateLog(
                    agent_id=agent_id,
                    entity=entity,
                    fact_type=fact.fact_type,
                    blocked_reason="Fact is contested",
                    confidence_at_block=fact.confidence,
                    conflict_id=fact.conflict_id
                ))
                db.commit()
        elif (confidence_threshold and
              fact.confidence < confidence_threshold):
            blocked.append({
                **fact_data,
                "blocked_reason": (
                    f"Confidence {fact.confidence:.3f} "
                    f"below threshold {confidence_threshold}"
                ),
                "action": "BLOCKED"
            })
            if agent_id:
                db.add(ActionGateLog(
                    agent_id=agent_id,
                    entity=entity,
                    fact_type=fact.fact_type,
                    blocked_reason=(
                        f"Confidence {fact.confidence:.3f} "
                        f"below threshold"
                    ),
                    confidence_at_block=fact.confidence
                ))
                db.commit()
        else:
            results.append(fact_data)

    return {
        "entity": entity,
        "requesting_agent": agent_id,
        "facts": results,
        "blocked": blocked,
        "has_blocked": len(blocked) > 0,
        "count": len(results)
    }

@app.get("/memory/all")
def get_all_facts(db: Session = Depends(get_db)):
    """
    All facts for dashboard memory state table.
    Optimized: single query for facts, single query for hash map,
    then join in Python instead of N+1 database round trips.
    """
    facts = db.query(Fact).order_by(
        Fact.timestamp.desc()
    ).all()

    if not facts:
        return {"facts": [], "total": 0}

    # Single query to get all hash mappings we need
    # instead of one query per fact
    entity_hashes = list(set(f.entity_hash for f in facts))
    hash_mappings = {
        hm.hash_value: hm.original_value
        for hm in db.query(HashMap).filter(
            HashMap.hash_value.in_(entity_hashes)
        ).all()
    }

    results = []
    for fact in facts:
        entity_readable = hash_mappings.get(
            fact.entity_hash,
            fact.entity_hash[:16]
        )
        results.append({
            "fact_id": fact.fact_id,
            "entity": entity_readable,
            "entity_hash": fact.entity_hash,
            "fact_type": fact.fact_type,
            "value": fact.raw_value,
            "value_hash": fact.value_hash,
            "agent_id": fact.agent_id,
            "confidence": fact.confidence,
            "status": fact.status,
            "extraction_type": fact.extraction_type,
            "source_file": fact.source_file,
            "corroboration_count": fact.corroboration_count,
            "superseded_by": fact.superseded_by,
            "timestamp": fact.timestamp.isoformat(),
            "observed_at": fact.observed_at.isoformat() if fact.observed_at else None
        })

    return {"facts": results, "total": len(results)}

@app.get("/memory/conflicts")
def get_conflicts(db: Session = Depends(get_db)):
    """
    All conflicts for dashboard.
    Optimized: batch load all related facts and hash mappings.
    """
    conflicts = db.query(Conflict).order_by(
        Conflict.timestamp.desc()
    ).all()

    if not conflicts:
        return {"conflicts": [], "total": 0}

    # Batch load all facts referenced by conflicts
    all_fact_ids = []
    for c in conflicts:
        if c.fact_id_a:
            all_fact_ids.append(c.fact_id_a)
        if c.fact_id_b:
            all_fact_ids.append(c.fact_id_b)

    facts_by_id = {
        f.fact_id: f
        for f in db.query(Fact).filter(
            Fact.fact_id.in_(list(set(all_fact_ids)))
        ).all()
    }

    # Batch load all hash mappings
    entity_hashes = list(set(c.entity_hash for c in conflicts))
    hash_mappings = {
        hm.hash_value: hm.original_value
        for hm in db.query(HashMap).filter(
            HashMap.hash_value.in_(entity_hashes)
        ).all()
    }

    results = []
    for conflict in conflicts:
        fact_a = facts_by_id.get(conflict.fact_id_a)
        fact_b = facts_by_id.get(conflict.fact_id_b)
        entity_readable = hash_mappings.get(
            conflict.entity_hash,
            conflict.entity_hash[:16]
        )

        conf_a = fact_a.confidence if fact_a else None
        conf_b = fact_b.confidence if fact_b else None
        gap = (
            round(abs(conf_a - conf_b), 4)
            if conf_a is not None and conf_b is not None
            else None
        )

        results.append({
            "conflict_id": conflict.conflict_id,
            "entity": entity_readable,
            "fact_type": conflict.fact_type,
            "status": conflict.status,
            "resolution_type": conflict.resolution_type,
            "resolution_reason": conflict.resolution_reason,
            "resolved_winner": conflict.resolved_winner,
            "detected_at": conflict.timestamp.isoformat(),
            "resolved_at": (
                conflict.resolved_at.isoformat()
                if conflict.resolved_at else None
            ),
            "value_a": fact_a.raw_value if fact_a else None,
            "agent_a": fact_a.agent_id if fact_a else None,
            "confidence_a": conf_a,
            "status_a": fact_a.status if fact_a else None,
            "value_b": fact_b.raw_value if fact_b else None,
            "agent_b": fact_b.agent_id if fact_b else None,
            "confidence_b": conf_b,
            "status_b": fact_b.status if fact_b else None,
            "confidence_gap": gap,
            "fact_id_a": conflict.fact_id_a,
            "fact_id_b": conflict.fact_id_b,
        })

    return {"conflicts": results, "total": len(results)}


@app.post("/memory/resolve")
def resolve_conflict(
    request: ResolveRequest,
    db: Session = Depends(get_db)
):
    """
    Resolve a contested conflict without changing the dashboard API.

    A human-verified resolution updates R(agent,fact_type). Coordinator/system
    resolutions may remain compatible with the existing pipeline, but they are
    deliberately excluded from trust learning because they are not independent
    verification labels.
    """
    conflict = db.query(Conflict).filter(
        Conflict.conflict_id == request.conflict_id
    ).first()

    if not conflict:
        raise HTTPException(
            status_code=404,
            detail=f"Conflict not found: {request.conflict_id}",
        )

    # Idempotency: repeated resolve requests must not double-count trust labels.
    if conflict.status in ["human_resolved", "auto_resolved"]:
        return {
            "status": "already_resolved",
            "conflict_id": conflict.conflict_id,
            "winner": conflict.resolved_winner,
        }

    if conflict.status != "flagged":
        raise HTTPException(
            status_code=400,
            detail=f"Conflict already resolved: {conflict.status}",
        )

    if request.winning_fact_id not in {
        conflict.fact_id_a,
        conflict.fact_id_b,
    }:
        raise HTTPException(
            status_code=400,
            detail="winning_fact_id must be one of the two displayed conflict candidates",
        )

    selected_fact = db.query(Fact).filter(
        Fact.fact_id == request.winning_fact_id
    ).first()

    if not selected_fact:
        raise HTTPException(
            status_code=404,
            detail="Could not find selected winning fact",
        )

    winning_value = normalize_value(selected_fact.raw_value)

    # Preserve the original dashboard/API loser fields for the displayed pair.
    display_loser_id = (
        conflict.fact_id_b
        if selected_fact.fact_id == conflict.fact_id_a
        else conflict.fact_id_a
    )
    display_loser_fact = db.query(Fact).filter(
        Fact.fact_id == display_loser_id
    ).first()

    # The resolver may have considered >2 observations even though the existing
    # Conflict table/dashboard shows the top two candidates. Resolve every fact
    # attached to this conflict consistently with the selected candidate value.
    conflict_facts = db.query(Fact).filter(
        Fact.entity_hash == conflict.entity_hash,
        Fact.fact_type == conflict.fact_type,
        Fact.conflict_id == conflict.conflict_id,
    ).all()

    if not conflict_facts:
        # Backward compatibility for older pairwise conflicts.
        conflict_facts = db.query(Fact).filter(
            Fact.fact_id.in_([conflict.fact_id_a, conflict.fact_id_b])
        ).all()

    winner_facts = []
    loser_facts = []
    for fact in conflict_facts:
        if normalize_value(fact.raw_value) == winning_value:
            fact.status = "active"
            fact.conflict_id = None
            winner_facts.append(fact)
        else:
            fact.status = "superseded"
            fact.superseded_by = selected_fact.fact_id
            fact.conflict_id = None
            loser_facts.append(fact)

    conflict.status = "human_resolved"
    conflict.resolved_winner = selected_fact.fact_id
    conflict.resolution_type = "human_review"
    conflict.resolution_reason = (
        request.reason or f"Resolved by {request.resolved_by}"
    )
    conflict.resolved_at = datetime.now(timezone.utc)

    # Verified-outcome learning: one correctness label per agent/fact context.
    # Automatic or Coordinator/system decisions do NOT update contextual trust.
    trust_updated = False
    if is_verified_human(request.resolved_by):
        by_agent = {}
        for fact in sorted(
            conflict_facts,
            key=lambda f: (f.timestamp or datetime.min),
        ):
            by_agent[fact.agent_id] = fact

        for agent_id, fact in by_agent.items():
            correct = normalize_value(fact.raw_value) == winning_value
            record_verified_outcome(
                db=db,
                agent_id=agent_id,
                fact_type=conflict.fact_type,
                correct=correct,
            )
        trust_updated = True

    db.add(AuditLog(
        event_type="human_resolved",
        fact_id=selected_fact.fact_id,
        agent_id=request.resolved_by,
        description=(
            f"Resolved {conflict.fact_type}: '{selected_fact.raw_value}' selected. "
            f"Verified contextual trust update: {trust_updated}. "
            f"Reason: {request.reason or 'Not specified'}"
        ),
    ))

    db.commit()

    return {
        "conflict_id": conflict.conflict_id,
        "status": "human_resolved",
        "winner_value": selected_fact.raw_value,
        "winner_agent": selected_fact.agent_id,
        "loser_value": (
            display_loser_fact.raw_value if display_loser_fact else None
        ),
        "loser_agent": (
            display_loser_fact.agent_id if display_loser_fact else None
        ),
        # Extra field for V2 multi-candidate conflicts; old frontend can ignore it.
        "loser_values": [fact.raw_value for fact in loser_facts],
        "trust_updated": trust_updated,
        "message": "Conflict resolved successfully",
    }


@app.post("/memory/check_action")
def check_action_gate(
    request: ActionCheckRequest,
    db: Session = Depends(get_db)
):
    """
    Check whether an agent may proceed with an action.

    Blocked actions are written to both ActionGateLog (specialized gate log)
    and AuditLog (full dashboard audit trail), so the frontend
    event_type=action_blocked filter returns these events.
    """
    entity_hash = hash_value(request.entity)
    fact_type = canonical_fact_type(request.fact_type)

    fact = db.query(Fact).filter(
        Fact.entity_hash == entity_hash,
        Fact.fact_type == fact_type,
        Fact.status.in_(["active", "contested"])
    ).order_by(Fact.timestamp.desc()).first()

    if not fact:
        return {
            "allowed": False,
            "reason": f"No fact found for {request.entity} / {request.fact_type}",
            "action": "BLOCKED"
        }

    threshold = request.confidence_threshold or 0.60

    if fact.status == "contested":
        blocked_reason = "Fact is contested — conflict unresolved"

        db.add(ActionGateLog(
            agent_id=request.agent_id,
            entity=request.entity,
            fact_type=fact_type,
            action_attempted=request.action_attempted,
            blocked_reason=blocked_reason,
            confidence_at_block=fact.confidence,
            conflict_id=fact.conflict_id
        ))

        db.add(AuditLog(
            event_type="action_blocked",
            fact_id=fact.fact_id,
            agent_id=request.agent_id,
            description=(
                f"Action blocked for {request.entity} / {fact_type}: "
                f"fact is contested. Attempted action: "
                f"{request.action_attempted or 'unspecified'}"
            )
        ))

        db.commit()

        return {
            "allowed": False,
            "reason": "Fact is contested — resolve conflict before proceeding",
            "fact_value": fact.raw_value,
            "confidence": fact.confidence,
            "status": fact.status,
            "action": "BLOCKED"
        }

    if fact.confidence < threshold:
        blocked_reason = (
            f"Confidence {fact.confidence:.3f} below threshold {threshold:.3f}"
        )

        db.add(ActionGateLog(
            agent_id=request.agent_id,
            entity=request.entity,
            fact_type=fact_type,
            action_attempted=request.action_attempted,
            blocked_reason=blocked_reason,
            confidence_at_block=fact.confidence
        ))

        db.add(AuditLog(
            event_type="action_blocked",
            fact_id=fact.fact_id,
            agent_id=request.agent_id,
            description=(
                f"Action blocked for {request.entity} / {fact_type}: "
                f"confidence {fact.confidence:.3f} below required threshold "
                f"{threshold:.3f}. Attempted action: "
                f"{request.action_attempted or 'unspecified'}"
            )
        ))

        db.commit()

        return {
            "allowed": False,
            "reason": blocked_reason,
            "fact_value": fact.raw_value,
            "confidence": fact.confidence,
            "status": fact.status,
            "action": "BLOCKED"
        }

    return {
        "allowed": True,
        "reason": "Fact is active and confidence is sufficient",
        "fact_value": fact.raw_value,
        "confidence": fact.confidence,
        "status": fact.status,
        "action": "ALLOWED"
    }


@app.get("/memory/agents")
def get_agents(db: Session = Depends(get_db)):
    """Agent trust scoreboard for dashboard."""
    agents = db.query(Agent).all()
    return {
        "agents": [
            {
                "agent_id": a.agent_id,
                "trust_score": a.trust_score,
                "reliability_score": a.reliability_score,
                "total_writes": a.total_writes or 0,
                "correct_writes": a.correct_writes or 0,
                "overturned_writes": a.overturned_writes or 0
            }
            for a in agents
        ]
    }


@app.get("/memory/audit")
def get_audit_log(
    entity: Optional[str] = None,
    agent_id: Optional[str] = None,
    event_type: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Return the full audit trail with optional dashboard filters.

    Supported filters:
      - entity
      - agent_id
      - event_type
      - date_from (inclusive)
      - date_to (inclusive)

    Results are always ordered newest first.

    Examples:
      GET /memory/audit
      GET /memory/audit?agent_id=billing_agent
      GET /memory/audit?event_type=conflict_detected
      GET /memory/audit?entity=INC0000001
      GET /memory/audit?date_from=2026-09-20&date_to=2026-09-22
    """
    # Keep dashboard/API requests bounded.
    limit = max(1, min(limit, 500))

    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=400,
            detail="date_from cannot be after date_to"
        )

    query = db.query(AuditLog)

    if agent_id:
        query = query.filter(
            AuditLog.agent_id == agent_id.strip()
        )

    if event_type:
        query = query.filter(
            AuditLog.event_type == event_type.strip().lower()
        )

    # AuditLog stores fact_id, not entity. Resolve the incident to its fact IDs.
    if entity:
        entity_hash = hash_value(entity)
        fact_ids = [
            fact.fact_id
            for fact in db.query(Fact).filter(
                Fact.entity_hash == entity_hash
            ).all()
        ]

        if not fact_ids:
            return {
                "logs": [],
                "total": 0,
                "filters": {
                    "entity": entity,
                    "agent_id": agent_id,
                    "event_type": event_type,
                    "date_from": date_from.isoformat() if date_from else None,
                    "date_to": date_to.isoformat() if date_to else None
                }
            }

        query = query.filter(
            AuditLog.fact_id.in_(fact_ids)
        )

    # Inclusive lower date boundary: YYYY-MM-DD 00:00:00 UTC.
    if date_from:
        start_datetime = datetime.combine(
            date_from,
            time.min,
            tzinfo=timezone.utc
        )
        query = query.filter(
            AuditLog.timestamp >= start_datetime
        )

    # Inclusive upper calendar date is implemented as an exclusive boundary at
    # midnight of the following day. This includes the whole selected date.
    if date_to:
        end_datetime = datetime.combine(
            date_to + timedelta(days=1),
            time.min,
            tzinfo=timezone.utc
        )
        query = query.filter(
            AuditLog.timestamp < end_datetime
        )

    logs = (
        query
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )

    return {
        "logs": [
            {
                "log_id": log.log_id,
                "event_type": log.event_type,
                "fact_id": log.fact_id,
                "agent_id": log.agent_id,
                "description": log.description,
                "timestamp": (
                    log.timestamp.isoformat()
                    if log.timestamp else None
                )
            }
            for log in logs
        ],
        "total": len(logs),
        "filters": {
            "entity": entity,
            "agent_id": agent_id,
            "event_type": event_type,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None
        }
    }


@app.get("/memory/action_gate_log")
def get_action_gate_log(db: Session = Depends(get_db)):
    """Action gate log for dashboard."""
    logs = db.query(ActionGateLog).order_by(
        ActionGateLog.timestamp.desc()
    ).all()

    return {
        "logs": [
            {
                "gate_id": log.gate_id,
                "agent_id": log.agent_id,
                "entity": log.entity,
                "fact_type": log.fact_type,
                "action_attempted": log.action_attempted,
                "blocked_reason": log.blocked_reason,
                "confidence_at_block": log.confidence_at_block,
                "timestamp": log.timestamp.isoformat()
            }
            for log in logs
        ]
    }

@app.get("/memory/resolution_feed")
def get_resolution_feed(
    since_seconds: int = 300,
    db: Session = Depends(get_db)
):
    """
    Returns all resolutions in the last N seconds.
    Agents can poll this before acting to check if any
    facts they plan to use were recently updated.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=since_seconds
    )

    recent_resolutions = db.query(Conflict).filter(
        Conflict.status.in_(["auto_resolved", "human_resolved"]),
        Conflict.resolved_at >= cutoff
    ).all()

    results = []
    for c in recent_resolutions:
        winner = db.query(Fact).filter(
            Fact.fact_id == c.resolved_winner
        ).first()
        entity_map = db.query(HashMap).filter(
            HashMap.hash_value == c.entity_hash
        ).first()

        results.append({
            "conflict_id": c.conflict_id,
            "entity": entity_map.original_value if entity_map else c.entity_hash[:16],
            "fact_type": c.fact_type,
            "resolved_winner": c.resolved_winner,
            "current_value": winner.raw_value if winner else None,
            "resolution_type": c.resolution_type,
            "resolved_at": c.resolved_at.isoformat() if c.resolved_at else None
        })

    return {
        "recent_resolutions": results,
        "count": len(results),
        "since_seconds": since_seconds
    }

@app.delete("/memory/reset")
def reset_memory(db: Session = Depends(get_db)):
    """
    Clear all data — used by evaluation script between test runs.
    Also clears Qdrant collection.
    WARNING: deletes all facts, conflicts, and logs.
    """
    from vector_store import client, COLLECTION_NAME
    from qdrant_client.models import Distance, VectorParams

    db.query(ActionGateLog).delete()
    db.query(AuditLog).delete()
    db.query(ContextTrust).delete()
    db.query(Conflict).delete()
    db.query(Fact).delete()
    db.query(Agent).delete()
    db.query(HashMap).delete()
    db.commit()

    # Recreate Qdrant collection
    try:
        client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    except Exception as e:
        print(f"Qdrant reset error: {e}")

    return {"message": "All memory cleared successfully"}