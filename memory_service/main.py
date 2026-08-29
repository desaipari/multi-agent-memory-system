import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import hashlib
from datetime import datetime, timezone
from access_control import get_default_access, filter_facts_for_agent
from contextlib import asynccontextmanager
from database import engine, get_db, Base
from models import (
    Fact, Agent, Conflict, AuditLog,
    HashMap, ActionGateLog
)
from vector_store import (
    ensure_collection_exists,
    store_embedding,
    find_similar_facts
)
from confidence_scorer import (
    compute_confidence,
    should_auto_resolve,
    update_agent_trust_after_resolution,
    DEFAULT_AGENT_TRUST_SCORE
)

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
    version="4.0.0",
    description="Week 4 — Performance optimized",
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

def store_hash_mapping(db: Session, original: str, hashed: str):
    existing = db.query(HashMap).filter(
        HashMap.hash_value == hashed
    ).first()
    if not existing:
        db.add(HashMap(hash_value=hashed, original_value=original))
        db.commit()

def ensure_agent_exists(db: Session, agent_id: str):
    agent = db.query(Agent).filter(
        Agent.agent_id == agent_id
    ).first()

    if not agent:
        initial_trust = 0.50

        db.add(Agent(
            agent_id=agent_id,
            trust_score=initial_trust,
            reliability_score=initial_trust
        ))
        db.commit()

def get_agent_trust(db: Session, agent_id: str) -> float:
    agent = db.query(Agent).filter(
        Agent.agent_id == agent_id
    ).first()
    if agent:
        return agent.trust_score
    return DEFAULT_AGENT_TRUST_SCORE

def check_and_handle_contradiction(
    db: Session,
    new_fact: Fact,
    entity_hash: str,
    fact_type: str,
    new_value: str
) -> Optional[dict]:
    """
    Detect contradiction and immediately resolve or flag it.

    Stage 1 — Direct PostgreSQL check (primary, most reliable)
    Stage 2 — Qdrant semantic search (catches paraphrasing)

    Resolution:
    - If confidence gap >= 0.30: auto-resolve
    - If confidence gap < 0.30: mark both contested, flag for human
    """

    # ── Stage 1: Direct check ──────────────────────────────────
    existing_facts = db.query(Fact).filter(
        Fact.entity_hash == entity_hash,
        Fact.fact_type == fact_type,
        Fact.status == "active",
        Fact.fact_id != new_fact.fact_id
    ).all()

    for existing_fact in existing_facts:
        if (existing_fact.raw_value and
            existing_fact.raw_value.strip().lower()
                != new_value.strip().lower()):

            return _handle_conflict(
                db, existing_fact, new_fact,
                entity_hash, fact_type,
                detection_method="direct_database_check"
            )

    # ── Stage 2: Qdrant semantic search ───────────────────────
    try:
        similar = find_similar_facts(
            entity_hash=entity_hash,
            fact_type=fact_type,
            value=new_value,
            threshold=0.70
        )

        for match in similar:
            if match["fact_id"] == new_fact.fact_id:
                continue

            existing_fact = db.query(Fact).filter(
                Fact.fact_id == match["fact_id"],
                Fact.status == "active"
            ).first()

            if not existing_fact:
                continue

            if (existing_fact.raw_value and
                existing_fact.raw_value.strip().lower()
                    != new_value.strip().lower()):

                return _handle_conflict(
                    db, existing_fact, new_fact,
                    entity_hash, fact_type,
                    detection_method="semantic_search",
                    similarity_score=match["similarity_score"]
                )

    except Exception as e:
        print(f"Stage 2 error (non-fatal): {e}")

    return None


def _handle_conflict(
    db: Session,
    existing_fact: Fact,
    new_fact: Fact,
    entity_hash: str,
    fact_type: str,
    detection_method: str,
    similarity_score: float = None
) -> dict:
    """
    Given two conflicting facts, decide resolution:
    - Auto-resolve if confidence gap >= 0.30
    - Flag as contested if gap < 0.30
    """
    conf_existing = existing_fact.confidence
    conf_new = new_fact.confidence

    print(f"\nCONTRADICTION DETECTED ({detection_method})")
    print(f"  fact_type: {fact_type}")
    print(f"  Existing: {existing_fact.raw_value} "
          f"(agent:{existing_fact.agent_id}, conf:{conf_existing:.3f})")
    print(f"  New:      {new_fact.raw_value} "
          f"(agent:{new_fact.agent_id}, conf:{conf_new:.3f})")

    conflict = Conflict(
        fact_id_a=existing_fact.fact_id,
        fact_id_b=new_fact.fact_id,
        entity_hash=entity_hash,
        fact_type=fact_type,
        status="flagged"
    )
    db.add(conflict)
    db.flush()  # get conflict_id before committing

    if should_auto_resolve(conf_existing, conf_new):
        # ── Auto-resolve ───────────────────────────────────────
        gap = abs(conf_existing - conf_new)

        if conf_existing >= conf_new:
            winner = existing_fact
            loser = new_fact
        else:
            winner = new_fact
            loser = existing_fact

        # Mark loser as superseded
        loser.status = "superseded"
        loser.superseded_by = winner.fact_id

        # Winner stays active
        winner.status = "active"

        # Update conflict record
        conflict.status = "auto_resolved"
        conflict.resolved_winner = winner.fact_id
        conflict.resolution_type = "auto_resolve"
        conflict.resolution_reason = (
            f"Auto-resolved: confidence gap {gap:.3f} >= 0.30. "
            f"Winner: {winner.agent_id} ({winner.confidence:.3f}) "
            f"over {loser.agent_id} ({loser.confidence:.3f})"
        )
        conflict.resolved_at = datetime.now(timezone.utc)

        # Update agent trust scores
        update_agent_trust_after_resolution(
            winner.agent_id, loser.agent_id, db
        )

        db.add(AuditLog(
            event_type="auto_resolved",
            fact_id=winner.fact_id,
            agent_id=winner.agent_id,
            description=(
                f"Auto-resolved {fact_type}: "
                f"'{winner.raw_value}' beat "
                f"'{loser.raw_value}' "
                f"(gap: {gap:.3f})"
            )
        ))
        db.commit()

        print(f"  AUTO-RESOLVED: {winner.raw_value} wins "
              f"(gap: {gap:.3f})")

        return {
            "conflict_id": conflict.conflict_id,
            "resolution": "auto_resolved",
            "winner_value": winner.raw_value,
            "winner_agent": winner.agent_id,
            "loser_value": loser.raw_value,
            "loser_agent": loser.agent_id,
            "confidence_gap": round(gap, 4),
            "detection_method": detection_method
        }

    else:
        # ── Flag as contested — human review needed ────────────
        gap = abs(conf_existing - conf_new)

        existing_fact.status = "contested"
        existing_fact.conflict_id = conflict.conflict_id
        new_fact.status = "contested"
        new_fact.conflict_id = conflict.conflict_id

        conflict.status = "flagged"
        conflict.resolution_type = "human_review"
        conflict.resolution_reason = (
            f"Flagged for human review: gap {gap:.3f} < 0.30. "
            f"Too close to auto-resolve safely."
        )

        db.add(AuditLog(
            event_type="conflict_detected",
            fact_id=new_fact.fact_id,
            agent_id=new_fact.agent_id,
            description=(
                f"Contested {fact_type}: "
                f"'{existing_fact.raw_value}' ({conf_existing:.3f}) "
                f"vs '{new_fact.raw_value}' ({conf_new:.3f}). "
                f"Gap {gap:.3f} too small to auto-resolve."
            )
        ))
        db.commit()

        print(f"  CONTESTED: gap {gap:.3f} too small — "
              f"flagged for human review")

        return {
            "conflict_id": conflict.conflict_id,
            "resolution": "contested",
            "value_a": existing_fact.raw_value,
            "agent_a": existing_fact.agent_id,
            "confidence_a": conf_existing,
            "value_b": new_fact.raw_value,
            "agent_b": new_fact.agent_id,
            "confidence_b": conf_new,
            "confidence_gap": round(gap, 4),
            "detection_method": detection_method,
            "message": "Flagged for human review"
        }


# ── Request Models ─────────────────────────────────────────────

class MemoryWriteRequest(BaseModel):
    entity: str
    fact_type: str
    value: str
    agent_id: str
    extraction_type: Optional[str] = "direct"
    confidence: Optional[float] = None  # None = auto-compute
    source_file: Optional[str] = None

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
        "version": "3.0.0",
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
    Store a new incident fact.

    Full flow:
    1. Ensure agent record exists
    2. Hash entity and value
    3. Check for corroboration (same entity+type+value, different agent)
    4. Compute confidence score
    5. Write to PostgreSQL
    6. Store embedding in Qdrant
    7. Check for contradictions
    8. Log event
    """
    ensure_agent_exists(db, request.agent_id)

    entity_hash = hash_value(request.entity)
    value_hash = hash_value(request.value)

    store_hash_mapping(db, request.entity, entity_hash)
    store_hash_mapping(db, request.value, value_hash)

    # ── Check for corroboration ────────────────────────────────
    # Same entity + same fact_type + same value + different agent
    # = corroboration, not contradiction
    existing_same_value = db.query(Fact).filter(
        Fact.entity_hash == entity_hash,
        Fact.fact_type == request.fact_type.lower().strip(),
        Fact.raw_value == request.value,
        Fact.agent_id != request.agent_id,
        Fact.status == "active"
    ).first()

    corroboration_count = 1
    if existing_same_value:
        # This is a corroboration — boost the existing fact's confidence
        corroboration_count = (
            existing_same_value.corroboration_count or 1
        ) + 1
        existing_same_value.corroboration_count = corroboration_count

        # Recompute confidence with higher corroboration
        agent_trust = get_agent_trust(db, existing_same_value.agent_id)
        new_conf = compute_confidence(
        agent_id=existing_same_value.agent_id,
        fact_type=existing_same_value.fact_type,  # ADD THIS
        extraction_type=existing_same_value.extraction_type,
        corroboration_count=corroboration_count,
        timestamp=existing_same_value.timestamp,
        db_trust_score=agent_trust
)
        existing_same_value.confidence = new_conf

        db.add(AuditLog(
            event_type="corroboration",
            fact_id=existing_same_value.fact_id,
            agent_id=request.agent_id,
            description=(
                f"{request.agent_id} corroborated "
                f"{request.fact_type}={request.value} "
                f"for {request.entity}. "
                f"Confidence updated to {new_conf:.3f}"
            )
        ))
        db.commit()

        print(f"\nCORROBORATION: {request.fact_type}={request.value} "
              f"confirmed by {request.agent_id}. "
              f"Confidence: {new_conf:.3f}")

    # ── Compute confidence for new fact ───────────────────────
    if request.confidence is not None:
        # Agent provided explicit confidence — use it
        computed_confidence = request.confidence
    else:
        # Auto-compute from formula
        agent_trust = get_agent_trust(db, request.agent_id)
        computed_confidence = compute_confidence(
    agent_id=request.agent_id,
    fact_type=request.fact_type.lower().strip(),
    extraction_type=request.extraction_type or "direct",
    corroboration_count=corroboration_count,
    db_trust_score=agent_trust
)

    # ── Write fact to PostgreSQL ───────────────────────────────
    # Add readable_by when creating the fact
    # ── Write fact to PostgreSQL ───────────────────────────────
    # Add readable_by when creating the fact
    fact = Fact(
        entity_hash=entity_hash,
        fact_type=request.fact_type.lower().strip(),
        value_hash=value_hash,
        raw_value=request.value,
        agent_id=request.agent_id,
        confidence=computed_confidence,
        status="active",
        extraction_type=request.extraction_type or "direct",
        source_file=request.source_file,
        corroboration_count=corroboration_count,
        readable_by=get_default_access(
            request.fact_type.lower().strip()
        )
    )
    db.add(fact)

    # Update agent write count
    agent = db.query(Agent).filter(
        Agent.agent_id == request.agent_id
    ).first()
    if agent:
        agent.total_writes = (agent.total_writes or 0) + 1
        db.commit()

    # ── Store embedding in Qdrant ──────────────────────────────
    store_embedding(
        fact_id=fact.fact_id,
        entity_hash=entity_hash,
        fact_type=request.fact_type.lower().strip(),
        value=request.value,
        agent_id=request.agent_id
    )

    # ── Check for contradictions ───────────────────────────────
    # Skip if this was a corroboration
    contradiction_result = None
    if not existing_same_value:
        contradiction_result = check_and_handle_contradiction(
            db=db,
            new_fact=fact,
            entity_hash=entity_hash,
            fact_type=request.fact_type.lower().strip(),
            new_value=request.value
        )

    # ── Log write event ────────────────────────────────────────
    db.add(AuditLog(
        event_type="write",
        fact_id=fact.fact_id,
        agent_id=request.agent_id,
        description=(
            f"{request.agent_id} wrote "
            f"{request.fact_type}={request.value} "
            f"for {request.entity} "
            f"(confidence:{computed_confidence:.3f}) "
            f"from {request.source_file or 'text input'}"
        )
    ))
    db.commit()

    return {
    "fact_id": fact.fact_id,
    "entity_hash": entity_hash,
    "fact_type": fact.fact_type,
    "value_hash": value_hash,
    "agent_id": request.agent_id,
    "confidence": computed_confidence,
    "status": fact.status,
    "corroboration": existing_same_value is not None,
    "contradiction_detected": contradiction_result is not None,
    "contradiction": contradiction_result,
    "message": (
        "Corroboration — confidence updated"
        if existing_same_value
        else "Contradiction detected — flagged for human review"
        if contradiction_result and
           contradiction_result.get("resolution") == "contested"
        else "Contradiction detected and auto-resolved"
        if contradiction_result and
           contradiction_result.get("resolution") == "auto_resolved"
        else "Fact stored successfully"
    )
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
            Fact.fact_type == fact_type.lower().strip()
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
            "timestamp": fact.timestamp.isoformat()
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
            "timestamp": fact.timestamp.isoformat()
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
    Human or Coordinator Agent resolves a contested conflict.
    Called from the dashboard Review button or Coordinator Agent.
    """
    conflict = db.query(Conflict).filter(
        Conflict.conflict_id == request.conflict_id
    ).first()

    if not conflict:
        raise HTTPException(
            status_code=404,
            detail=f"Conflict not found: {request.conflict_id}"
        )

    if conflict.status not in ["flagged"]:
        raise HTTPException(
            status_code=400,
            detail=f"Conflict already resolved: {conflict.status}"
        )

    # Find winner and loser facts
    if request.winning_fact_id == conflict.fact_id_a:
        winner_id = conflict.fact_id_a
        loser_id = conflict.fact_id_b
    elif request.winning_fact_id == conflict.fact_id_b:
        winner_id = conflict.fact_id_b
        loser_id = conflict.fact_id_a
    else:
        raise HTTPException(
            status_code=400,
            detail="winning_fact_id must be one of the two conflicting facts"
        )

    winner_fact = db.query(Fact).filter(
        Fact.fact_id == winner_id
    ).first()
    loser_fact = db.query(Fact).filter(
        Fact.fact_id == loser_id
    ).first()

    if not winner_fact or not loser_fact:
        raise HTTPException(
            status_code=404,
            detail="Could not find conflicting facts"
        )

    # Apply resolution
    winner_fact.status = "active"
    loser_fact.status = "superseded"
    loser_fact.superseded_by = winner_fact.fact_id

    # Clear contested flags
    winner_fact.conflict_id = None
    loser_fact.conflict_id = None

    conflict.status = "human_resolved"
    conflict.resolved_winner = winner_id
    conflict.resolution_type = "human_review"
    conflict.resolution_reason = (
        request.reason or
        f"Resolved by {request.resolved_by}"
    )
    conflict.resolved_at = datetime.now(timezone.utc)

    # Update agent trust scores
    update_agent_trust_after_resolution(
        winner_fact.agent_id,
        loser_fact.agent_id,
        db
    )

    db.add(AuditLog(
        event_type="human_resolved",
        fact_id=winner_fact.fact_id,
        agent_id=request.resolved_by,
        description=(
            f"Human resolved {conflict.fact_type}: "
            f"'{winner_fact.raw_value}' "
            f"({winner_fact.agent_id}) chosen over "
            f"'{loser_fact.raw_value}' "
            f"({loser_fact.agent_id}). "
            f"Reason: {request.reason or 'Not specified'}"
        )
    ))
    db.commit()

    return {
        "conflict_id": conflict.conflict_id,
        "status": "human_resolved",
        "winner_value": winner_fact.raw_value,
        "winner_agent": winner_fact.agent_id,
        "loser_value": loser_fact.raw_value,
        "loser_agent": loser_fact.agent_id,
        "message": "Conflict resolved successfully"
    }


@app.post("/memory/check_action")
def check_action_gate(
    request: ActionCheckRequest,
    db: Session = Depends(get_db)
):
    """
    Check if an agent can proceed with an action based on
    the confidence and status of the fact it needs.

    Returns allowed=True or allowed=False with reason.
    Called by agents before taking any important action.
    """
    entity_hash = hash_value(request.entity)

    fact = db.query(Fact).filter(
    Fact.entity_hash == entity_hash,
    Fact.fact_type == request.fact_type.lower().strip(),
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
        # Log the block
        db.add(ActionGateLog(
            agent_id=request.agent_id,
            entity=request.entity,
            fact_type=request.fact_type,
            action_attempted=request.action_attempted,
            blocked_reason="Fact is contested — conflict unresolved",
            confidence_at_block=fact.confidence,
            conflict_id=fact.conflict_id
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
        db.add(ActionGateLog(
            agent_id=request.agent_id,
            entity=request.entity,
            fact_type=request.fact_type,
            action_attempted=request.action_attempted,
            blocked_reason=(
                f"Confidence {fact.confidence:.3f} "
                f"below threshold {threshold}"
            ),
            confidence_at_block=fact.confidence
        ))
        db.commit()

        return {
            "allowed": False,
            "reason": (
                f"Confidence {fact.confidence:.3f} "
                f"below required threshold {threshold}"
            ),
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
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Full audit trail with optional filters.
    
    event_type options:
      write, corroboration, conflict_detected,
      auto_resolved, human_resolved, action_blocked
    
    Examples:
      GET /memory/audit
      GET /memory/audit?agent_id=billing_agent
      GET /memory/audit?event_type=conflict_detected
      GET /memory/audit?agent_id=intake_agent&limit=20
    """
    query = db.query(AuditLog).order_by(
        AuditLog.timestamp.desc()
    )

    if agent_id:
        query = query.filter(AuditLog.agent_id == agent_id)

    if event_type:
        query = query.filter(AuditLog.event_type == event_type)

    # Entity filter requires joining through facts
    # since audit log stores fact_id not entity directly
    if entity:
        entity_hash = hash_value(entity)
        # Get all fact_ids for this entity
        fact_ids = [
            f.fact_id for f in db.query(Fact).filter(
                Fact.entity_hash == entity_hash
            ).all()
        ]
        if fact_ids:
            query = query.filter(
                AuditLog.fact_id.in_(fact_ids)
            )
        else:
            return {"logs": [], "total": 0, "filters": {
                "entity": entity, "agent_id": agent_id,
                "event_type": event_type
            }}

    logs = query.limit(limit).all()

    return {
        "logs": [
            {
                "log_id": log.log_id,
                "event_type": log.event_type,
                "fact_id": log.fact_id,
                "agent_id": log.agent_id,
                "description": log.description,
                "timestamp": log.timestamp.isoformat()
            }
            for log in logs
        ],
        "total": len(logs),
        "filters": {
            "entity": entity,
            "agent_id": agent_id,
            "event_type": event_type
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