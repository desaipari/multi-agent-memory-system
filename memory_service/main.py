from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import hashlib
from datetime import datetime

from database import engine, get_db, Base
from models import Fact, Agent, AuditLog, HashMap

# Create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Multi-Agent Incident Memory Service",
    version="1.0.0",
    description="Shared memory layer for IT incident management agents"
)

# Allow React dashboard to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Helper Functions ─────────────────────────────────────────

def hash_value(text: str) -> str:
    """Convert any string to SHA-256 hash. Sensitive data never stored raw."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()

def store_hash_mapping(db: Session, original: str, hashed: str):
    """Store hash-to-original mapping for de-anonymisation on retrieval."""
    existing = db.query(HashMap).filter(
        HashMap.hash_value == hashed
    ).first()
    if not existing:
        mapping = HashMap(hash_value=hashed, original_value=original)
        db.add(mapping)
        db.commit()

def ensure_agent_exists(db: Session, agent_id: str):
    """Create agent record if this agent has not written before."""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        agent = Agent(agent_id=agent_id)
        db.add(agent)
        db.commit()

# ─── Request Models ───────────────────────────────────────────

class MemoryWriteRequest(BaseModel):
    # Plain text incident ID e.g. "INC0000001"
    entity: str
    
    # What kind of fact: priority, state, assignment_group,
    # category, opened_date, resolved_by
    fact_type: str
    
    # Plain text value e.g. "2-High", "Resolved", "Network Team"
    value: str
    
    # Which agent: intake_agent, delivery_agent, 
    # billing_agent, coordinator_agent
    agent_id: str
    
    extraction_type: Optional[str] = "direct"
    confidence: Optional[float] = 0.5
    
    # Which CSV file this came from
    source_file: Optional[str] = None

# ─── Endpoints ────────────────────────────────────────────────

@app.get("/memory/health")
def health_check():
    """Verify the memory service is running correctly."""
    return {
        "status": "running",
        "service": "Multi-Agent Incident Memory Service",
        "version": "1.0.0",
        "scenario": "IT Incident Management",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/memory/write")
def write_memory(
    request: MemoryWriteRequest,
    db: Session = Depends(get_db)
):
    """
    Store a new incident fact from an agent.
    
    Flow:
    1. Ensure agent record exists
    2. Hash the incident ID and value (sensitive data protection)
    3. Store hash-to-original mappings locally
    4. Create fact record with hashed values
    5. Log write event in audit trail
    6. Return fact ID and hashes (not plain text values)
    """
    ensure_agent_exists(db, request.agent_id)

    # Hash sensitive values before storage
    entity_hash = hash_value(request.entity)
    value_hash = hash_value(request.value)

    # Store mappings so dashboard can de-hash for display
    store_hash_mapping(db, request.entity, entity_hash)
    store_hash_mapping(db, request.value, value_hash)

    # Create the fact record
    fact = Fact(
        entity_hash=entity_hash,
        fact_type=request.fact_type.lower().strip(),
        value_hash=value_hash,
        raw_value=request.value,      # kept locally for dashboard
        agent_id=request.agent_id,
        confidence=request.confidence,
        status="active",
        extraction_type=request.extraction_type,
        source_file=request.source_file
    )
    db.add(fact)

    # Write to audit trail
    log = AuditLog(
        event_type="write",
        fact_id=fact.fact_id,
        agent_id=request.agent_id,
        description=(
            f"{request.agent_id} wrote "
            f"{request.fact_type}={request.value} "
            f"for incident {request.entity} "
            f"from {request.source_file or 'text input'}"
        )
    )
    db.add(log)
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
        "source_file": request.source_file,
        "message": "Fact stored successfully"
    }


@app.get("/memory/read")
def read_memory(
    entity: str,
    fact_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve active facts for a given incident.
    De-hashes values using local hash_map table.
    Returns human-readable incident data for dashboard display.
    """
    entity_hash = hash_value(entity)

    query = db.query(Fact).filter(
        Fact.entity_hash == entity_hash,
        Fact.status == "active"
    )

    if fact_type:
        query = query.filter(
            Fact.fact_type == fact_type.lower().strip()
        )

    facts = query.all()

    if not facts:
        raise HTTPException(
            status_code=404,
            detail=f"No active facts found for incident: {entity}"
        )

    results = []
    for fact in facts:
        results.append({
            "fact_id": fact.fact_id,
            "entity": entity,
            "entity_hash": fact.entity_hash,
            "fact_type": fact.fact_type,
            "value": fact.raw_value,         # human readable
            "value_hash": fact.value_hash,   # hash shown for transparency
            "agent_id": fact.agent_id,
            "confidence": fact.confidence,
            "status": fact.status,
            "extraction_type": fact.extraction_type,
            "source_file": fact.source_file,
            "timestamp": fact.timestamp.isoformat()
        })

    return {
        "entity": entity,
        "facts": results,
        "count": len(results)
    }


@app.get("/memory/all")
def get_all_facts(db: Session = Depends(get_db)):
    """
    Return all stored facts regardless of entity.
    Used by the dashboard to populate the memory state table.
    Returns facts with human-readable values via raw_value field.
    """
    facts = db.query(Fact).order_by(Fact.timestamp.desc()).all()

    results = []
    for fact in facts:
        # De-hash entity for display
        entity_mapping = db.query(HashMap).filter(
            HashMap.hash_value == fact.entity_hash
        ).first()
        entity_readable = entity_mapping.original_value if entity_mapping else fact.entity_hash

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
            "timestamp": fact.timestamp.isoformat()
        })

    return {"facts": results, "total": len(results)}