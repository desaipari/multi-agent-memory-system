from sqlalchemy import Column, String, Float, DateTime, Text, Integer, Boolean
from sqlalchemy.sql import func
from database import Base
import uuid

class Fact(Base):
    __tablename__ = "facts"

    fact_id = Column(String, primary_key=True,
                     default=lambda: str(uuid.uuid4()))
    entity_hash = Column(String, nullable=False, index=True)
    fact_type = Column(String, nullable=False)
    value_hash = Column(String, nullable=False)
    raw_value = Column(String, nullable=True)
    agent_id = Column(String, nullable=False)
    confidence = Column(Float, default=0.5)
    timestamp = Column(DateTime, server_default=func.now())

    # active, superseded, contested
    status = Column(String, default="active")

    # direct, inferred
    extraction_type = Column(String, default="direct")

    source_file = Column(String, nullable=True)

    # Week 3 additions
    # tracks how many independent agents confirmed this fact
    corroboration_count = Column(Integer, default=1)

    # if superseded, which fact replaced it
    superseded_by = Column(String, nullable=True)

    # if contested, which conflict caused it
    conflict_id = Column(String, nullable=True)


class Agent(Base):
    __tablename__ = "agents"

    agent_id = Column(String, primary_key=True)
    trust_score = Column(Float, default=0.5)
    total_writes = Column(Integer, default=0)
    correct_writes = Column(Integer, default=0)
    overturned_writes = Column(Integer, default=0)

    # Week 3 additions
    # recalculated after each conflict resolution
    reliability_score = Column(Float, default=0.5)


class Conflict(Base):
    __tablename__ = "conflicts"

    conflict_id = Column(String, primary_key=True,
                         default=lambda: str(uuid.uuid4()))
    fact_id_a = Column(String, nullable=False)
    fact_id_b = Column(String, nullable=False)
    entity_hash = Column(String, nullable=False)
    fact_type = Column(String, nullable=False)

    # flagged, auto_resolved, human_resolved
    status = Column(String, default="flagged")

    # which fact_id won the resolution
    resolved_winner = Column(String, nullable=True)

    # auto_resolve or human_review
    resolution_type = Column(String, nullable=True)

    resolution_reason = Column(Text, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    log_id = Column(String, primary_key=True,
                    default=lambda: str(uuid.uuid4()))

    # write, conflict_detected, auto_resolved,
    # human_resolved, action_blocked, corroboration
    event_type = Column(String, nullable=False)
    fact_id = Column(String, nullable=True)
    agent_id = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())


class HashMap(Base):
    __tablename__ = "hash_map"

    hash_value = Column(String, primary_key=True)
    original_value = Column(String, nullable=False)


class ActionGateLog(Base):
    __tablename__ = "action_gate_log"

    gate_id = Column(String, primary_key=True,
                     default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, nullable=False)
    entity = Column(String, nullable=False)
    fact_type = Column(String, nullable=False)
    action_attempted = Column(String, nullable=True)
    blocked_reason = Column(Text, nullable=True)
    confidence_at_block = Column(Float, nullable=True)
    conflict_id = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())