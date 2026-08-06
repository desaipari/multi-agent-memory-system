from sqlalchemy import Column, String, Float, DateTime, Text, Integer
from sqlalchemy.sql import func
from database import Base
import uuid

class Fact(Base):
    __tablename__ = "facts"
    
    # Every stored incident fact gets a unique ID
    fact_id = Column(String, primary_key=True, 
                     default=lambda: str(uuid.uuid4()))
    
    # entity_hash = SHA-256 hash of incident ID e.g. "INC0000001"
    # Raw incident ID never stored here — only the hash
    entity_hash = Column(String, nullable=False, index=True)
    
    # fact_type = what kind of fact: priority, state, assignment_group etc
    fact_type = Column(String, nullable=False)
    
    # value_hash = SHA-256 hash of the actual value e.g. "2-High"
    value_hash = Column(String, nullable=False)
    
    # raw_value = plain text value, stored locally for dashboard display
    # This never leaves the local system — dashboard reads from here
    raw_value = Column(String, nullable=True)
    
    # which of the four agents wrote this fact
    agent_id = Column(String, nullable=False)
    
    # confidence score 0.0 to 1.0 — starts at 0.5, scoring comes Week 3
    confidence = Column(Float, default=0.5)
    
    timestamp = Column(DateTime, server_default=func.now())
    
    # active = current truth
    # superseded = overruled by higher confidence fact, kept in history
    # contested = conflict detected, neither resolved yet
    status = Column(String, default="active")
    
    # direct = agent stated this explicitly from source document
    # inferred = agent interpreted or derived this
    extraction_type = Column(String, default="direct")
    
    # which source file this came from
    source_file = Column(String, nullable=True)


class Agent(Base):
    __tablename__ = "agents"
    
    agent_id = Column(String, primary_key=True)
    trust_score = Column(Float, default=0.5)
    total_writes = Column(Integer, default=0)
    correct_writes = Column(Integer, default=0)
    overturned_writes = Column(Integer, default=0)


class Conflict(Base):
    __tablename__ = "conflicts"
    
    conflict_id = Column(String, primary_key=True, 
                         default=lambda: str(uuid.uuid4()))
    fact_id_a = Column(String, nullable=False)
    fact_id_b = Column(String, nullable=False)
    entity_hash = Column(String, nullable=False)
    fact_type = Column(String, nullable=False)
    
    # flagged = detected, awaiting resolution
    # auto_resolved = system resolved it based on confidence
    # human_resolved = coordinator agent or human resolved via dashboard
    status = Column(String, default="flagged")
    resolution = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"
    
    log_id = Column(String, primary_key=True, 
                    default=lambda: str(uuid.uuid4()))
    
    # write, conflict_detected, auto_resolved, 
    # human_resolved, action_blocked
    event_type = Column(String, nullable=False)
    fact_id = Column(String, nullable=True)
    agent_id = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())


class HashMap(Base):
    __tablename__ = "hash_map"
    
    # Maps SHA-256 hash back to original plain text value
    # This is the local de-anonymisation table
    hash_value = Column(String, primary_key=True)
    original_value = Column(String, nullable=False)