from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Observation:
    fact_id: str
    agent_id: str
    fact_type: str
    value: str
    extraction_type: str
    observed_at: Optional[datetime]


@dataclass(frozen=True)
class TrustEstimate:
    probability: float
    evidence_count: int
    level: str