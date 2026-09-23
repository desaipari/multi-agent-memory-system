from dataclasses import dataclass
from typing import Protocol

from .config import (
    PRIOR_STRENGTH,
    get_expert_prior,
)


@dataclass(frozen=True)
class TrustStats:
    successes: int = 0
    failures: int = 0

    @property
    def n(self):
        return self.successes + self.failures


class TrustProvider(Protocol):
    def get_trust(
        self,
        agent_id,
        fact_type,
        prior_strength=PRIOR_STRENGTH
    ):
        ...


def posterior_mean(
    prior,
    stats,
    prior_strength=PRIOR_STRENGTH
):
    alpha_0 = (
        prior_strength * prior
    )

    beta_0 = (
        prior_strength
        * (1.0 - prior)
    )

    denominator = (
        alpha_0
        + beta_0
        + stats.successes
        + stats.failures
    )

    if denominator == 0:
        return prior

    return (
        alpha_0
        + stats.successes
    ) / denominator


def contextual_trust(
    provider,
    agent_id,
    fact_type,
    extraction_type=None,
    prior_strength=PRIOR_STRENGTH
):
    if hasattr(
        provider,
        "get_trust"
    ):
        return provider.get_trust(
            agent_id=agent_id,
            fact_type=fact_type,
            prior_strength=prior_strength,
        )

    prior = get_expert_prior(
        agent_id,
        fact_type
    )

    stats = provider.get_stats(
        agent_id,
        fact_type
    )

    probability = posterior_mean(
        prior=prior,
        stats=stats,
        prior_strength=prior_strength
    )

    return {
        "probability": probability,
        "evidence_count": stats.n,
        "prior": prior,
        "successes": stats.successes,
        "failures": stats.failures,
    }