from .config import (
    PRIOR_STRENGTH,
    get_expert_prior,
)

from .trust import (
    TrustStats,
    posterior_mean,
)


class FixedTrustProvider:
    def get_trust(
        self,
        agent_id,
        fact_type,
        prior_strength=PRIOR_STRENGTH
    ):
        prior = get_expert_prior(
            agent_id,
            fact_type
        )

        return {
            "probability": prior,
            "evidence_count": 0,
            "prior": prior,
            "successes": 0,
            "failures": 0,
        }


class InMemoryTrustProvider:
    def __init__(self):
        self.stats = {}

    def get_stats(
        self,
        agent_id,
        fact_type,
        extraction_type=None
    ):
        key = (
            agent_id,
            fact_type
        )

        return self.stats.get(
            key,
            TrustStats()
        )

    def get_trust(
        self,
        agent_id,
        fact_type,
        prior_strength=PRIOR_STRENGTH
    ):
        prior = get_expert_prior(
            agent_id,
            fact_type
        )

        stats = self.get_stats(
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

    def set_stats(
        self,
        agent_id,
        fact_type,
        extraction_type=None,
        successes=0,
        failures=0
    ):
        key = (
            agent_id,
            fact_type
        )

        self.stats[key] = TrustStats(
            successes=successes,
            failures=failures
        )

    def record_verified_outcome(
        self,
        agent_id,
        fact_type,
        extraction_type=None,
        correct=None
    ):
        if correct is None:
            if isinstance(
                extraction_type,
                bool
            ):
                correct = extraction_type
            else:
                raise ValueError(
                    "correct must be True or False"
                )

        key = (
            agent_id,
            fact_type
        )

        current = self.stats.get(
            key,
            TrustStats()
        )

        if correct:
            updated = TrustStats(
                successes=(
                    current.successes + 1
                ),
                failures=current.failures,
            )
        else:
            updated = TrustStats(
                successes=current.successes,
                failures=(
                    current.failures + 1
                ),
            )

        self.stats[key] = updated

    def snapshot(self):
        return dict(self.stats)


class OracleTrustProvider:
    def __init__(
        self,
        probabilities
    ):
        self.probabilities = probabilities

    def _lookup(
        self,
        agent_id,
        fact_type
    ):
        agent_values = (
            self.probabilities.get(
                agent_id,
                {}
            )
        )

        if fact_type in agent_values:
            return agent_values[
                fact_type
            ]

        return get_expert_prior(
            agent_id,
            fact_type
        )

    def get_trust(
        self,
        agent_id,
        fact_type,
        prior_strength=PRIOR_STRENGTH
    ):
        probability = self._lookup(
            agent_id,
            fact_type
        )

        return {
            "probability": probability,
            "evidence_count": None,
            "prior": get_expert_prior(
                agent_id,
                fact_type
            ),
            "successes": None,
            "failures": None,
        }