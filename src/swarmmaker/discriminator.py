"""Discriminators and voting helpers."""
from __future__ import annotations

from typing import Sequence

from .schemas import AtomicSolution, DecompositionProposal
from .voting import AheadByKVoter, VoteOutcome


class DecompositionDiscriminator:
    """Votes among decomposition proposals using ahead-by-K."""

    def __init__(self, ahead_by: int) -> None:
        self.voter = AheadByKVoter(ahead_by)

    def select(self, proposals: Sequence[DecompositionProposal]) -> VoteOutcome[DecompositionProposal]:
        return self.voter.select(proposals, lambda proposal: proposal.signature)


class SolutionDiscriminator:
    """Votes among atomic solutions."""

    def __init__(self, ahead_by: int) -> None:
        self.voter = AheadByKVoter(ahead_by)

    def select(self, solutions: Sequence[AtomicSolution]) -> VoteOutcome[AtomicSolution]:
        return self.voter.select(solutions, lambda solution: solution.signature)
