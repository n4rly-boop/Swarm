"""Code-only verifier for MAKER outputs."""
from __future__ import annotations

from typing import Tuple

from .schemas import AtomicSolution, TaskState


class StateVerifier:
    """Applies simple structural checks and updates the shared task state."""

    def __init__(self) -> None:
        self._seen_signatures: set[str] = set()

    def verify_solution(self, solution: AtomicSolution) -> Tuple[bool, str]:
        if not solution.solution.strip():
            return False, "solution must not be empty"
        if not solution.work_shown.strip():
            return False, "work_shown must document intermediate steps"
        if solution.signature in self._seen_signatures:
            return False, "duplicate solution signature"
        return True, "validated"

    def commit_solution(self, problem: str, solution: AtomicSolution, state: TaskState) -> None:
        self._seen_signatures.add(solution.signature)
        state.solved_subproblems[problem] = solution.solution
        state.current_problem = problem

    def compose(self, compose_fn: str, solution_a: str, solution_b: str) -> str:
        return (
            f"{compose_fn.strip()}\n\n"
            f"Subproblem A result:\n{solution_a.strip()}\n\n"
            f"Subproblem B result:\n{solution_b.strip()}"
        )
