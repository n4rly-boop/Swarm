"""Verification and quality gates for SwarmMaker."""
import math
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

try:
    import sympy as sp
except Exception:
    sp = None

from .schemas import (
    AtomicSolution,
    DecompositionProposal,
    FinalAnswer,
    ProblemFact,
    SupportPoint,
    TaskState,
    ThresholdConfig,
)

if TYPE_CHECKING:
    from .adapters.base import BaseDomainAdapter


class StateVerifier:
    """Applies structural checks, quality gates, and progress scoring."""

    def __init__(self, thresholds: Optional[ThresholdConfig] = None) -> None:
        """Initialize verifier with thresholds.

        Args:
            thresholds: Centralized threshold config. Uses defaults if None.
        """
        self.thresholds = thresholds or ThresholdConfig()
        self._seen_signatures: set[str] = set()
        self._fact_keys: List[str] = []

    # Atomic solution checks -------------------------------------------------
    def verify_solution(self, solution: AtomicSolution) -> Tuple[bool, str]:
        """Verify an atomic solution meets quality requirements.

        Note: This does NOT check for duplicates - duplicates in the candidate
        pool are essential for voting consensus. Duplicate tracking only happens
        when committing the final selected solution.

        Args:
            solution: Solution to verify.

        Returns:
            Tuple of (is_valid, reason).
        """
        if not solution.solution.strip():
            return False, "solution must not be empty"

        if not solution.work_shown.strip():
            return False, "work_shown must document intermediate steps"

        # Truncate work_shown if too long
        if len(solution.work_shown.strip()) > self.thresholds.max_work_shown_chars:
            solution.work_shown = self._truncate(
                solution.work_shown.strip(),
                self.thresholds.max_work_shown_chars,
            )

        return True, "validated"

    def commit_solution(
        self,
        problem: str,
        solution: AtomicSolution,
        state: TaskState,
        *,
        depth: int,
        step_id: int,
    ) -> None:
        """Commit a validated solution to the task state.

        Args:
            problem: Problem that was solved.
            solution: Validated solution.
            state: Task state to update.
            depth: Current depth.
            step_id: Current step ID.
        """
        self._seen_signatures.add(solution.signature)
        fact_key = f"step-{step_id}"

        fact = ProblemFact(
            problem=problem.strip(),
            solution=solution.solution.strip(),
            work_shown=self._truncate(
                solution.work_shown.strip(),
                self.thresholds.max_work_shown_chars,
            ),
            confidence=solution.confidence,
            depth=depth,
        )

        state.solved_subproblems[problem] = fact.solution
        state.current_problem = problem
        state.draft_answer = fact.solution
        state.facts[fact_key] = fact
        self._fact_keys.append(fact_key)

        # FIFO eviction for facts
        if len(self._fact_keys) > self.thresholds.max_facts:
            oldest = self._fact_keys.pop(0)
            state.facts.pop(oldest, None)

    # Decomposition gates ----------------------------------------------------
    def validate_decomposition(
        self,
        parent: str,
        proposal: DecompositionProposal,
    ) -> Tuple[bool, str]:
        """Validate that a decomposition is a real reduction, not a tautology.

        Args:
            parent: Parent problem being decomposed.
            proposal: Proposed decomposition.

        Returns:
            Tuple of (is_valid, reason).
        """
        # If marked atomic, allow it through - orchestrator will handle it
        if proposal.is_atomic:
            return True, "atomic proposal"

        parent_norm = self._normalize(parent)
        sub_a = self._normalize(proposal.subproblem_a)
        sub_b = self._normalize(proposal.subproblem_b)

        # Basic structural checks
        if sub_a == parent_norm or sub_b == parent_norm:
            return False, "subproblem duplicates parent"

        if sub_a == sub_b:
            return False, "subproblems must differ"

        if self._is_tautology(proposal.subproblem_a) or self._is_tautology(proposal.subproblem_b):
            return False, "tautological subproblem rejected"

        # Check subproblems are substantial (not just restating or trivial)
        if len(proposal.subproblem_a.strip()) < 10 or len(proposal.subproblem_b.strip()) < 10:
            return False, "subproblems too short"

        return True, "ok"

    def score_decomposition(
        self,
        proposal: DecompositionProposal,
        votes: Dict[str, int],
    ) -> float:
        """Score a decomposition proposal based on votes and structure.

        Args:
            proposal: Proposal to score.
            votes: Vote counts by signature.

        Returns:
            Score value (higher is better).
        """
        vote_weight = votes.get(proposal.signature, 0)
        # Prefer non-atomic (actual decompositions) over atomic fallbacks
        structure_weight = 0.5 if not proposal.is_atomic else 0.0
        # Penalize overly verbose proposals
        total_len = len(proposal.subproblem_a) + len(proposal.subproblem_b) + len(proposal.compose_fn)
        verbosity_penalty = total_len / 900.0
        return vote_weight + structure_weight - verbosity_penalty

    # Atomic selection scoring -----------------------------------------------
    def score_solution(
        self,
        solution: AtomicSolution,
        votes: Dict[str, int],
    ) -> float:
        """Score an atomic solution for selection.

        Args:
            solution: Solution to score.
            votes: Vote counts by signature.

        Returns:
            Score value (higher is better).
        """
        vote_weight = votes.get(solution.signature, 0)
        novelty = 0.8 if solution.signature not in self._seen_signatures else 0.0
        verbosity_penalty = len(solution.work_shown) / 600.0
        return vote_weight + novelty + solution.confidence - verbosity_penalty

    # Utility helpers --------------------------------------------------------
    def compose(self, compose_fn: str, solution_a: str, solution_b: str) -> str:
        """Compose two sub-solutions using the compose function.

        Args:
            compose_fn: Instructions for composition.
            solution_a: First sub-solution.
            solution_b: Second sub-solution.

        Returns:
            Composed result string.
        """
        return (
            f"{compose_fn.strip()}\n\n"
            f"Subproblem A result:\n{solution_a.strip()}\n\n"
            f"Subproblem B result:\n{solution_b.strip()}"
        )

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _is_tautology(self, text: str) -> bool:
        """Detect tautological subproblems that don't reduce the problem."""
        lowered = text.lower()
        # Pattern: "solve y = ... for y" - asking to solve for what's already given
        taut_pattern = re.compile(r"solve\s+([a-z]\w*)\s*=\s*[^=]+?\s+for\s+\1\b")
        if taut_pattern.search(lowered):
            return True
        # Rewrite/restate don't reduce complexity
        if lowered.startswith("rewrite ") or lowered.startswith("restate "):
            return True
        return False


class GlobalVerifier:
    """Global validator for basic sanity checks and optional sympy validation.

    Note: Completeness checking is done by CompletenessChecker (LLM-based).
    This class handles:
    - Basic sanity checks (answer not empty)
    - Mathematical validation using sympy (if points are provided)
    - Domain adapter verification (if adapter provided)
    """

    def __init__(
        self,
        thresholds: Optional[ThresholdConfig] = None,
        adapter: Optional["BaseDomainAdapter"] = None,
    ) -> None:
        """Initialize verifier with thresholds and optional adapter.

        Args:
            thresholds: Centralized threshold config. Uses defaults if None.
            adapter: Optional domain adapter for specialized verification.
        """
        self.thresholds = thresholds or ThresholdConfig()
        self.adapter = adapter

    def verify(
        self,
        task: str,
        payload: FinalAnswer,
        state: TaskState,
    ) -> Tuple[bool, str]:
        """Perform basic sanity checks and optional sympy validation.

        Args:
            task: Original task description.
            payload: Final answer payload.
            state: Current task state.

        Returns:
            Tuple of (is_valid, reason).
        """
        answer = payload.answer.strip()
        if not answer or len(answer) < 3:
            return False, "final answer is empty"

        # Domain adapter verification (has priority)
        if self.adapter:
            context = {
                "task": task,
                "equations": payload.support.equations if payload.support else [],
                "points": [
                    {"x": p.x, "y": p.y}
                    for p in (payload.support.points if payload.support else [])
                ],
            }
            ok, reason = self.adapter.verify_solution(answer, context)
            if not ok:
                return False, f"domain verification failed: {reason}"

        # Validate points mathematically if provided (fallback sympy check)
        if payload.support and payload.support.points:
            ok, reason = self._verify_points(
                task,
                payload.support.points,
                payload.support.equations if payload.support else [],
            )
            if not ok:
                return False, reason

        return True, "validated"

    def _verify_points(
        self,
        task: str,
        points: Sequence[SupportPoint],
        equations: Sequence[str],
    ) -> Tuple[bool, str]:
        """Validate that provided points satisfy the equations from the task.

        Args:
            task: Original task (for equation extraction).
            points: Points to validate.
            equations: Equations to validate against.

        Returns:
            Tuple of (is_valid, reason).
        """
        if sp is None or not points:
            return True, "symbolic check skipped"

        eqs = list(equations or [])
        if not eqs:
            eqs.extend(self._extract_equations(task))

        if not eqs:
            return True, "no equations to validate"

        x_var, y_var = sp.symbols("x y")
        local_dict = {"x": x_var, "y": y_var}
        parsed = []

        for raw in eqs[:4]:
            normalized = raw.replace("^", "**")
            if "=" in normalized:
                left, right = normalized.split("=", 1)
            else:
                left, right = normalized, "0"
            try:
                left_expr = sp.sympify(left, locals=local_dict)
                right_expr = sp.sympify(right, locals=local_dict)
                parsed.append(sp.simplify(left_expr - right_expr))
            except Exception:
                continue

        if not parsed:
            return True, "could not parse equations"

        tolerance = self.thresholds.sympy_tolerance

        for point in points[:4]:
            px = point.x
            py = point.y
            if px == 0.0 and py == 0.0:
                continue  # Skip default/unset points

            for expr in parsed:
                try:
                    result = expr.subs({"x": px, "y": py})
                    numeric = float(result.evalf())
                except Exception:
                    continue

                if not math.isfinite(numeric):
                    return False, "non-finite verification value"

                if abs(numeric) > tolerance:
                    return False, f"point ({px}, {py}) does not satisfy equation"

        return True, "equations satisfied"

    def _extract_equations(self, task: str) -> List[str]:
        """Extract equations from task text for validation.

        Args:
            task: Task text.

        Returns:
            List of equation strings.
        """
        cleaned = re.sub(r"\band\b", ",", task, flags=re.IGNORECASE)
        matches = re.findall(r"([xy]\s*=[^,;]+)", cleaned)
        return [match.strip() for match in matches]
