"""Verification and quality gates for SwarmMaker."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import sympy as sp
except Exception:  # pragma: no cover - fallback when sympy missing
    sp = None

from .schemas import (
    AtomicSolution,
    DecompositionProposal,
    FinalAnswer,
    ProblemFact,
    SupportPoint,
    TaskState,
)


class StateVerifier:
    """Applies structural checks, quality gates, and progress scoring."""

    ACTION_KEYWORDS = (
        "solve",
        "find",
        "compute",
        "determine",
        "substitute",
        "eliminate",
        "compare",
        "differentiate",
        "integrate",
        "simplify",
        "factor",
    )

    def __init__(self, *, max_work_shown_chars: int = 1200, max_facts: int = 6) -> None:
        self._seen_signatures: set[str] = set()
        self.max_work_shown_chars = max_work_shown_chars
        self.max_facts = max_facts
        self._fact_keys: List[str] = []

    # Atomic solution checks -------------------------------------------------
    def verify_solution(self, solution: AtomicSolution) -> Tuple[bool, str]:
        if not solution.solution.strip():
            return False, "solution must not be empty"
        if not solution.work_shown.strip():
            return False, "work_shown must document intermediate steps"
        if len(solution.work_shown.strip()) > self.max_work_shown_chars:
            solution.work_shown = self._truncate(solution.work_shown.strip(), self.max_work_shown_chars)
        if solution.signature in self._seen_signatures:
            return False, "duplicate solution signature"
        return True, "validated"

    def commit_solution(self, problem: str, solution: AtomicSolution, state: TaskState, *, depth: int, step_id: int) -> None:
        self._seen_signatures.add(solution.signature)
        fact_key = f"step-{step_id}"
        fact = ProblemFact(
            problem=problem.strip(),
            solution=solution.solution.strip(),
            work_shown=self._truncate(solution.work_shown.strip(), self.max_work_shown_chars),
            confidence=solution.confidence,
            depth=depth,
        )
        state.solved_subproblems[problem] = fact.solution
        state.current_problem = problem
        state.draft_answer = fact.solution
        state.facts[fact_key] = fact
        self._fact_keys.append(fact_key)
        if len(self._fact_keys) > self.max_facts:
            oldest = self._fact_keys.pop(0)
            state.facts.pop(oldest, None)

    # Decomposition gates ----------------------------------------------------
    def validate_decomposition(self, parent: str, proposal: DecompositionProposal) -> Tuple[bool, str]:
        parent_norm = self._normalize(parent)
        sub_a = self._normalize(proposal.subproblem_a)
        sub_b = self._normalize(proposal.subproblem_b)
        if sub_a == parent_norm or sub_b == parent_norm:
            return False, "subproblem duplicates parent"
        if sub_a == sub_b:
            return False, "subproblems must differ"
        if self._is_tautology(proposal.subproblem_a) or self._is_tautology(proposal.subproblem_b):
            return False, "tautological subproblem rejected"
        if not self._has_action(proposal.subproblem_a) or not self._has_action(proposal.subproblem_b):
            return False, "subproblems must describe concrete reductions"
        return True, "ok"

    def score_decomposition(self, proposal: DecompositionProposal, votes: Dict[str, int]) -> float:
        vote_weight = votes.get(proposal.signature, 0)
        keyword_hits = sum(self._keyword_hits(text) for text in (proposal.subproblem_a, proposal.subproblem_b))
        progress_weight = 0.2 * keyword_hits + (0.5 if not proposal.is_atomic else 0.0)
        verbosity_penalty = (len(proposal.subproblem_a) + len(proposal.subproblem_b) + len(proposal.compose_fn)) / 900.0
        correctness_weight = 0.3 if "combine" in proposal.compose_fn.lower() else 0.0
        return vote_weight + progress_weight + correctness_weight - verbosity_penalty

    # Atomic selection scoring -----------------------------------------------
    def score_solution(self, solution: AtomicSolution, votes: Dict[str, int]) -> float:
        vote_weight = votes.get(solution.signature, 0)
        novelty = 0.8 if solution.signature not in self._seen_signatures else 0.0
        verbosity_penalty = len(solution.work_shown) / 600.0
        return vote_weight + novelty + solution.confidence - verbosity_penalty

    # Utility helpers --------------------------------------------------------
    def compose(self, compose_fn: str, solution_a: str, solution_b: str) -> str:
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
        lowered = text.lower()
        taut_pattern = re.compile(r"solve\s+([a-z]\w*)\s*=\s*[^=]+?\s+for\s+\1\b")
        if taut_pattern.search(lowered):
            return True
        if lowered.startswith("rewrite ") or lowered.startswith("restate "):
            return True
        return False

    def _has_action(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in self.ACTION_KEYWORDS)

    def _keyword_hits(self, text: str) -> int:
        lowered = text.lower()
        return sum(1 for keyword in self.ACTION_KEYWORDS if keyword in lowered)


class GlobalVerifier:
    """Global validator that ensures the final answer addresses the original task."""

    STOPWORDS = {"find", "the", "and", "points", "point", "what", "determine", "solve"}

    def verify(self, task: str, payload: FinalAnswer, state: TaskState) -> Tuple[bool, str]:
        answer = payload.answer.strip()
        if not answer or len(answer) < 3:
            return False, "final answer is empty"

        coverage = self._covers_keywords(task, answer, payload)
        has_struct_support = payload.support is not None and (
            bool(payload.support.points) or bool(payload.support.equations)
        )

        if payload.support and payload.support.points:
            ok, reason = self._verify_points(task, payload.support.points, payload.support.equations)
            if not ok:
                return False, reason
            # Successful structured verification is strong enough even if lexical coverage is weak.
            if ok:
                coverage = True

        if not coverage and not has_struct_support:
            return False, "final answer missing required task terms"

        return True, "validated"

    def _covers_keywords(self, task: str, answer: str, payload: FinalAnswer) -> bool:
        answer_lower = answer.lower()
        support_text = (payload.support.summary or "").lower() if payload.support else ""
        tokens = [tok for tok in re.findall(r"[a-z]{4,}", task.lower()) if tok not in self.STOPWORDS]
        if not tokens:
            return True
        hits = 0
        for token in set(tokens):
            if token in answer_lower or token in support_text:
                hits += 1
        return hits >= min(2, len(set(tokens)))

    def _verify_points(self, task: str, points: Sequence[SupportPoint], equations: Sequence[str]) -> Tuple[bool, str]:
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

        for point in points[:4]:
            px = point.values.get("x")
            py = point.values.get("y")
            if px is None and py is None:
                continue
            for expr in parsed:
                try:
                    result = expr.subs({"x": px, "y": py})
                    numeric = float(result.evalf())
                except Exception:
                    continue
                if not math.isfinite(numeric):
                    return False, "non-finite verification value"
                if abs(numeric) > 1e-5:
                    return False, f"point {point.values} does not satisfy equation"
        return True, "equations satisfied"

    def _extract_equations(self, task: str) -> List[str]:
        cleaned = re.sub(r"\band\b", ",", task, flags=re.IGNORECASE)
        matches = re.findall(r"([xy]\s*=[^,;]+)", cleaned)
        return [match.strip() for match in matches]
