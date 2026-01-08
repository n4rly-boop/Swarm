"""Deterministic orchestrator implementing the MAKER loop.

This module coordinates all agents and implements the core algorithm:
- Recursive decomposition with voting
- Atomic solving with ahead-by-K consensus
- Error handling with logging
- Timeout and cycle detection
- Progress tracking and stagnation handling
"""
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .completeness import CompletenessChecker
from .composer import FinalComposer
from .decomposer import Decomposer
from .io import EventLogger
from .llm import MetricsTracker
from .red_flag import RedFlagGuard
from .schemas import (
    AtomicSolution,
    CompletenessResult,
    DecompositionProposal,
    FinalAnswer,
    RunArtifacts,
    RunResult,
    RunStats,
    SchemaValidationError,
    StepTrace,
    SwarmConfig,
    TaskState,
    canonical_json,
)
from .solver import AtomicSolver
from .verify import GlobalVerifier, StateVerifier
from .voting import DecompositionDiscriminator, SolutionDiscriminator


class OrchestratorError(RuntimeError):
    """Base error for orchestrator failures."""

    pass


class TimeoutError(OrchestratorError):
    """Raised when orchestrator exceeds total timeout."""

    pass


class CycleDetectedError(OrchestratorError):
    """Raised when circular decomposition is detected."""

    pass


# ---------------------------------------------------------------------------
# Progress Tracker (absorbed from progress.py)
# ---------------------------------------------------------------------------


@dataclass
class ProgressTracker:
    """Tracks whether the run state is making meaningful progress."""

    max_stagnant_rounds: int
    _last_hash: str = ""
    _stagnant_rounds: int = 0

    def record(self, state: TaskState) -> Dict[str, Any]:
        """Record current state and check for stagnation.

        Args:
            state: Current task state.

        Returns:
            Dict with 'changed', 'hash', and 'stagnant' keys.
        """
        # Hash only semantic content, not metadata
        payload = {
            "facts_solutions": sorted([fact.solution for fact in state.facts.values()]),
            "draft": state.draft_answer,
            "solved_values": sorted(state.solved_subproblems.values()),
        }
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

        changed = digest != self._last_hash
        if changed:
            self._last_hash = digest
            self._stagnant_rounds = 0
            state.progress_version += 1
        else:
            self._stagnant_rounds += 1

        state.progress_hash = digest
        return {"changed": changed, "hash": digest, "stagnant": self._stagnant_rounds}

    def stagnant(self) -> bool:
        return self._stagnant_rounds >= self.max_stagnant_rounds

    def reset(self) -> None:
        self._last_hash = ""
        self._stagnant_rounds = 0


@dataclass
class RunReporter:
    """Streams textual updates about steps and metrics."""

    emit: Callable[[str], None]

    def start(self, task: str, config: SwarmConfig) -> None:
        self.emit("")
        self.emit("=== SwarmMaker MAKER Run ===")
        self.emit(f"Task: {task}")
        self.emit(
            f"Models: reasoning={config.models.reasoning}, "
            f"execution={config.models.execution}, "
            "discriminator=ahead-by-k"
        )
        self.emit(
            f"Batch size={config.batch_size}, ahead_by={config.ahead_by}, "
            f"max_depth={config.max_depth}, max_rounds={config.max_rounds}"
        )
        self.emit("")

    def step(self, *, step_id: int, kind: str, problem: str, depth: int) -> None:
        normalized = problem.replace("\n", " ")[:120]
        self.emit(f"[step {step_id:02d}] {kind.upper()} depth={depth} :: {normalized}")

    def info(self, message: str) -> None:
        self.emit(f"  -> {message}")

    def metrics(self, snapshot: Dict[str, float]) -> None:
        self.emit(
            f"  tokens: {int(snapshot.get('tokens_in', 0))}/{int(snapshot.get('tokens_out', 0))} "
            f"| llm_calls: {int(snapshot.get('llm_calls', 0))} "
            f"| votes: {int(snapshot.get('consensus_votes', 0))} "
            f"| retries: {int(snapshot.get('retries', 0))} "
            f"| steps: {int(snapshot.get('steps', 0))} "
            f"| budget: {int(snapshot.get('budget_remaining', 0))}"
        )


# ---------------------------------------------------------------------------
# Runtime Container
# ---------------------------------------------------------------------------


@dataclass
class MakerRuntime:
    """Container for all runtime components."""

    config: SwarmConfig
    decomposer: Decomposer
    solver: AtomicSolver
    decomposition_discriminator: DecompositionDiscriminator
    solution_discriminator: SolutionDiscriminator
    red_flag: RedFlagGuard
    verifier: StateVerifier
    global_verifier: GlobalVerifier
    completeness_checker: CompletenessChecker
    composer: FinalComposer
    logger: EventLogger
    metrics: MetricsTracker
    progress_tracker: ProgressTracker
    reporter: Optional[RunReporter] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class MakerOrchestrator:
    """Recursive orchestrator coordinating decomposer and solver agents.

    Key features:
    - Timeout detection at orchestrator level
    - Cycle detection in decomposition
    - Error handling with logging for all LLM calls
    - Progress tracking with stagnation handling
    """

    def __init__(self, runtime: MakerRuntime) -> None:
        self.runtime = runtime
        self.task_state: Optional[TaskState] = None
        self.step_traces: List[StepTrace] = []
        self._step_counter = 0
        self.final_payload: Optional[FinalAnswer] = None
        self._force_finalize = False
        self._non_final_steps = 0
        self._stagnation_triggered = False

        # Timeout tracking
        self._deadline: float = 0.0

        # Cycle detection
        self._decomposition_stack: Set[str] = set()

        # Solution memoization
        self._solution_cache: Dict[str, str] = {}

    def run(self, task: str) -> RunResult:
        """Execute the MAKER loop for the given task.

        Args:
            task: Task description to solve.

        Returns:
            RunResult with final answer and execution stats.
        """
        # Initialize state
        self.task_state = TaskState(task=task, current_problem=task)
        self.runtime.progress_tracker.reset()
        self.final_payload = None
        self._force_finalize = False
        self._non_final_steps = 0
        self._stagnation_triggered = False
        self._decomposition_stack.clear()
        self._solution_cache.clear()

        # Set deadline for timeout
        self._deadline = time.perf_counter() + self.runtime.config.timeout_total_seconds

        start = time.perf_counter()
        self._reporter_info(f"Starting task: {task}")

        try:
            # Try direct solve first
            direct_payload = self._execute_direct_phase()

            if direct_payload:
                final_payload = direct_payload
            else:
                # Full decomposition path
                intermediate = self._solve_problem(task, depth=0)
                final_payload = self._finalize(stage="decomposed")

                if not final_payload and intermediate:
                    self._reporter_info("Falling back to best intermediate result.")
                    self.task_state.draft_answer = intermediate
                    final_payload = self._finalize(stage="fallback")

        except TimeoutError as e:
            self._reporter_info(f"Timeout: {e}")
            self.runtime.logger.log(
                "timeout",
                {"error": str(e)},
                agent="orchestrator",
                stage="run",
                step_id=self._step_counter,
            )
            final_payload = self._emergency_finalize()

        except CycleDetectedError as e:
            self._reporter_info(f"Cycle detected: {e}")
            final_payload = self._emergency_finalize()

        final_answer = final_payload.answer if final_payload else None
        elapsed = time.perf_counter() - start

        stats = RunStats(
            elapsed_s=elapsed,
            llm_calls=self.runtime.metrics.llm_calls,
            tokens_in=self.runtime.metrics.tokens_in,
            tokens_out=self.runtime.metrics.tokens_out,
            retries=self.runtime.metrics.retries,
            consensus_votes=self.runtime.metrics.consensus_votes,
        )

        result = RunResult(
            task=task,
            final_answer=final_answer,
            final_payload=final_payload,
            steps=self.step_traces,
            stats=stats,
            artifacts=RunArtifacts(),
        )

        if not final_payload:
            result.stats.aborted_reason = "final answer not verified"

        return result

    # -----------------------------------------------------------------------
    # Timeout Checking
    # -----------------------------------------------------------------------

    def _check_timeout(self) -> None:
        """Check if deadline has been exceeded."""
        if time.perf_counter() > self._deadline:
            raise TimeoutError(
                f"Orchestrator timeout ({self.runtime.config.timeout_total_seconds}s) exceeded"
            )

    # -----------------------------------------------------------------------
    # Direct Solve Phase
    # -----------------------------------------------------------------------

    def _execute_direct_phase(self) -> Optional[FinalAnswer]:
        """Attempt a direct low-cost solve before decomposition."""
        if not self.task_state:
            return None

        self._check_timeout()

        self.runtime.logger.log(
            "direct_solve_attempt",
            {"task": self.task_state.task},
            agent="orchestrator",
            stage="direct",
            step_id=0,
        )

        self._reporter_info("Attempting direct low-cost solve before decomposition.")

        try:
            self._solve_atomic(
                self.task_state.task,
                depth=0,
                forced=False,
                mode="direct",
                batch_size=2,  # Small batch for direct attempt
                max_rounds=1,
            )
        except RuntimeError as e:
            # Log the failure instead of silently returning
            self.runtime.logger.log(
                "direct_solve_failed",
                {"error": str(e)},
                agent="orchestrator",
                stage="direct",
                step_id=0,
            )
            return None

        payload = self._finalize(stage="direct")

        if payload:
            self._reporter_info("Direct stage satisfied the task.")
        else:
            self._reporter_info("Direct stage could not be verified; continuing.")

        return payload

    # -----------------------------------------------------------------------
    # Recursive Problem Solving
    # -----------------------------------------------------------------------

    def _solve_problem(self, problem: str, depth: int) -> str:
        """Recursively solve a problem via decomposition or atomic solving."""
        if not self.task_state:
            raise RuntimeError("Task state not initialized")

        self._check_timeout()

        # Check memoization cache
        problem_hash = hashlib.sha256(problem.strip().lower().encode()).hexdigest()[:16]
        if problem_hash in self._solution_cache:
            self.runtime.logger.log(
                "cache_hit",
                {"problem": problem[:50]},
                agent="orchestrator",
                stage="solve",
                step_id=self._step_counter,
            )
            return self._solution_cache[problem_hash]

        # Cycle detection
        if problem_hash in self._decomposition_stack:
            self.runtime.logger.log(
                "cycle_detected",
                {"problem": problem[:100]},
                agent="orchestrator",
                stage="decomposition",
                step_id=self._step_counter,
            )
            self.task_state.add_note(
                f"cycle detected, forcing atomic: {problem[:50]}...",
                self.runtime.config.thresholds.max_notes,
            )
            return self._solve_atomic(problem, depth, forced=True, mode="cycle_break")

        self._decomposition_stack.add(problem_hash)

        try:
            self.task_state.current_problem = problem
            self.task_state.depth = depth

            if self._force_finalize:
                return self.task_state.draft_answer or problem

            if depth >= self.runtime.config.max_depth:
                result = self._solve_atomic(problem, depth, forced=True, mode="max_depth")
                self._solution_cache[problem_hash] = result
                return result

            proposal, trace = self._run_decomposition(problem, depth)

            if not proposal or proposal.is_atomic:
                result = self._solve_atomic(problem, depth, forced=proposal is None, mode="atomic")
                self._solution_cache[problem_hash] = result
                return result

            self.task_state.decomposition_tree[problem] = {
                "subproblem_a": proposal.subproblem_a,
                "subproblem_b": proposal.subproblem_b,
                "compose_fn": proposal.compose_fn,
                "depth": depth,
            }

            result_a = self._solve_problem(proposal.subproblem_a, depth + 1)
            result_b = self._solve_problem(proposal.subproblem_b, depth + 1)
            combined = self.runtime.verifier.compose(proposal.compose_fn, result_a, result_b)

            self.task_state.solved_subproblems[problem] = combined
            self._record_progress("compose")
            self._solution_cache[problem_hash] = combined

            return combined

        finally:
            self._decomposition_stack.discard(problem_hash)

    # -----------------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------------

    def _finalize(
        self,
        stage: str,
        feedback: Optional[str] = None,
        _gap_depth: int = 0,
    ) -> Optional[FinalAnswer]:
        """Compose and verify the final answer."""
        if not self.task_state:
            return None

        self._check_timeout()

        if _gap_depth > 2:
            self._reporter_info("Max gap-fill depth reached.")
            return None

        reason = feedback
        payload: Optional[FinalAnswer] = None

        for attempt in range(2):
            step_id = self._next_step_id()

            # Compose with error handling
            try:
                candidate = self.runtime.composer.compose(
                    task=self.task_state.task,
                    state=self.task_state,
                    step_id=step_id,
                    feedback=reason if attempt else feedback,
                )
            except SchemaValidationError as e:
                self.runtime.logger.log(
                    "composer_error",
                    {"error": str(e), "attempt": attempt},
                    agent="composer",
                    stage="finalize",
                    step_id=step_id,
                )
                self.task_state.add_note(
                    f"composer error: {e}",
                    self.runtime.config.thresholds.max_notes,
                )
                continue
            except Exception as e:
                self.runtime.logger.log(
                    "composer_exception",
                    {"error": str(e), "type": type(e).__name__},
                    agent="composer",
                    stage="finalize",
                    step_id=step_id,
                )
                continue

            # Check completeness with error handling
            try:
                completeness = self.runtime.completeness_checker.check(
                    task=self.task_state.task,
                    answer=candidate.answer,
                    state=self.task_state,
                    step_id=step_id,
                )
            except SchemaValidationError as e:
                self.runtime.logger.log(
                    "completeness_error",
                    {"error": str(e)},
                    agent="completeness",
                    stage="finalize",
                    step_id=step_id,
                )
                # Fallback: assume incomplete
                completeness = CompletenessResult(complete=False, requirements=[], missing_work=[])
            except Exception as e:
                self.runtime.logger.log(
                    "completeness_exception",
                    {"error": str(e), "type": type(e).__name__},
                    agent="completeness",
                    stage="finalize",
                    step_id=step_id,
                )
                completeness = CompletenessResult(complete=False, requirements=[], missing_work=[])

            self.runtime.logger.log(
                "completeness_check",
                {
                    "stage": stage,
                    "attempt": attempt,
                    "complete": completeness.complete,
                    "requirements": [r.model_dump() for r in completeness.requirements],
                    "missing_work": completeness.missing_work,
                    "candidate": candidate.model_dump(),
                },
                agent="completeness",
                stage="finalize",
                step_id=step_id,
                model=self.runtime.config.get_model("reasoning"),
            )

            if completeness.complete:
                # Run basic sanity checks
                ok, verify_reason = self.runtime.global_verifier.verify(
                    self.task_state.task, candidate, self.task_state
                )
                if ok:
                    self._reporter_info("Answer complete and verified.")
                    payload = candidate
                    break
                else:
                    reason = verify_reason
                    self.task_state.add_note(
                        f"verification failed: {verify_reason}",
                        self.runtime.config.thresholds.max_notes,
                    )
                    self.runtime.metrics.increment_retry()
                    continue

            # Not complete - fill gaps
            if completeness.missing_work:
                self._reporter_info(f"Filling {len(completeness.missing_work)} missing requirement(s).")

                for missing in completeness.missing_work:
                    try:
                        gap_solution = self._solve_atomic(
                            missing,
                            depth=self.task_state.depth,
                            forced=True,
                            mode="gap_fill",
                        )
                        self.task_state.add_note(
                            f"gap filled: {missing[:50]}... -> {gap_solution[:50]}...",
                            self.runtime.config.thresholds.max_notes,
                        )
                        self._reporter_info(f"Gap filled: {missing[:40]}...")
                    except RuntimeError as e:
                        self.runtime.logger.log(
                            "gap_fill_failed",
                            {"missing": missing[:100], "error": str(e)},
                            agent="orchestrator",
                            stage="gap_fill",
                            step_id=step_id,
                        )
                        self.task_state.add_note(
                            f"gap fill failed: {missing[:50]}... ({e})",
                            self.runtime.config.thresholds.max_notes,
                        )

                # Recompose with filled gaps
                return self._finalize(stage="gap_filled", _gap_depth=_gap_depth + 1)

            # No missing work but incomplete
            reason = "; ".join(
                r.reason for r in completeness.requirements if r.status == "MISSING"
            )
            self.task_state.add_note(
                f"incomplete: {reason}",
                self.runtime.config.thresholds.max_notes,
            )
            self.runtime.metrics.increment_retry()

        if payload:
            self.final_payload = payload
        else:
            # LLM composer failed - try code-based fallback using draft_answer
            if self.task_state.draft_answer:
                self._reporter_info("Using code-based fallback for final answer.")
                payload = FinalAnswer(
                    answer=self.task_state.draft_answer,
                    confidence=0.5,  # Moderate confidence for fallback
                )
                self.final_payload = payload
            else:
                self._reporter_info("Final answer could not be completed.")

        return payload

    def _emergency_finalize(self) -> Optional[FinalAnswer]:
        """Emergency finalization when timeout or error occurs."""
        if not self.task_state or not self.task_state.draft_answer:
            return None

        self._reporter_info("Emergency finalization with current draft.")

        return FinalAnswer(
            answer=self.task_state.draft_answer,
            confidence=0.3,  # Low confidence for emergency
        )

    # -----------------------------------------------------------------------
    # Decomposition
    # -----------------------------------------------------------------------

    def _run_decomposition(
        self,
        problem: str,
        depth: int,
    ) -> Tuple[Optional[DecompositionProposal], StepTrace]:
        """Run decomposition with voting."""
        self._check_timeout()

        step_id = self._next_step_id()
        self._report_step(step_id, "decomposition", problem, depth)

        trace = StepTrace(step_id=step_id, kind="decomposition", problem=problem, depth=depth)

        self.runtime.logger.log(
            "decomposition_start",
            {"problem": problem, "depth": depth},
            step_id=step_id,
            stage="decomposition",
            agent="decomposer",
            model=self.runtime.config.get_model("reasoning"),
        )

        proposals: List[DecompositionProposal] = []
        selected_proposal: Optional[DecompositionProposal] = None
        outcome = None

        for round_idx in range(self.runtime.config.max_rounds):
            self._check_timeout()

            # Temperature escalation per paper
            temperature = (
                self.runtime.config.thresholds.temperature_first_vote
                if round_idx == 0
                else self.runtime.config.thresholds.temperature_subsequent
            )

            batch = self.runtime.decomposer.generate(
                problem, depth, step_id, self.runtime.config.batch_size, temperature=temperature
            )

            filtered: List[DecompositionProposal] = []
            for proposal in batch:
                allowed, reason = self.runtime.verifier.validate_decomposition(problem, proposal)
                if not allowed:
                    trace.rejections.append({"reason": reason, "proposal": proposal.model_dump()})
                    self.runtime.logger.log(
                        "decomposition_rejected",
                        {"reason": reason, "proposal": proposal.model_dump()},
                        step_id=step_id,
                        stage="decomposition",
                        agent="verifier",
                    )
                    continue
                filtered.append(proposal)

            if not filtered:
                trace.notes = "quality gate rejected batch"
                continue

            proposals.extend(filtered)
            if not proposals:
                continue

            outcome = self.runtime.decomposition_discriminator.select(proposals)
            trace.candidates = [p.model_dump() for p in proposals]
            trace.votes = outcome.votes
            self.runtime.metrics.add_votes(sum(outcome.votes.values()))

            self.runtime.logger.log(
                "decomposition_votes",
                {"votes": outcome.votes, "count": len(proposals)},
                step_id=step_id,
                stage="decomposition",
                agent="discriminator",
                model="ahead-by-k",
            )

            candidate = self._select_decomposition_candidate(proposals, outcome.votes)
            if candidate:
                trace.chosen = candidate.model_dump()

            # Check for confident consensus
            if candidate and outcome.confident:
                self.runtime.logger.log(
                    "decomposition_selected",
                    {"proposal": candidate.model_dump(), "confident": True},
                    step_id=step_id,
                    stage="decomposition",
                    agent="discriminator",
                    model="ahead-by-k",
                )
                self._reporter_info("Decomposition consensus reached.")
                selected_proposal = candidate
                break

            # Check for atomic consensus
            if candidate and candidate.is_atomic:
                selected_proposal = candidate
                break

        else:
            # Max rounds exhausted
            if proposals:
                outcome = self.runtime.decomposition_discriminator.select(proposals)
                trace.votes = outcome.votes
                self.runtime.metrics.add_votes(sum(outcome.votes.values()))

                candidate = self._select_decomposition_candidate(proposals, outcome.votes)
                if candidate:
                    trace.chosen = candidate.model_dump()
                    selected_proposal = candidate

        self.step_traces.append(trace)
        self._after_step("decomposition")
        self._report_metrics()

        return (selected_proposal, trace)

    # -----------------------------------------------------------------------
    # Atomic Solving
    # -----------------------------------------------------------------------

    def _solve_atomic(
        self,
        problem: str,
        depth: int,
        *,
        forced: bool,
        mode: str,
        batch_size: Optional[int] = None,
        max_rounds: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Solve an atomic problem with voting."""
        if not self.task_state:
            raise RuntimeError("Task state not initialized")

        self._check_timeout()

        state = self.task_state
        step_id = self._next_step_id()
        self._report_step(step_id, "atomic", problem, depth)

        trace = StepTrace(step_id=step_id, kind="atomic", problem=problem, depth=depth)

        self.runtime.logger.log(
            "atomic_start",
            {"problem": problem, "depth": depth, "forced": forced, "mode": mode},
            step_id=step_id,
            stage="atomic",
            agent="solver",
            model=self.runtime.config.get_model("execution"),
        )

        accepted: List[AtomicSolution] = []
        rejections: List[dict] = []
        trace.notes = f"mode={mode}"

        rounds = max_rounds or self.runtime.config.max_rounds
        sample_size = batch_size or self.runtime.config.batch_size

        for round_idx in range(rounds):
            self._check_timeout()

            # Temperature escalation per paper
            round_temp = temperature
            if round_temp is None:
                round_temp = (
                    self.runtime.config.thresholds.temperature_first_vote
                    if round_idx == 0
                    else self.runtime.config.thresholds.temperature_subsequent
                )

            batch = self.runtime.solver.solve(
                problem, step_id, sample_size, temperature=round_temp, round_idx=round_idx
            )

            for candidate in batch:
                allowed, rejection = self.runtime.red_flag.inspect(candidate)
                if not allowed:
                    reason = {
                        "pattern": rejection.pattern if rejection else "unknown",
                        "reason": rejection.reason if rejection else "red flag",
                        "solution": candidate.model_dump(),
                    }
                    rejections.append(reason)
                    self.runtime.logger.log(
                        "red_flag_reject",
                        reason,
                        step_id=step_id,
                        stage="atomic",
                        agent="red_flag",
                        model=self.runtime.config.get_model("execution"),
                    )
                    continue

                ok, msg = self.runtime.verifier.verify_solution(candidate)
                if not ok:
                    rejections.append({
                        "pattern": "verifier",
                        "reason": msg,
                        "solution": candidate.model_dump(),
                    })
                    self.runtime.logger.log(
                        "verifier_reject",
                        {"reason": msg, "solution": candidate.model_dump()},
                        step_id=step_id,
                        stage="atomic",
                        agent="verifier",
                        model=self.runtime.config.get_model("execution"),
                    )
                    continue

                accepted.append(candidate)

            outcome = self.runtime.solution_discriminator.select(accepted)
            trace.candidates = [s.model_dump() for s in accepted]
            trace.votes = outcome.votes
            trace.rejections = rejections
            self.runtime.metrics.add_votes(sum(outcome.votes.values()))

            self.runtime.logger.log(
                "solution_votes",
                {"votes": outcome.votes, "accepted": len(accepted)},
                step_id=step_id,
                stage="atomic",
                agent="discriminator",
                model="ahead-by-k",
            )

            candidate = self._select_solution_candidate(accepted, outcome.votes)

            # Require confidence for normal modes
            if candidate and outcome.confident:
                trace.chosen = candidate.model_dump()
                self.runtime.logger.log(
                    "solution_selected",
                    {"solution": candidate.model_dump(), "confident": True},
                    step_id=step_id,
                    stage="atomic",
                    agent="discriminator",
                    model="ahead-by-k",
                )
                self.step_traces.append(trace)
                self._after_step("atomic")
                self._reporter_info("Atomic solution selected with confidence.")
                self._report_metrics()
                self.runtime.verifier.commit_solution(problem, candidate, state, depth=depth, step_id=step_id)
                self._record_progress(f"atomic:{mode}")
                return candidate.solution

        # Fallback selection
        if not accepted:
            raise RuntimeError(f"No valid atomic solutions generated for: {problem}")

        outcome = self.runtime.solution_discriminator.select(accepted)
        candidate = self._select_solution_candidate(accepted, outcome.votes)
        winner = candidate or outcome.winner or accepted[0]

        trace.candidates = [s.model_dump() for s in accepted]
        trace.votes = outcome.votes
        trace.chosen = winner.model_dump()
        trace.rejections = rejections
        self.runtime.metrics.add_votes(sum(outcome.votes.values()))

        self.runtime.logger.log(
            "solution_selected",
            {"solution": trace.chosen, "confident": outcome.confident},
            step_id=step_id,
            stage="atomic",
            agent="discriminator",
            model="ahead-by-k",
        )

        self.step_traces.append(trace)
        self._after_step("atomic")
        self._reporter_info("Atomic solution selected via fallback.")
        self._report_metrics()
        self.runtime.verifier.commit_solution(problem, winner, state, depth=depth, step_id=step_id)
        self._record_progress(f"atomic:{mode}")

        return winner.solution

    # -----------------------------------------------------------------------
    # Selection Helpers
    # -----------------------------------------------------------------------

    def _select_decomposition_candidate(
        self,
        proposals: Sequence[DecompositionProposal],
        votes: Dict[str, int],
    ) -> Optional[DecompositionProposal]:
        best: Optional[DecompositionProposal] = None
        best_score = float("-inf")
        for proposal in proposals:
            score = self.runtime.verifier.score_decomposition(proposal, votes)
            if score > best_score:
                best = proposal
                best_score = score
        return best

    def _select_solution_candidate(
        self,
        solutions: Sequence[AtomicSolution],
        votes: Dict[str, int],
    ) -> Optional[AtomicSolution]:
        best: Optional[AtomicSolution] = None
        best_score = float("-inf")
        for solution in solutions:
            score = self.runtime.verifier.score_solution(solution, votes)
            if score > best_score:
                best = solution
                best_score = score
        return best

    # -----------------------------------------------------------------------
    # Progress Tracking
    # -----------------------------------------------------------------------

    def _next_step_id(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def _report_step(self, step_id: int, kind: str, problem: str, depth: int) -> None:
        if self.runtime.reporter:
            self.runtime.reporter.step(step_id=step_id, kind=kind, problem=problem, depth=depth)

    def _report_metrics(self) -> None:
        if self.runtime.reporter:
            snapshot = self.runtime.metrics.snapshot(len(self.step_traces))
            self.runtime.reporter.metrics(snapshot)

    def _reporter_info(self, message: str) -> None:
        if self.runtime.reporter:
            self.runtime.reporter.info(message)

    def _after_step(self, kind: str) -> None:
        """Check step limits after each step."""
        self._non_final_steps += 1

        # Force finalization after too many non-final steps
        max_steps = self.runtime.config.max_depth * 2  # Reasonable limit
        if self._non_final_steps >= max_steps and not self._force_finalize:
            self._force_finalize = True
            self.runtime.logger.log(
                "forced_finalization",
                {"kind": kind, "limit": max_steps},
                agent="orchestrator",
                stage="progress",
                step_id=self._step_counter,
            )
            if self.task_state:
                self.task_state.add_note(
                    "step cap triggered; forcing finalization.",
                    self.runtime.config.thresholds.max_notes,
                )

    def _record_progress(self, note: str) -> None:
        """Record progress and check for stagnation."""
        if not self.task_state:
            return

        snapshot = self.runtime.progress_tracker.record(self.task_state)

        if not snapshot["changed"] and self.runtime.progress_tracker.stagnant():
            if not self._stagnation_triggered:
                self._stagnation_triggered = True
                self.runtime.logger.log(
                    "stagnation_detected",
                    {"note": note, "hash": snapshot["hash"]},
                    agent="orchestrator",
                    stage="progress",
                    step_id=self._step_counter,
                )
                self.task_state.add_note(
                    f"stagnation detected ({note})",
                    self.runtime.config.thresholds.max_notes,
                )
                self._force_finalize = True
                self._handle_stagnation()

    def _handle_stagnation(self) -> None:
        """Handle stagnation with one more attempt at reduced temperature."""
        if not self.task_state:
            return

        # Try one more solve with lower temperature
        scaled_temp = max(self.runtime.config.thresholds.temperature_subsequent * 0.5, 0.05)

        try:
            self._solve_atomic(
                self.task_state.task,
                depth=self.task_state.depth,
                forced=True,
                mode="stagnation",
                batch_size=1,
                max_rounds=1,
                temperature=scaled_temp,
            )
        except RuntimeError as e:
            self.runtime.logger.log(
                "stagnation_recovery_failed",
                {"error": str(e)},
                agent="orchestrator",
                stage="stagnation",
                step_id=self._step_counter,
            )
