"""Deterministic orchestrator implementing the MAKER loop."""


import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .completeness import CompletenessChecker
from .composer import FinalComposer
from .decomposer import Decomposer
from .discriminator import DecompositionDiscriminator, SolutionDiscriminator
from .io import EventLogger
from .llm import MetricsTracker
from .progress import ProgressTracker, RunReporter
from .red_flag import RedFlagGuard
from .schemas import (
    AtomicSolution,
    CompletenessResult,
    DecompositionProposal,
    FinalAnswer,
    RunArtifacts,
    RunResult,
    RunStats,
    StepTrace,
    SwarmConfig,
    TaskState,
)
from .solver import AtomicSolver
from .verify import GlobalVerifier, StateVerifier


@dataclass
class MakerRuntime:
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


class MakerOrchestrator:
    """Recursive orchestrator coordinating decomposer and solver agents."""

    def __init__(self, runtime: MakerRuntime) -> None:
        self.runtime = runtime
        self.task_state: Optional[TaskState] = None
        self.step_traces: List[StepTrace] = []
        self._step_counter = 0
        self.final_payload: Optional[FinalAnswer] = None
        self._force_finalize = False
        self._non_final_steps = 0
        self._stagnation_triggered = False

    def run(self, task: str) -> RunResult:
        self.task_state = TaskState(task=task, current_problem=task)
        self.runtime.progress_tracker.reset()
        self.final_payload = None
        self._force_finalize = False
        self._non_final_steps = 0
        self._stagnation_triggered = False
        start = time.perf_counter()
        self._reporter_info(f"Starting task: {task}")
        direct_payload = self._execute_direct_phase()
        if direct_payload:
            final_payload = direct_payload
        else:
            intermediate = self._solve_problem(task, depth=0)
            final_payload = self._finalize(stage="decomposed")
            if not final_payload and intermediate:
                self._reporter_info("Falling back to best intermediate result.")
                self.task_state.draft_answer = intermediate
                final_payload = self._finalize(stage="fallback")
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
        artifacts = RunArtifacts()
        result = RunResult(
            task=task,
            final_answer=final_answer,
            final_payload=final_payload,
            steps=self.step_traces,
            stats=stats,
            artifacts=artifacts,
        )
        if not final_payload:
            result.stats.aborted_reason = "final answer not verified"
        return result

    # Internal helpers -------------------------------------------------
    def _execute_direct_phase(self) -> Optional[FinalAnswer]:
        if not self.task_state:
            return None
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
                batch_size=self.runtime.config.direct_attempt_batch_size,
                max_rounds=self.runtime.config.direct_attempt_rounds,
                temperature=self.runtime.config.temperature_solver * 0.7,
            )
        except RuntimeError:
            return None
        payload = self._finalize(stage="direct")
        if payload:
            self._reporter_info("Direct stage satisfied the task.")
        else:
            self._reporter_info("Direct stage could not be verified; continuing.")
        return payload

    def _solve_problem(self, problem: str, depth: int) -> str:
        if not self.task_state:
            raise RuntimeError("Task state not initialized")
        self.task_state.current_problem = problem
        self.task_state.depth = depth
        if self._force_finalize:
            return self.task_state.draft_answer or problem

        if depth >= self.runtime.config.max_depth:
            return self._solve_atomic(problem, depth, forced=True, mode="max_depth")

        proposal, trace = self._run_decomposition(problem, depth)
        if not proposal or proposal.is_atomic:
            return self._solve_atomic(problem, depth, forced=proposal is None, mode="atomic")

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
        return combined

    def _finalize(self, stage: str, feedback: Optional[str] = None, _gap_depth: int = 0) -> Optional[FinalAnswer]:
        if not self.task_state:
            return None
        if _gap_depth > 2:
            self._reporter_info("Max gap-fill depth reached.")
            return None

        reason = feedback
        payload: Optional[FinalAnswer] = None

        for attempt in range(2):
            step_id = self._next_step_id()
            candidate = self.runtime.composer.compose(
                task=self.task_state.task,
                state=self.task_state,
                step_id=step_id,
                feedback=reason if attempt else feedback,
            )

            # Check completeness with LLM
            completeness = self.runtime.completeness_checker.check(
                task=self.task_state.task,
                answer=candidate.answer,
                state=self.task_state,
                step_id=step_id,
            )

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
                model=self.runtime.config.model_solver,
            )

            if completeness.complete:
                # Also run basic sanity checks (sympy validation if points present)
                ok, verify_reason = self.runtime.global_verifier.verify(
                    self.task_state.task, candidate, self.task_state
                )
                if ok:
                    self._reporter_info("Answer complete and verified.")
                    payload = candidate
                    break
                else:
                    reason = verify_reason
                    self.task_state.notes.append(f"verification failed: {verify_reason}")
                    self.runtime.metrics.increment_retry()
                    continue

            # Not complete - fill gaps with additional atomic solves
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
                        self.task_state.notes.append(f"gap filled: {missing[:50]}... -> {gap_solution[:50]}...")
                        self._reporter_info(f"Gap filled: {missing[:40]}...")
                    except RuntimeError as e:
                        self.task_state.notes.append(f"gap fill failed: {missing[:50]}... ({e})")

                # Recompose with filled gaps
                return self._finalize(stage="gap_filled", _gap_depth=_gap_depth + 1)

            # No missing work but incomplete - use reason as feedback
            reason = "; ".join(
                r.reason for r in completeness.requirements if r.status == "MISSING"
            )
            self.task_state.notes.append(f"incomplete: {reason}")
            self.runtime.metrics.increment_retry()

        if payload:
            self.final_payload = payload
        else:
            self._reporter_info("Final answer could not be completed.")
        return payload

    def _run_decomposition(self, problem: str, depth: int) -> Tuple[Optional[DecompositionProposal], StepTrace]:
        step_id = self._next_step_id()
        self._report_step(step_id, "decomposition", problem, depth)
        trace = StepTrace(step_id=step_id, kind="decomposition", problem=problem, depth=depth)
        self.runtime.logger.log(
            "decomposition_start",
            {"problem": problem, "depth": depth},
            step_id=step_id,
            stage="decomposition",
            agent="decomposer",
            model=self.runtime.config.model_decomposer,
        )
        proposals: List[DecompositionProposal] = []
        prefer_atomic_forced = False
        outcome = None
        selected_proposal: Optional[DecompositionProposal] = None
        for _ in range(self.runtime.config.max_rounds):
            batch = self.runtime.decomposer.generate(problem, depth, step_id, self.runtime.config.batch_size)
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
            trace.candidates = [proposal.model_dump() for proposal in proposals]
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
            atomic_ratio = self._atomic_ratio(proposals)
            atomic_samples_ready = len(proposals) >= self.runtime.config.prefer_atomic_min_samples
            should_force_atomic = (
                (not prefer_atomic_forced)
                and atomic_samples_ready
                and atomic_ratio >= self.runtime.config.prefer_atomic_ratio
            )

            if candidate and (candidate.is_atomic or outcome.confident):
                self.runtime.logger.log(
                    "decomposition_selected",
                    {"proposal": candidate.model_dump(), "confident": outcome.confident},
                    step_id=step_id,
                    stage="decomposition",
                    agent="discriminator",
                    model="ahead-by-k",
                )
                self._reporter_info("Decomposition consensus reached.")
                selected_proposal = candidate
                break
            if should_force_atomic:
                prefer_atomic_forced = True
                fallback = self._best_atomic_candidate(proposals, outcome.votes if outcome else {})
                if fallback:
                    trace.chosen = fallback.model_dump()
                    self.runtime.logger.log(
                        "decomposition_selected",
                        {"proposal": fallback.model_dump(), "confident": False, "forced_atomic": True},
                        step_id=step_id,
                        stage="decomposition",
                        agent="discriminator",
                        model="ahead-by-k",
                    )
                    self._reporter_info("Decomposition forced atomic due to low consensus.")
                    selected_proposal = fallback
                    break
        else:
            # Max rounds exhausted - pick best available proposal (may be None)
            outcome = self.runtime.decomposition_discriminator.select(proposals)
            trace.votes = outcome.votes
            self.runtime.metrics.add_votes(sum(outcome.votes.values()))
            candidate = self._select_decomposition_candidate(proposals, outcome.votes if outcome else {})
            if candidate:
                trace.chosen = candidate.model_dump()
                self.runtime.logger.log(
                    "decomposition_selected",
                    {"proposal": candidate.model_dump(), "confident": False},
                    step_id=step_id,
                    stage="decomposition",
                    agent="discriminator",
                    model="ahead-by-k",
                )
                selected_proposal = candidate
            else:
                fallback = self._best_atomic_candidate(proposals, outcome.votes if outcome else {})
                if fallback:
                    trace.chosen = fallback.model_dump()
                    self.runtime.logger.log(
                        "decomposition_selected",
                        {"proposal": fallback.model_dump(), "confident": False, "forced_atomic": True},
                        step_id=step_id,
                        stage="decomposition",
                        agent="discriminator",
                        model="ahead-by-k",
                    )
                    self._reporter_info("Max rounds reached; falling back to atomic proposal.")
                    selected_proposal = fallback

        self.step_traces.append(trace)
        self._after_step("decomposition")
        self._report_metrics()
        winner = selected_proposal or (outcome.winner if outcome and outcome.winner else None)
        return (winner, trace)

    def _atomic_ratio(self, proposals: Sequence[DecompositionProposal]) -> float:
        if not proposals:
            return 0.0
        atomic = sum(1 for proposal in proposals if proposal.is_atomic)
        return atomic / len(proposals)

    def _best_atomic_candidate(
        self, proposals: Sequence[DecompositionProposal], votes: Dict[str, int]
    ) -> Optional[DecompositionProposal]:
        best: Optional[DecompositionProposal] = None
        best_votes = -1
        for proposal in proposals:
            if not proposal.is_atomic:
                continue
            count = votes.get(proposal.signature, 0)
            if count > best_votes:
                best = proposal
                best_votes = count
        return best

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
        if not self.task_state:
            raise RuntimeError("Task state not initialized")
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
            model=self.runtime.config.model_solver,
        )
        accepted: List[AtomicSolution] = []
        rejections: List[dict] = []
        trace.notes = f"mode={mode}"
        rounds = max_rounds or self.runtime.config.max_rounds
        sample_size = batch_size or self.runtime.config.batch_size

        for _ in range(rounds):
            batch = self.runtime.solver.solve(problem, step_id, sample_size, temperature=temperature)
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
                        model=self.runtime.config.model_solver,
                    )
                    continue
                ok, msg = self.runtime.verifier.verify_solution(candidate)
                if not ok:
                    rejections.append({"pattern": "verifier", "reason": msg, "solution": candidate.model_dump()})
                    self.runtime.logger.log(
                        "verifier_reject",
                        {"reason": msg, "solution": candidate.model_dump()},
                        step_id=step_id,
                        stage="atomic",
                        agent="verifier",
                        model=self.runtime.config.model_solver,
                    )
                    continue
                accepted.append(candidate)
            outcome = self.runtime.solution_discriminator.select(accepted)
            trace.candidates = [solution.model_dump() for solution in accepted]
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
            if candidate and (outcome.confident or mode in {"direct", "stagnation"}):
                trace.chosen = candidate.model_dump()
                self.runtime.logger.log(
                    "solution_selected",
                    {"solution": candidate.model_dump(), "confident": outcome.confident},
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

        if not accepted:
            raise RuntimeError(f"No valid atomic solutions generated for: {problem}")

        outcome = self.runtime.solution_discriminator.select(accepted)
        candidate = self._select_solution_candidate(accepted, outcome.votes)
        winner = candidate or outcome.winner or accepted[0]
        trace.candidates = [solution.model_dump() for solution in accepted]
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

    def _select_decomposition_candidate(
        self, proposals: Sequence[DecompositionProposal], votes: Dict[str, int]
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
        self, solutions: Sequence[AtomicSolution], votes: Dict[str, int]
    ) -> Optional[AtomicSolution]:
        best: Optional[AtomicSolution] = None
        best_score = float("-inf")
        for solution in solutions:
            score = self.runtime.verifier.score_solution(solution, votes)
            if score > best_score:
                best = solution
                best_score = score
        return best

    def _after_step(self, kind: str) -> None:
        self._non_final_steps += 1
        if self._non_final_steps >= self.runtime.config.max_non_final_steps and not self._force_finalize:
            self._force_finalize = True
            self.runtime.logger.log(
                "forced_finalization",
                {"kind": kind, "limit": self.runtime.config.max_non_final_steps},
                agent="orchestrator",
                stage="progress",
                step_id=self._step_counter,
            )
            if self.task_state:
                self.task_state.notes.append("non-final step cap triggered; forcing finalization.")

    def _record_progress(self, note: str) -> None:
        if not self.task_state:
            return
        snapshot = self.runtime.progress_tracker.record(self.task_state)
        if not snapshot["changed"] and self.runtime.progress_tracker.stagnant() and not self._stagnation_triggered:
            self._stagnation_triggered = True
            self.runtime.logger.log(
                "stagnation_detected",
                {"note": note, "hash": snapshot["hash"]},
                agent="orchestrator",
                stage="progress",
                step_id=self._step_counter,
            )
            self.task_state.notes.append(f"stagnation detected ({note})")
            self._force_finalize = True
            self._handle_stagnation()

    def _handle_stagnation(self) -> None:
        if not self.task_state:
            return
        scaled_temp = max(self.runtime.config.temperature_solver * self.runtime.config.stagnation_temperature_scale, 0.1)
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
        except RuntimeError:
            pass
