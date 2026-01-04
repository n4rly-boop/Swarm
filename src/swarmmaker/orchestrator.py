"""Deterministic orchestrator implementing the MAKER loop."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .decomposer import Decomposer
from .discriminator import DecompositionDiscriminator, SolutionDiscriminator
from .io import EventLogger
from .llm import MetricsTracker
from .progress import RunReporter
from .red_flag import RedFlagGuard
from .schemas import (
    AtomicSolution,
    DecompositionProposal,
    RunArtifacts,
    RunResult,
    RunStats,
    StepTrace,
    SwarmConfig,
    TaskState,
)
from .solver import AtomicSolver
from .verify import StateVerifier


@dataclass
class MakerRuntime:
    config: SwarmConfig
    decomposer: Decomposer
    solver: AtomicSolver
    decomposition_discriminator: DecompositionDiscriminator
    solution_discriminator: SolutionDiscriminator
    red_flag: RedFlagGuard
    verifier: StateVerifier
    logger: EventLogger
    metrics: MetricsTracker
    reporter: Optional[RunReporter] = None


class MakerOrchestrator:
    """Recursive orchestrator coordinating decomposer and solver agents."""

    def __init__(self, runtime: MakerRuntime) -> None:
        self.runtime = runtime
        self.task_state: Optional[TaskState] = None
        self.step_traces: List[StepTrace] = []
        self._step_counter = 0

    def run(self, task: str) -> RunResult:
        self.task_state = TaskState(task=task, current_problem=task)
        start = time.perf_counter()
        self._reporter_info(f"Starting task: {task}")
        final_answer = self._solve_problem(task, depth=0)
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
        return RunResult(
            task=task,
            final_answer=final_answer,
            steps=self.step_traces,
            stats=stats,
            artifacts=artifacts,
        )

    # Internal helpers -------------------------------------------------
    def _solve_problem(self, problem: str, depth: int) -> str:
        if not self.task_state:
            raise RuntimeError("Task state not initialized")
        self.task_state.current_problem = problem
        self.task_state.depth = depth

        if depth >= self.runtime.config.max_depth:
            return self._solve_atomic(problem, depth, forced=True)

        proposal, trace = self._run_decomposition(problem, depth)
        if not proposal or proposal.is_atomic:
            return self._solve_atomic(problem, depth, forced=proposal is None)

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
        return combined

    def _run_decomposition(self, problem: str, depth: int) -> Tuple[Optional[DecompositionProposal], StepTrace]:
        step_id = self._next_step_id()
        self._report_step(step_id, "decomposition", problem, depth)
        trace = StepTrace(step_id=step_id, kind="decomposition", problem=problem, depth=depth)
        self.runtime.logger.log(
            "decomposition_start",
            {"problem": problem, "depth": depth},
            step_id=step_id,
            stage="decomposition",
            agent="orchestrator",
            model=self.runtime.config.model_decomposer,
        )
        proposals: List[DecompositionProposal] = []
        outcome = None
        for _ in range(self.runtime.config.max_rounds):
            batch = self.runtime.decomposer.generate(problem, depth, step_id, self.runtime.config.batch_size)
            proposals.extend(batch)
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
                model=self.runtime.config.model_discriminator,
            )
            if outcome.winner:
                trace.chosen = outcome.winner.model_dump()
                if outcome.winner.is_atomic or outcome.confident:
                    self.runtime.logger.log(
                        "decomposition_selected",
                        {"proposal": outcome.winner.model_dump(), "confident": outcome.confident},
                        step_id=step_id,
                        stage="decomposition",
                        agent="discriminator",
                        model=self.runtime.config.model_discriminator,
                    )
                    self._reporter_info("Decomposition consensus reached.")
                    break
        else:
            # Max rounds exhausted - pick best available proposal (may be None)
            outcome = self.runtime.decomposition_discriminator.select(proposals)
            trace.votes = outcome.votes
            self.runtime.metrics.add_votes(sum(outcome.votes.values()))
            if outcome.winner:
                trace.chosen = outcome.winner.model_dump()
                self.runtime.logger.log(
                    "decomposition_selected",
                    {"proposal": outcome.winner.model_dump(), "confident": False},
                    step_id=step_id,
                    stage="decomposition",
                    agent="discriminator",
                    model=self.runtime.config.model_discriminator,
                )

        self.step_traces.append(trace)
        self._report_metrics()
        return ((outcome.winner if outcome and outcome.winner else None), trace)

    def _solve_atomic(self, problem: str, depth: int, *, forced: bool) -> str:
        if not self.task_state:
            raise RuntimeError("Task state not initialized")
        state = self.task_state

        step_id = self._next_step_id()
        self._report_step(step_id, "atomic", problem, depth)
        trace = StepTrace(step_id=step_id, kind="atomic", problem=problem, depth=depth)
        self.runtime.logger.log(
            "atomic_start",
            {"problem": problem, "depth": depth, "forced": forced},
            step_id=step_id,
            stage="atomic",
            agent="orchestrator",
            model=self.runtime.config.model_solver,
        )
        accepted: List[AtomicSolution] = []
        rejections: List[dict] = []

        for _ in range(self.runtime.config.max_rounds):
            batch = self.runtime.solver.solve(problem, step_id, self.runtime.config.batch_size)
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
                model=self.runtime.config.model_discriminator,
            )
            if outcome.winner and outcome.confident:
                trace.chosen = outcome.winner.model_dump()
                self.runtime.logger.log(
                    "solution_selected",
                    {"solution": outcome.winner.model_dump(), "confident": True},
                    step_id=step_id,
                    stage="atomic",
                    agent="discriminator",
                    model=self.runtime.config.model_discriminator,
                )
                self.step_traces.append(trace)
                self._reporter_info("Atomic solution selected with confidence.")
                self._report_metrics()
                self.runtime.verifier.commit_solution(problem, outcome.winner, state)
                return outcome.winner.solution

        if not accepted:
            raise RuntimeError(f"No valid atomic solutions generated for: {problem}")

        outcome = self.runtime.solution_discriminator.select(accepted)
        trace.candidates = [solution.model_dump() for solution in accepted]
        trace.votes = outcome.votes
        trace.chosen = outcome.winner.model_dump() if outcome.winner else accepted[0].model_dump()
        trace.rejections = rejections
        self.runtime.metrics.add_votes(sum(outcome.votes.values()))
        self.runtime.logger.log(
            "solution_selected",
            {"solution": trace.chosen, "confident": outcome.confident},
            step_id=step_id,
            stage="atomic",
            agent="discriminator",
            model=self.runtime.config.model_discriminator,
        )
        self.step_traces.append(trace)
        self._reporter_info("Atomic solution selected via fallback.")
        self._report_metrics()
        winner = outcome.winner or accepted[0]
        self.runtime.verifier.commit_solution(problem, winner, state)
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
