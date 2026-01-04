"""LangGraph workflow for SwarmMaker."""
import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .callbacks import LiveSwarmDisplay
from .consensus import ConsensusEngine
from .domain_state import DomainState
from .finalizer import BestEffortFinalizer
from .io import EventLogger
from .llm import LLMClient, MetricsTracker
from .policies import PolicyEngine
from .progress import ProgressTracker
from .router import TaskAnalysis, TaskRouter
from .schemas import (
    Action,
    AgentCallMeta,
    PlannerStep,
    RunArtifacts,
    RunResult,
    RunStats,
    StepRecord,
    SwarmConfig,
)
from .termination import TerminationAuthority
from .verify import ActionVerifier

try:  # pragma: no cover - optional LangSmith dependency
    from langsmith import Client as LangSmithClient
except Exception:  # pragma: no cover
    LangSmithClient = None  # type: ignore[assignment]


class GraphState(TypedDict, total=False):
    task: str
    runtime: "RuntimeContext"
    task_analysis: TaskAnalysis
    domain_state: DomainState
    planner_step: Optional[PlannerStep]
    candidates: List[Action]
    chosen_action: Optional[Action]
    judge_needed: bool
    done: bool
    abort_reason: Optional[str]
    final_answer: Optional[str]
    draft_answer: Optional[str]
    history_signatures: List[str]
    steps_completed: int
    history: List[StepRecord]
    votes: Dict[str, int]
    retry_step: bool
    force_final_next_step: bool
    trigger_best_effort_final: bool
    budget_constrained_mode: bool


@dataclass
class RuntimeContext:
    llm: LLMClient
    consensus: ConsensusEngine
    verifier: ActionVerifier
    display: LiveSwarmDisplay
    events: EventLogger
    metrics: MetricsTracker
    langsmith: "LangSmithManager"
    config: SwarmConfig
    history: List[StepRecord]
    router: TaskRouter
    policy_engine: PolicyEngine
    progress: ProgressTracker
    termination: TerminationAuthority
    finalizer: BestEffortFinalizer


class LangSmithManager:
    """Creates a parent LangSmith run for the CLI session."""

    def __init__(self, enabled: bool, project_name: str, task: str) -> None:
        self.enabled = enabled and bool(LangSmithClient)
        self.project_name = project_name
        self.task = task
        self.client = None
        self.run_id: Optional[str] = None
        self.run_url: Optional[str] = None
        if self.enabled and LangSmithClient:
            try:
                self.client = LangSmithClient()
                run = self.client.create_run(  # type: ignore[call-arg]
                    name="SwarmMaker",
                    inputs={"task": task},
                    project_name=project_name,
                    run_type="workflow",
                    tags=["swarmmaker"],
                    metadata={"task": task},
                )
                self.run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
                if not self.run_id and isinstance(run, dict):
                    self.run_id = run.get("id")
                self.run_url = getattr(run, "url", None) or getattr(run, "dashboard_url", None)
                if not self.run_url and isinstance(run, dict):
                    self.run_url = run.get("url")
            except Exception:  # pragma: no cover - network
                self.enabled = False

    def complete(self, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        if not (self.client and self.run_id):
            return
        try:
            self.client.update_run(  # type: ignore[call-arg]
                run_id=self.run_id,
                outputs=outputs or {},
                error=error,
            )
        except Exception:  # pragma: no cover - telemetry only
            pass


def build_graph(config: SwarmConfig) -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("check", check_node(config))
    graph.add_node("plan", plan_node(config))
    graph.add_node("propose", propose_node(config))
    graph.add_node("aggregate", aggregate_node(config))
    graph.add_node("verify_apply", verify_apply_node(config))
    graph.add_node("final", final_node())

    graph.set_entry_point("check")
    graph.add_conditional_edges(
        "check",
        lambda s: "final" if s.get("abort_reason") or s.get("done") else "plan",
        {"plan": "plan", "final": "final"},
    )
    graph.add_edge("plan", "propose")
    graph.add_edge("propose", "aggregate")
    graph.add_edge("aggregate", "verify_apply")
    graph.add_conditional_edges("verify_apply", _verify_router, {"check": "check", "final": "final"})
    graph.add_edge("final", END)
    graph.add_edge(START, "check")
    return graph


def check_node(config: SwarmConfig):
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]
        metrics = runtime.metrics
        steps = state.get("steps_completed", 0)

        if state.get("done") or state.get("abort_reason"):
            return state

        # Policy enforcement (replaces manual budget checks)
        violation = runtime.policy_engine.enforce_all(state, runtime)
        if violation:
            state["abort_reason"] = violation
            runtime.events.log(
                "abort",
                {"reason": violation},
                message=violation,
            )
            runtime.display.log_event(f"Stopping: {violation}")
            state["done"] = True
            return state

        # Termination authority decides when to stop or force finalization
        termination = runtime.termination.decide(state, runtime)

        if termination.force_final:
            state["force_final_next_step"] = True
            runtime.events.log(
                "forced_finalization",
                {"reason": termination.message},
                message=termination.message,
            )
            runtime.display.log_event(termination.message)

        if termination.should_terminate:
            if termination.reason and termination.reason.value in ["stagnation", "budget_exceeded", "max_steps"]:
                # Trigger best-effort finalization
                state["trigger_best_effort_final"] = True
            else:
                state["abort_reason"] = termination.message
                state["done"] = True

        runtime.display.update_metrics(
            metrics.snapshot(
                steps,
                budget_remaining=max(config.max_total_tokens - metrics.tokens_total(), 0),
            )
        )
        return state

    return _node


def plan_node(config: SwarmConfig):
    def _node(state: GraphState) -> GraphState:
        if state.get("abort_reason"):
            return state
        runtime = state["runtime"]
        if state.pop("retry_step", False):
            planner_step = state.get("planner_step")
            if planner_step:
                runtime.display.log_event(f"Retrying planner step {planner_step.step_id}")
                runtime.events.log(
                    "plan_retry",
                    {"step_id": planner_step.step_id},
                    step_id=planner_step.step_id,
                    agent="planner",
                    stage="plan",
                )
                runtime.display.set_panel_text(
                    "PLANNER",
                    json.dumps(planner_step.model_dump(), ensure_ascii=False, indent=2),
                )
                state["candidates"] = []
                state["chosen_action"] = None
                state["votes"] = {}
                state["judge_needed"] = False
                return state
        step_id = state.get("steps_completed", 0) + 1

        # Check if system forcing finalization
        if state.pop("force_final_next_step", False):
            # System override - force FINAL without LLM call
            planner_step = PlannerStep(
                step_id=step_id,
                step_goal="Synthesize complete final answer from all progress made so far",
                stop_condition="done",
                worker_max_tokens=512,
            )
            runtime.display.log_event("System forcing finalization (no planner call)")
        else:
            # Normal planner flow
            domain_state = state.get("domain_state")
            draft = state.get("draft_answer")
            metrics = runtime.metrics
            budget_remaining = max(config.max_total_tokens - metrics.tokens_total(), 0)

            # Summarize domain state progress
            domain_summary = ""
            if domain_state and hasattr(domain_state, "model_dump"):
                domain_data = domain_state.model_dump()
                # Compact representation
                domain_summary = json.dumps(domain_data, ensure_ascii=False)[:500]  # Limit size

            summary = {
                "task": state["task"],
                "domain_state_summary": domain_summary,
                "draft_answer": draft,
                "steps_completed": state.get("steps_completed", 0),
                "history_signatures": state.get("history_signatures", [])[-5:],
                "token_budget_remaining": budget_remaining,
                "max_steps": config.max_steps,
            }

            # Check task strategy
            task_analysis = state.get("task_analysis")
            force_done = False
            if task_analysis:
                from .router import OrchestrationStrategy
                if task_analysis.strategy == OrchestrationStrategy.SINGLE_SHOT and step_id == 1:
                    force_done = True

            planner_prompt = [
                SystemMessage(
                    content=(
                        "You are the SwarmMaker planner coordinating specialist workers.\n"
                        "If the task can be answered now based on the domain state and draft answer, "
                        'set stop_condition="done" and step_goal="produce final answer".\n'
                        "Otherwise specify the single next goal the workers should execute.\n\n"
                        "IMPORTANT: Create GRANULAR, ATOMIC steps that require minimal computation.\n"
                        "- Each step should do ONE specific thing (e.g., 'simplify 2x+1=x²-5x+7 to x²-7x+6=0')\n"
                        "- Workers should be able to complete the step in 1-2 sentences with actual results\n"
                        "- Avoid compound goals like 'factor AND solve' - split into separate steps\n"
                        "- Prefer concrete goals with specific equations/values over abstract instructions\n"
                        "- Good: 'Factor x²-7x+6 into two binomials' (shows actual factors)\n"
                        "- Bad: 'Factor and solve the equation' (too many sub-tasks)\n\n"
                        + ("CRITICAL: This is a single-shot task - you MUST set stop_condition='done'.\n\n" if force_done else "")
                        + "Output ONLY the PlannerStep JSON. Never include chain-of-thought or commentary."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Task: {state['task']}\n"
                        f"Context summary:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n"
                        f"Choose worker_max_tokens <= {min(512, budget_remaining or 256)}."
                    )
                ),
            ]
            meta = AgentCallMeta(agent="planner", stage="PLANNER", step_id=step_id)
            schema = PlannerStep.model_json_schema()
            result = runtime.llm.structured_completion(
                planner_prompt,
                meta=meta,
                model=config.model_planner,
                temperature=config.temperature_planner,
                schema_name="PlannerStep",
                schema=schema,
                parser=PlannerStep.model_validate,
            )
            planner_step = result.content

            # Force stop_condition for single-shot
            if force_done and planner_step.stop_condition != "done":
                planner_step.stop_condition = "done"
        runtime.events.log(
            "planner_step",
            planner_step.model_dump(),
            step_id=planner_step.step_id,
            agent="planner",
            stage="plan",
            model=config.model_planner,
        )
        runtime.display.set_panel_text("PLANNER", json.dumps(planner_step.model_dump(), ensure_ascii=False, indent=2))

        # Create step record with domain state snapshot
        domain_state = state.get("domain_state")
        domain_snapshot = None
        if domain_state and hasattr(domain_state, "model_dump"):
            domain_snapshot = domain_state.model_dump()

        record = StepRecord(
            step_id=planner_step.step_id,
            planner_step=planner_step,
            domain_state_snapshot=domain_snapshot,
            draft_answer=state.get("draft_answer"),
            candidate_signatures=[],
        )
        runtime.history.append(record)
        state["planner_step"] = planner_step
        state["retry_step"] = False
        state["candidates"] = []
        state["chosen_action"] = None
        state["votes"] = {}
        state["judge_needed"] = False
        return state

    return _node


def propose_node(config: SwarmConfig):
    async def _node(state: GraphState) -> GraphState:
        planner_step = state.get("planner_step")
        if not planner_step:
            state["abort_reason"] = "planner missing"
            return state
        runtime = state["runtime"]
        actions: List[Action] = []
        votes: Dict[str, int] = {}
        lock = asyncio.Lock()
        stop_event = asyncio.Event()
        final_required = planner_step.stop_condition == "done" or "final" in planner_step.step_goal.lower()
        style_pool = [
            "check arithmetic carefully",
            "summarize existing notes",
            "compare competing ideas",
            "draft the final answer",
            "highlight missing info",
        ]

        async def run_worker(idx: int) -> None:
            if stop_event.is_set():
                return
            stage = f"VOTER#{idx}" if idx <= runtime.display.visible_voters else "VOTERS(+extra)"
            rand = random.Random(config.seed_base + idx)
            style = style_pool[rand.randrange(len(style_pool))]
            meta = AgentCallMeta(agent=f"voter_{idx}", stage=stage, step_id=planner_step.step_id, voter_id=idx)
            prompt = _voter_prompt(
                task=state["task"],
                planner_step=planner_step,
                domain_state=state.get("domain_state"),
                draft=state.get("draft_answer"),
                voter_index=idx,
                style_hint=style,
                show_rationale=config.show_rationale,
            )

            try:
                result = await asyncio.to_thread(
                    runtime.llm.structured_completion,
                    prompt,
                    meta=meta,
                    model=config.model_worker,
                    temperature=config.temperature_worker,
                    schema_name="Action",
                    schema=Action.model_json_schema(),
                    parser=Action.model_validate,
                    max_output_tokens=planner_step.worker_max_tokens,
                )
            except Exception as err:  # pragma: no cover - network
                runtime.metrics.increment_retry()
                runtime.display.set_panel_text(stage, f"Worker error: {err}")
                runtime.events.log(
                    "worker_error",
                    {"error": str(err)},
                    step_id=planner_step.step_id,
                    agent=meta.agent,
                    stage=stage,
                    model=config.model_worker,
                )
                return
            action = result.content
            if final_required and action.action_type != "FINAL":
                runtime.events.log(
                    "worker_invalid_final",
                    {"action": action.model_dump()},
                    step_id=planner_step.step_id,
                    agent=meta.agent,
                    stage=stage,
                    signature=action.signature,
                    model=config.model_worker,
                )
                runtime.display.log_event("Worker failed to produce FINAL answer; retrying.")
                return
            # Log individual worker action for traceability
            runtime.events.log(
                "worker_action",
                {"action": action.model_dump()},
                step_id=planner_step.step_id,
                agent=meta.agent,
                stage=stage,
                signature=action.signature,
                model=config.model_worker,
            )
            runtime.display.set_panel_text(stage, json.dumps(action.model_dump(), ensure_ascii=False, indent=2))
            async with lock:
                actions.append(action)
                votes[action.signature] = votes.get(action.signature, 0) + 1
                # Build signature mapping for pre-validation
                signature_to_action = {a.signature: a for a in actions}
                leader = runtime.consensus.leader_if_ahead(votes, signature_to_action, planner_step)
                if leader and not stop_event.is_set():
                    runtime.display.log_event(f"Early consensus on {leader}")
                    stop_event.set()

        # Adapt worker count based on strategy and budget constraints
        task_analysis = state.get("task_analysis")
        budget_constrained = state.get("budget_constrained_mode", False)

        if budget_constrained:
            worker_count = 1  # Budget-aware policy forcing single worker
            runtime.display.log_event("Budget constrained - using single worker")
        elif task_analysis:
            from .router import OrchestrationStrategy
            if task_analysis.strategy in [OrchestrationStrategy.DETERMINISTIC, OrchestrationStrategy.SINGLE_SHOT]:
                worker_count = 1  # No voting needed for deterministic/single-shot
                runtime.display.log_event(f"Strategy {task_analysis.strategy} - using single worker")
            else:
                worker_count = config.swarm_size  # Full consensus
        else:
            worker_count = config.swarm_size  # Default

        tasks = [asyncio.create_task(run_worker(idx)) for idx in range(1, worker_count + 1)]
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if stop_event.is_set():
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                break
        await asyncio.gather(*tasks, return_exceptions=True)
        if final_required and not actions:
            state["retry_step"] = True
            runtime.display.log_event("Final answer required but no worker produced FINAL action.")
            # Note: retry count is incremented in verify_apply_node to avoid double counting
            return state
        runtime.metrics.add_votes(sum(votes.values()))
        runtime.display.set_panel_text("AGGREGATE", json.dumps({"votes": votes}, ensure_ascii=False))
        runtime.events.log(
            "voter_batch",
            {"count": len(actions), "votes": votes},
            step_id=planner_step.step_id,
            agent="workers",
            stage="propose",
            model=config.model_worker,
        )
        state["candidates"] = actions
        state["votes"] = votes
        _update_history(state, candidate_signatures=[a.signature for a in actions])
        return state

    return _node


def aggregate_node(config: SwarmConfig):
    def _node(state: GraphState) -> GraphState:
        planner_step = state.get("planner_step")
        runtime = state["runtime"]
        actions = state.get("candidates", [])
        if not planner_step or not actions:
            state["retry_step"] = True
            runtime.display.log_event("No worker actions; re-planning.")
            return state
        result = runtime.consensus.decide(actions)
        runtime.display.log_event(f"Votes: {result.votes}")
        runtime.events.log(
            "consensus",
            {"votes": result.votes},
            step_id=planner_step.step_id,
            stage="consensus",
        )
        state["votes"] = result.votes
        if result.needs_judge:
            selection = _call_judge(state, result.top_candidates, config)
            if not selection:
                runtime.display.log_event("Judge deferred decision; re-planning.")
                state["retry_step"] = True
                return state
            signature = selection.get("selected_signature") if isinstance(selection, dict) else None
            if signature in ("1", "2"):
                state["retry_step"] = True
                runtime.display.log_event("Judge returned index not signature -> retrying.")
                return state
            if not isinstance(signature, str):
                runtime.display.log_event("Judge returned invalid payload; re-planning.")
                state["retry_step"] = True
                return state
            if signature == "none":
                runtime.display.log_event("Judge rejected all candidates; re-planning.")
                state["retry_step"] = True
                return state
            chosen_action = next((c for c in result.top_candidates if c.signature == signature), None)
            if not chosen_action:
                runtime.display.log_event("Judge chose unknown signature; re-planning.")
                state["retry_step"] = True
                return state
            state["chosen_action"] = chosen_action
            state["judge_needed"] = True
            state["retry_step"] = False
            runtime.display.set_panel_text(
                "JUDGE",
                json.dumps({"selected_signature": signature}, ensure_ascii=False, indent=2),
            )
            runtime.events.log(
                "judge_choice",
                {"selected_signature": signature},
                step_id=planner_step.step_id,
                agent="judge",
                stage="judge",
                signature=signature,
                model=config.model_judge,
            )
            _update_history(
                state,
                chosen_signature=signature,
                judge_used=True,
            )
            return state
        if result.selected is None:
            runtime.display.log_event("Consensus produced no winner; re-planning.")
            state["retry_step"] = True
            return state
        state["chosen_action"] = result.selected
        state["judge_needed"] = False
        state["retry_step"] = False
        _update_history(state, chosen_signature=result.selected.signature)
        return state

    return _node


def _call_judge(state: GraphState, candidates: Sequence[Action], config: SwarmConfig) -> Optional[Dict[str, str]]:
    runtime = state["runtime"]
    planner_step = state.get("planner_step")
    if not planner_step:
        return None
    content = json.dumps(
        [
            {
                "signature": candidate.signature,
                "action": candidate.model_dump(mode="json"),
            }
            for candidate in candidates
        ],
        ensure_ascii=False,
        indent=2,
    )
    messages = [
        SystemMessage(
            content=(
                "You are the SwarmMaker judge.\n"
                "Choose exactly ONE of the candidate actions as the best next action.\n"
                "Return ONLY valid JSON with this schema:\n"
                '{ "selected_signature": "<exact candidate signature string>" }\n'
                'OR if both are disqualified: { "selected_signature": "none" }\n'
                "Important: selected_signature MUST match one of the provided candidate signatures exactly."
            )
        ),
        HumanMessage(
            content=(
                f"Planner step goal: {planner_step.step_goal}\n"
                "Candidates (each has an implicit signature = canonical JSON of action_type+args):\n"
                f"{content}\n"
            )
        ),
    ]
    meta = AgentCallMeta(agent="judge", stage="JUDGE", step_id=planner_step.step_id)
    judge_schema = {
        "type": "object",
        "properties": {"selected_signature": {"type": "string"}},
        "required": ["selected_signature"],
        "additionalProperties": False,
    }
    try:
        result = runtime.llm.structured_completion(
            messages,
            meta=meta,
            model=config.model_judge,
            temperature=config.temperature_judge,
            schema_name="JudgeSelection",
            schema=judge_schema,
            parser=lambda data: data,
        )
        return result.content
    except Exception as err:  # pragma: no cover - network
        runtime.display.log_event(f"Judge failed: {err}")
        runtime.events.log(
            "judge_error",
            {"error": str(err)},
            step_id=planner_step.step_id,
            agent="judge",
            stage="judge",
            model=config.model_judge,
        )
        return None


def verify_apply_node(config: SwarmConfig):
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]
        planner_step = state.get("planner_step")
        action = state.get("chosen_action")
        if state.get("abort_reason"):
            return state
        if not planner_step or not action:
            state["retry_step"] = True
            runtime.display.log_event("Missing action; starting new plan.")
            retries = _increment_retries(state)
            if retries >= config.max_retries:
                state["abort_reason"] = "max retries reached"
                state["done"] = True
            return state
        ok, reason = runtime.verifier.verify(planner_step, action)
        if not ok:
            runtime.events.log(
                "verify_reject",
                {"reason": reason, "action": action.model_dump()},
                step_id=planner_step.step_id,
                stage="verify",
                signature=action.signature,
            )
            runtime.display.log_event(f"Verifier rejected action: {reason}")
            state["retry_step"] = True
            retries = _increment_retries(state)
            if retries >= config.max_retries:
                state["abort_reason"] = "max retries reached"
                state["done"] = True
            return state
        runtime.verifier.apply(action, state)
        runtime.events.log(
            "action_applied",
            action.model_dump(),
            step_id=planner_step.step_id,
            stage="verify",
            signature=action.signature,
        )
        runtime.display.set_panel_text("VERIFIER", json.dumps(action.model_dump(), ensure_ascii=False, indent=2))

        # Progress tracking after action applied
        snapshot = runtime.progress.snapshot(state)
        progress_delta = runtime.progress.measure_progress(snapshot)
        runtime.progress.history.append(snapshot)
        runtime.events.log(
            "progress_delta",
            {"delta": progress_delta, "step_id": planner_step.step_id},
            step_id=planner_step.step_id,
        )

        _update_history(state, verifier_passed=True, final_answer=state.get("final_answer"))
        state["retry_step"] = False
        state["planner_step"] = None
        steps_completed = state.get("steps_completed", 0) + 1
        state["steps_completed"] = steps_completed
        runtime.display.update_metrics(
            runtime.metrics.snapshot(
                steps_completed,
                budget_remaining=max(config.max_total_tokens - runtime.metrics.tokens_total(), 0),
            )
        )
        if action.action_type == "FINAL":
            state["final_answer"] = state.get("final_answer") or state.get("draft_answer")
            state["done"] = True
        else:
            state["done"] = False
        return state

    return _node


def _verify_router(state: GraphState) -> str:
    if state.get("abort_reason") or state.get("done"):
        return "final"
    return "check"


def final_node():
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]

        # Check if best-effort finalization triggered
        if state.get("trigger_best_effort_final"):
            runtime.display.log_event("Triggering best-effort finalization...")
            finalization = runtime.finalizer.finalize(state["task"], state, runtime)

            final_answer = finalization.final_answer
            if finalization.is_partial:
                final_answer += f"\n\n[Confidence: {finalization.confidence:.0%}, Method: {finalization.synthesis_method}, Partial: Yes]"

            state["final_answer"] = final_answer
            runtime.events.log("best_effort_final", {
                "confidence": finalization.confidence,
                "method": finalization.synthesis_method,
                "is_partial": finalization.is_partial,
            })
        else:
            # Normal finalization
            final_answer = state.get("final_answer") or state.get("draft_answer") or "No final answer."

        runtime.display.set_panel_text("FINAL", final_answer)
        runtime.display.log_event("Run finished.")
        runtime.events.log("final", {"final_answer": final_answer})
        runtime.langsmith.complete(outputs={"final_answer": final_answer}, error=state.get("abort_reason"))
        return state

    return _node


def run_swarm(*, task: str, config: SwarmConfig, runtime: RuntimeContext) -> RunResult:
    # Task routing - classify task and select strategy
    from .domain_state import DomainStateFactory

    task_analysis = runtime.router.analyze(task)
    runtime.events.log("task_routing", {
        "category": task_analysis.category,
        "strategy": task_analysis.strategy,
        "confidence": task_analysis.confidence,
        "estimated_steps": task_analysis.estimated_steps,
        "requires_creativity": task_analysis.requires_creativity,
        "is_well_defined": task_analysis.is_well_defined,
    })
    runtime.display.log_event(f"Strategy: {task_analysis.strategy} (category: {task_analysis.category})")

    # Initialize domain state
    domain_state = DomainStateFactory.create(task_analysis.category)

    graph = build_graph(config)
    initial_state: GraphState = {
        "task": task,
        "runtime": runtime,
        "task_analysis": task_analysis,
        "domain_state": domain_state,
        "draft_answer": None,
        "history_signatures": [],
        "history": runtime.history,
        "steps_completed": 0,
        "done": False,
        "retry_step": False,
    }
    compiled = graph.compile()
    final_state = asyncio.run(compiled.ainvoke(initial_state, config={"recursion_limit": 100}))
    stats_snapshot = runtime.metrics.snapshot(final_state.get("steps_completed", 0))
    run_stats = RunStats(
        elapsed_s=stats_snapshot.get("elapsed", 0.0),
        llm_calls=runtime.metrics.llm_calls,
        tokens_in=runtime.metrics.tokens_in,
        tokens_out=runtime.metrics.tokens_out,
        retries=runtime.metrics.retries,
        consensus_votes=runtime.metrics.consensus_votes,
        aborted_reason=final_state.get("abort_reason"),
    )
    artifacts = RunArtifacts(
        langsmith_run_url=runtime.langsmith.run_url,
    )
    return RunResult(
        task=task,
        final_answer=final_state.get("final_answer") or final_state.get("draft_answer"),
        steps=list(runtime.history),
        stats=run_stats,
        artifacts=artifacts,
    )


def _voter_prompt(
    *,
    task: str,
    planner_step: PlannerStep,
    domain_state,
    draft: Optional[str],
    voter_index: int,
    style_hint: str,
    show_rationale: bool,
) -> List[Any]:
    rationale_instruction = "Include rationale (<=1 sentence)." if show_rationale else "Do not include rationale."

    # Summarize domain state for context
    domain_summary = ""
    if domain_state and hasattr(domain_state, "model_dump"):
        domain_data = domain_state.model_dump()
        domain_summary = json.dumps(domain_data, ensure_ascii=False)[:300]  # Compact
    else:
        domain_summary = "No progress yet"

    draft_answer = json.dumps(draft, ensure_ascii=False)

    # Determine if FINAL is required
    final_required = planner_step.stop_condition == "done" or "final" in planner_step.step_goal.lower()

    if final_required:
        system_instructions = (
            f"You are SwarmMaker worker #{voter_index}.\n"
            "CRITICAL: The planner has set stop_condition='done' - you MUST produce a FINAL answer.\n"
            "You MUST output ONLY valid JSON matching the Action schema with action_type='FINAL'.\n"
            'Action schema format:\n'
            f'  {{"step_id": {planner_step.step_id}, "action_type": "FINAL", '
            '"args": {"content": "..."}'
            + (', "rationale": "..."' if show_rationale else "")
            + ', "confidence": 0.95}\n'
            "Rules:\n"
            f"- step_id MUST be exactly {planner_step.step_id} (copy this number exactly).\n"
            "- action_type MUST be 'FINAL' - any other type will be REJECTED.\n"
            "- args.content MUST contain your complete, detailed answer to the original task.\n"
            "- Include ALL actual work, calculations, reasoning, and final results in args.content.\n"
            "- Use the notes and context to synthesize a comprehensive final answer.\n"
            "- Do NOT use placeholder text - provide the real, complete answer.\n"
            f"- {rationale_instruction}\n"
            "- confidence should be a number between 0 and 1.\n"
            "- No markdown formatting in the JSON, no extra keys.\n"
        )
    else:
        system_instructions = (
            f"You are SwarmMaker worker #{voter_index}.\n"
            "You MUST output ONLY valid JSON matching the Action schema.\n"
            'Action schema:\n{ "step_id": int, "action_type": "NOTE"|"DO", "args": object'
            + (', "rationale": string' if show_rationale else "")
            + ', "confidence": number }\n'
            "Rules:\n"
            f"- step_id MUST equal {planner_step.step_id} exactly (copy this number).\n"
            "- Choose action_type based on what you're doing:\n"
            '  NOTE: Use when adding observations or intermediate findings.\n'
            '  DO: Use when performing an action or computation.\n\n'
            "- CRITICAL: Show ACTUAL WORK, not descriptions.\n"
            "  GOOD examples:\n"
            '    {{"action_type": "DO", "args": {{"content": "x^2 - 5x + 7 = 2x + 1, so x^2 - 7x + 6 = 0"}}}}\n'
            '    {{"action_type": "NOTE", "args": {{"content": "Factoring: x^2 - 7x + 6 = (x-1)(x-6)"}}}}\n'
            '    {{"action_type": "DO", "args": {{"content": "Solving (x-1)(x-6)=0 gives x=1 or x=6"}}}}\n'
            "  BAD examples (TOO VAGUE - DO NOT DO THIS):\n"
            '    {{"action_type": "DO", "args": {{"content": "Simplify the equation"}}}}\n'
            '    {{"action_type": "NOTE", "args": {{"content": "I will factor the quadratic"}}}}\n'
            '    {{"action_type": "DO", "args": {{"content": "Solve for x"}}}}\n\n'
            "- Show equations, numbers, and results - not intentions or descriptions.\n"
            "- Do NOT use ASK_CLARIFY unless you have truly missing information that prevents any progress.\n"
            f"- {rationale_instruction}\n"
            "- confidence should be between 0 and 1.\n"
            "- No markdown formatting, no extra JSON keys.\n"
        )

    return [
        SystemMessage(content=system_instructions),
        HumanMessage(
            content=(
                f"Task: {task}\n"
                f"Step goal: {planner_step.step_goal}\n"
                f"Stop condition: {planner_step.stop_condition}\n"
                f"Progress so far: {domain_summary}\n"
                f"Draft answer: {draft_answer}\n"
                f"Style hint: {style_hint}\n"
            )
        ),
    ]


def _increment_retries(state: GraphState) -> int:
    history = state.get("history")
    if not history:
        return 0
    last = history[-1]
    updated = last.retries + 1
    history[-1] = last.model_copy(update={"retries": updated})
    return updated


def _update_history(state: GraphState, **updates: Any) -> None:
    history = state.get("history")
    if not history:
        return
    last = history[-1]
    history[-1] = last.model_copy(update=updates)
