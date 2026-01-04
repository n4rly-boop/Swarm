"""LangGraph workflow for SwarmMaker."""
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TypedDict, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError
from langgraph.graph import END, StateGraph

from .callbacks import LiveSwarmDisplay
from .consensus import ConsensusEngine
from .io import EventLogger
from .llm import LLMClient, MetricsTracker
from .schemas import (
    Action,
    AgentCallMeta,
    PlannerStep,
    RunArtifacts,
    RunResult,
    RunStats,
    StepRecord,
    StructuredLLMOutput,
    StructuredParseResult,
    SwarmConfig,
)
from .verify import ActionVerifier

try:  # Optional LangSmith client for root run tracking.
    from langsmith import Client as LangSmithClient
except Exception:  # pragma: no cover - optional dependency at runtime.
    LangSmithClient = None  # type: ignore[assignment]


class GraphState(TypedDict, total=False):
    task: str
    config: SwarmConfig
    runtime: "RuntimeContext"
    step_counter: int
    steps_completed: int
    planner_step: Optional[PlannerStep]
    candidates: List[Action]
    judge_candidates: List[Action]
    chosen_action: Optional[Action]
    needs_judge: bool
    retrying: bool
    done: bool
    final_answer: Optional[str]
    abort_reason: Optional[str]
    history: List[StepRecord]


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
                run = self.client.create_run(
                    name="SwarmMaker",
                    inputs={"task": task},
                    project_name=project_name,
                    run_type="chain",
                    tags=["swarmmaker"],
                    metadata={"task": task},
                )
                self.run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
                if not self.run_id and isinstance(run, dict):
                    self.run_id = run.get("id")
                self.run_url = getattr(run, "url", None) or getattr(run, "dashboard_url", None)
                if not self.run_url and isinstance(run, dict):
                    self.run_url = run.get("url")
            except Exception:
                self.enabled = False
                self.client = None
                self.run_id = None
                self.run_url = None

    def complete(self, outputs: Optional[Dict] = None, error: Optional[str] = None) -> None:
        if not (self.client and self.run_id):
            return
        try:
            self.client.update_run(  # type: ignore[call-arg]
                run_id=self.run_id,
                outputs=outputs or {},
                error=error,
            )
        except Exception:
            pass


def build_graph(config: SwarmConfig) -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node(config))
    graph.add_node("voters", voters_node(config))
    graph.add_node("consensus", consensus_node())
    graph.add_node("judge", judge_node(config))
    graph.add_node("verify", verify_node(config))
    graph.add_node("final", final_node())

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        lambda state: "final" if state.get("abort_reason") else "voters",
        {
            "final": "final",
            "voters": "voters",
        },
    )

    graph.add_edge("voters", "consensus")

    graph.add_conditional_edges(
        "consensus",
        lambda state: "judge" if state.get("needs_judge") else "verify",
        {
            "judge": "judge",
            "verify": "verify",
        },
    )

    graph.add_conditional_edges(
        "judge",
        lambda state: "voters" if state.get("retrying") else "verify",
        {
            "voters": "voters",
            "verify": "verify",
        },
    )

    graph.add_conditional_edges(
        "verify",
        verify_router,
        {
            "planner": "planner",
            "voters": "voters",
            "final": "final",
        },
    )

    graph.add_edge("final", END)
    return graph


def planner_node(config: SwarmConfig):
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]
        metrics = runtime.metrics
        if state.get("done"):
            state["abort_reason"] = state.get("abort_reason") or "already done"
            return state
        if metrics.tokens_total() >= config.max_total_tokens:
            state["abort_reason"] = "token budget exceeded"
            runtime.events.log("abort", {"reason": state["abort_reason"]})
            runtime.display.log_event("Budget exceeded, stopping.")
            state["done"] = True
            return state
        if state.get("steps_completed", 0) >= config.max_steps:
            state["abort_reason"] = "max steps reached"
            runtime.events.log("abort", {"reason": state["abort_reason"]})
            runtime.display.log_event("Max steps reached.")
            state["done"] = True
            return state
        elapsed = runtime.metrics.snapshot(state.get("steps_completed", 0)).get("elapsed", 0.0)
        if elapsed >= config.max_wall_seconds:
            state["abort_reason"] = "wall clock limit reached"
            runtime.events.log("abort", {"reason": state["abort_reason"]})
            runtime.display.log_event("Wall clock limit reached.")
            state["done"] = True
            return state

        step_id = state.get("steps_completed", 0) + 1
        budget_remaining = max(0, config.max_total_tokens - metrics.tokens_total())
        planner_input = {
            "task": state["task"],
            "token_budget_remaining": budget_remaining,
            "swarm_size": config.swarm_size,
            "recent_history": [
                {
                    "step_id": record.step_id,
                    "goal": record.planner_step.step_goal,
                    "chosen": record.chosen_signature,
                }
                for record in state.get("history", [])[-3:]
            ],
        }
        worker_token_hint = max(
            32,
            min(
                512,
                (budget_remaining // max(config.swarm_size, 1)) or 32,
            ),
        )
        schema_hint = json.dumps(
            {
                "step_id": step_id,
                "step_goal": "describe the immediate subtask you want workers to perform next",
                "expected_action_schema": "Action",
                "stop_condition": "continue|done",
                "worker_max_tokens": worker_token_hint,
            },
            ensure_ascii=False,
        )
        examples = (
            '{ "step_id": %d, "step_goal": "outline solution approach", '
            '"expected_action_schema": "Action", "stop_condition": "continue", '
            '"worker_max_tokens": %d }'
            % (step_id, worker_token_hint)
        )
        wrapper_hint = _structured_wrapper_instructions("PlannerStep")
        messages = [
            SystemMessage(
                content=(
                    "You are the SwarmMaker planner.\n"
                    f"{wrapper_hint}\n"
                    "Fields (all required): step_id(int), step_goal(str), expected_action_schema (literal \"Action\"), "
                    "stop_condition (either \"continue\" or \"done\"), worker_max_tokens (int between 16 and 2048 indicating the completion token cap per worker)."
                )
            ),
            HumanMessage(
                content=(
                    f"Task:\n{state['task']}\n\n"
                    f"Current state:\n{json.dumps(planner_input, ensure_ascii=False)}\n\n"
                    f"Recommend a `worker_max_tokens` value <= {worker_token_hint} unless the task clearly requires more detail.\n"
                    f"Schema hint:\n{schema_hint}\n"
                    f"Example:\n{examples}\n"
                    "Remember: respond with raw JSON only; never wrap in markdown fences."
                )
            ),
        ]
        meta = AgentCallMeta(agent="planner", stage="PLANNER", step_id=step_id)
        planner_result = _request_json(
            runtime.llm,
            messages,
            meta=meta,
            model=config.model_planner,
            temperature=config.temperature_planner,
            parser=PlannerStep.model_validate,
            output_schema=PlannerStep.model_json_schema(),
            runtime=runtime,
        )
        planner_step = planner_result.content
        runtime.events.log("planner_step", planner_step.model_dump())
        if planner_result.thinking or planner_result.thinking_tokens is not None:
            runtime.events.log(
                "planner_thinking",
                {
                    "thinking": planner_result.thinking,
                    "thinking_tokens": planner_result.thinking_tokens,
                },
            )
        runtime.display.set_panel_text(
            "PLANNER",
            json.dumps(
                {
                    "thinking": planner_result.thinking,
                    "thinking_tokens": planner_result.thinking_tokens,
                    "output": planner_step.model_dump(),
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
        history = runtime.history
        state["history"] = history
        history.append(
            StepRecord(
                step_id=planner_step.step_id,
                planner_step=planner_step,
                planner_thinking=planner_result.thinking,
                planner_thinking_tokens=planner_result.thinking_tokens,
            )
        )
        state["step_counter"] = step_id
        state["planner_step"] = planner_step
        state["candidates"] = []
        state["judge_candidates"] = []
        state["chosen_action"] = None
        state["needs_judge"] = False
        state["retrying"] = False
        _update_metrics(runtime, state.get("steps_completed", 0))
        return state

    return _node


def voters_node(config: SwarmConfig):
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]
        planner_step = state.get("planner_step")
        if not planner_step:
            state["abort_reason"] = "planner missing"
            runtime.display.log_event("Planner output missing.")
            return state
        if _budget_exhausted(runtime):
            state["abort_reason"] = "token budget exceeded"
            runtime.events.log("abort", {"reason": state["abort_reason"]})
            runtime.display.log_event("Budget exceeded, stopping.")
            state["done"] = True
            return state
        actions: List[Action] = []
        style_pool = [
            "Explore alternative ideas.",
            "Focus on execution details.",
            "Check constraints and blockers.",
            "Draft candidate final answers.",
            "Plan follow-up analysis.",
        ]
        voter_thinking: Dict[int, Optional[str]] = {}
        voter_thinking_tokens: Dict[int, Optional[int]] = {}
        worker_token_cap = max(16, min(planner_step.worker_max_tokens, runtime.config.max_total_tokens))
        for idx in range(1, config.swarm_size + 1):
            stage = f"VOTER#{idx}" if idx <= runtime.display.visible_voters else "VOTERS(+extra)"
            meta = AgentCallMeta(agent=f"voter_{idx}", stage=stage, step_id=planner_step.step_id, voter_id=idx)
            style_hint = style_pool[random.Random(config.seed_base + idx).randrange(len(style_pool))]
            voter_messages = _voter_prompt(
                state["task"],
                planner_step,
                idx,
                config.show_rationale,
                style_hint,
            )
            try:
                action_result = _request_json(
                    runtime.llm,
                    voter_messages,
                    meta=meta,
                    model=config.model_worker,
                    temperature=config.temperature_worker,
                    parser=Action.model_validate,
                    output_schema=Action.model_json_schema(),
                    runtime=runtime,
                    max_output_tokens=worker_token_cap,
                )
            except Exception as err:
                runtime.metrics.increment_retry()
                runtime.events.log(
                    "worker_failed",
                    {
                        "voter_id": idx,
                        "step_id": planner_step.step_id,
                        "error": str(err),
                    },
                )
                runtime.display.set_panel_text(stage, f"Worker failed after retries: {err}")
                continue
            action = action_result.content
            runtime.display.set_panel_text(
                stage,
                json.dumps(
                    {
                        "thinking": action_result.thinking,
                        "thinking_tokens": action_result.thinking_tokens,
                        "output": action.model_dump(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            if action_result.thinking is not None:
                voter_thinking[idx] = action_result.thinking
            if action_result.thinking_tokens is not None:
                voter_thinking_tokens[idx] = action_result.thinking_tokens
            actions.append(action)
        runtime.events.log("voter_batch", {"count": len(actions), "step_id": planner_step.step_id})
        state["candidates"] = actions
        _update_history(
            state,
            candidates_signatures=[action.signature for action in actions],
            voter_thinking=voter_thinking or {},
            voter_thinking_tokens=voter_thinking_tokens or {},
        )
        return state

    return _node


def consensus_node():
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]
        actions = state.get("candidates", [])
        result = runtime.consensus.decide(actions)
        runtime.metrics.add_votes(sum(result.votes.values()))
        signatures = list(result.votes.items())
        runtime.events.log("consensus", {"votes": signatures})
        runtime.display.log_event(f"Consensus votes: {signatures}")
        state["needs_judge"] = result.needs_judge
        state["judge_candidates"] = result.top_candidates
        if result.needs_judge:
            runtime.display.set_panel_text("JUDGE", "Awaiting decision between top candidates.")
        else:
            runtime.display.set_panel_text("JUDGE", "Consensus reached without judge.")
        if not result.needs_judge:
            state["chosen_action"] = result.selected
        return state

    return _node


def judge_node(config: SwarmConfig):
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]
        if _budget_exhausted(runtime):
            state["abort_reason"] = "token budget exceeded"
            runtime.events.log("abort", {"reason": state["abort_reason"]})
            runtime.display.log_event("Budget exceeded before judge call.")
            state["done"] = True
            return state
        candidates = state.get("judge_candidates", [])
        planner_step = state.get("planner_step")
        if not candidates or not planner_step:
            state["retrying"] = True
            return state
        content = json.dumps([c.model_dump() for c in candidates], ensure_ascii=False, indent=2)
        wrapper_hint = _structured_wrapper_instructions("JudgeSelection", include_example=False)
        messages = [
            SystemMessage(
                content="You are the SwarmMaker judge. Choose the better JSON action and wrap it in the structured schema.\n"
                f"{wrapper_hint}\n"
                'Your `output` object must look like {"selected_signature": "<signature-or-none>"} where selecting "none" requests another vote.'
            ),
            HumanMessage(
                content=(
                    f"Planner step: {planner_step.step_goal}\n"
                    f"Candidates:\n{content}\n"
                    "Remember: respond with raw JSON only; do NOT wrap in markdown."
                )
            ),
        ]
        meta = AgentCallMeta(agent="judge", stage="JUDGE", step_id=planner_step.step_id)
        judge_schema = {
            "type": "object",
            "properties": {
                "selected_signature": {
                    "type": "string",
                    "description": "Signature of the chosen action, or the literal string \"none\" to request another vote.",
                }
            },
            "required": ["selected_signature"],
            "additionalProperties": False,
        }
        response = _request_json(
            runtime.llm,
            messages,
            meta=meta,
            model=config.model_judge,
            temperature=config.temperature_judge,
            parser=lambda payload: payload,
            output_schema=judge_schema,
            runtime=runtime,
        )
        selection = response.content
        signature = selection.get("selected_signature") if isinstance(selection, dict) else None
        if response.thinking or response.thinking_tokens is not None:
            runtime.events.log(
                "judge_thinking",
                {"thinking": response.thinking, "thinking_tokens": response.thinking_tokens},
            )
        if signature == "none":
            runtime.metrics.increment_retry()
            runtime.events.log("judge_none", {"step_id": planner_step.step_id})
            runtime.display.log_event("Judge rejected both candidates; retrying voters.")
            runtime.display.set_panel_text("JUDGE", "No selection; requesting new votes.")
            state["retrying"] = True
            state["chosen_action"] = None
            _update_history(
                state,
                judge_thinking=response.thinking,
                judge_thinking_tokens=response.thinking_tokens,
            )
            return state
        state["retrying"] = False
        selected = next((c for c in candidates if c.signature == signature), None)
        if not selected:
            runtime.display.log_event("Judge selected unknown signature, retrying.")
            state["retrying"] = True
            runtime.metrics.increment_retry()
            return state
        state["chosen_action"] = selected
        runtime.display.set_panel_text(
            "JUDGE",
            json.dumps(
                {
                    "thinking": response.thinking,
                    "thinking_tokens": response.thinking_tokens,
                    "output": {"selected_signature": selected.signature},
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
        _update_history(
            state,
            judge_used=True,
            chosen_signature=selected.signature,
            judge_thinking=response.thinking,
            judge_thinking_tokens=response.thinking_tokens,
        )
        return state

    return _node


def verify_node(config: SwarmConfig):
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]
        planner_step = state.get("planner_step")
        chosen = state.get("chosen_action")
        if not planner_step or not chosen:
            state["retrying"] = True
            runtime.metrics.increment_retry()
            runtime.events.log("verify_missing", {"step": planner_step.step_id if planner_step else None})
            runtime.display.log_event("Verifier missing action; retrying voters.")
            _increment_history_retry(state)
            last_record = _last_record(state)
            current_retries = last_record.retries if last_record else 0
            if current_retries >= config.max_retries:
                state["retrying"] = False
                state["abort_reason"] = "max retries reached"
                runtime.events.log("abort", {"reason": state["abort_reason"]})
                runtime.display.log_event("Max retries reached; aborting.")
                state["done"] = True
            return state
        ok, reason = runtime.verifier.verify(
            planner_step,
            chosen,
            dry_run=config.dry_run,
        )
        if not ok:
            runtime.metrics.increment_retry()
            runtime.events.log("verify_fail", {"reason": reason})
            runtime.display.log_event(f"Verifier rejected action: {reason}")
            state["retrying"] = True
            _increment_history_retry(state)
            last_record = _last_record(state)
            current_retries = last_record.retries if last_record else 0
            if current_retries >= config.max_retries:
                state["retrying"] = False
                state["abort_reason"] = "max retries reached"
                runtime.events.log("abort", {"reason": state["abort_reason"]})
                runtime.display.log_event("Max retries reached; aborting.")
                state["done"] = True
            return state
        runtime.events.log("action_applied", {"signature": chosen.signature})
        runtime.display.log_event(f"Action applied: {chosen.signature}")
        runtime.display.set_panel_text("VERIFIER", f"Accepted {chosen.signature}")
        _update_history(state, chosen_signature=chosen.signature, verifier_passed=True)
        state["steps_completed"] = state.get("steps_completed", 0) + 1
        state["retrying"] = False
        state["done"] = planner_step.stop_condition == "done" or chosen.action_type == "final_answer"
        if chosen.action_type == "final_answer":
            state["final_answer"] = str(chosen.args.get("content") or chosen.args)
        _update_metrics(runtime, state.get("steps_completed", 0))
        return state

    return _node


def verify_router(state: GraphState) -> str:
    if state.get("abort_reason"):
        return "final"
    if state.get("done"):
        return "final"
    if state.get("retrying"):
        return "voters"
    return "planner"


def final_node():
    def _node(state: GraphState) -> GraphState:
        runtime = state["runtime"]
        final_answer = state.get("final_answer") or "No final answer."
        runtime.display.set_panel_text("FINAL", final_answer)
        runtime.display.log_event("Finalized run.")
        runtime.langsmith.complete(outputs={"final_answer": final_answer}, error=state.get("abort_reason"))
        return state

    return _node


def _last_record(state: GraphState) -> Optional[StepRecord]:
    history = state.get("history") or []
    return history[-1] if history else None


def _update_history(state: GraphState, **updates) -> None:
    history = state.get("history") or []
    if not history:
        return
    last = history[-1]
    history[-1] = last.model_copy(update=updates)


def _increment_history_retry(state: GraphState) -> None:
    history = state.get("history") or []
    if not history:
        return
    last = history[-1]
    retries = last.retries + 1
    history[-1] = last.model_copy(update={"retries": retries})


def _update_metrics(runtime: RuntimeContext, steps: int) -> None:
    budget_remaining = max(
        runtime.config.max_total_tokens - runtime.metrics.tokens_total(),
        0,
    )
    snapshot = runtime.metrics.snapshot(steps, budget_remaining=budget_remaining)
    runtime.display.update_metrics(snapshot)


def _budget_exhausted(runtime: RuntimeContext) -> bool:
    return runtime.metrics.tokens_total() >= runtime.config.max_total_tokens


def _extract_json_payload(text: str) -> str:
    """Best-effort extraction of JSON body by removing fences and headers."""

    stripped = text.strip()
    if not stripped:
        return stripped
    # Remove stray provider control tokens.
    for token in ("<|fim_middle|>", "<|fim_end|>", "<|assistant|>", "<|user|>"):
        stripped = stripped.replace(token, "")
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]  # drop opening fence
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") or stripped.startswith("["):
        candidate = _slice_first_json_blob(stripped)
        if candidate:
            return candidate
    candidate = _slice_first_json_blob(stripped)
    return candidate or stripped


def _slice_first_json_blob(text: str) -> Optional[str]:
    """Return first balanced JSON object/array substring if possible."""

    stack: List[str] = []
    start: Optional[int] = None
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if start is None:
            if ch in ("{", "["):
                start = idx
                stack.append("}" if ch == "{" else "]")
            continue
        if not stack:
            break
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        prev_char = text[idx - 1] if idx > 0 else ""
        if ch == '"' and prev_char != "\\":
            in_string = not in_string
            continue
        if in_string:
            continue
        expected = stack[-1]
        if ch == expected:
            stack.pop()
            if not stack:
                return text[start : idx + 1].strip()
        elif ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
    return None


def _extract_missing_fields(err: ValidationError) -> List[str]:
    fields: List[str] = []
    for detail in err.errors():
        if detail.get("type") == "missing":
            loc = detail.get("loc")
            if not loc:
                continue
            name = ".".join(str(part) for part in loc)
            fields.append(name)
    return fields


def _structured_wrapper_instructions(schema_name: str, *, include_example: bool = True) -> str:
    """Shared hint describing the enforced structured output wrapper."""

    hint = (
        "Return ONLY valid JSON, no prose or markdown fences.\n"
        "Your response MUST be a JSON object with keys `thinking` (string, optional), "
        "`thinking_tokens` (integer, optional), and `output` (object). "
        f"The `output` object MUST strictly match the {schema_name} schema with valid JSON."
    )
    if include_example:
        example = {
            "thinking": "short hidden reasoning",
            "thinking_tokens": 24,
            "output": {
                "placeholder": "replace with schema-compliant fields"
            },
        }
        hint += f" Example: {json.dumps(example, ensure_ascii=False)}."
    hint += " No extra keys, markdown fences, or commentary are allowed."
    return hint


def _build_structured_schema(output_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON schema for StructuredLLMOutput with injected output schema."""

    return {
        "type": "object",
        "properties": {
            "thinking": {
                "type": "string",
                "description": "Optional hidden reasoning.",
            },
            "thinking_tokens": {
                "type": "integer",
                "minimum": 0,
                "description": "Token count spent inside thinking.",
            },
            "output": output_schema,
        },
        "required": ["output"],
        "additionalProperties": False,
    }


def _structured_response_format(meta: AgentCallMeta, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return OpenAI response_format payload for strict schema enforcement."""

    schema_name = _sanitize_schema_name(meta)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        },
    }


def _sanitize_schema_name(meta: AgentCallMeta) -> str:
    candidate = f"{meta.agent}_{meta.stage}_schema"
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", candidate)
    sanitized = sanitized.strip("_")[:64]
    return sanitized or "SwarmMakerSchema"


TParsed = TypeVar("TParsed")


def _request_json(
    llm: LLMClient,
    messages: Sequence,
    *,
    meta: AgentCallMeta,
    model: str,
    temperature: float,
    parser: Callable[[Any], TParsed],
    output_schema: Dict[str, Any],
    runtime: RuntimeContext,
    max_output_tokens: Optional[int] = None,
) -> StructuredParseResult[TParsed]:
    attempts = 2
    wrapper_schema = _build_structured_schema(output_schema)
    schema_directive = SystemMessage(
        content=(
            "STRICT JSON SCHEMA (respond with JSON only, beginning with `{`):\n"
            f"{json.dumps(wrapper_schema, ensure_ascii=False, indent=2)}"
        )
    )
    response_format = _structured_response_format(meta, wrapper_schema)
    base_messages = [schema_directive, *messages]
    prompt_messages = list(base_messages)
    for attempt in range(attempts):
        try:
            text = llm.complete(
                prompt_messages,
                meta=meta,
                model=model,
                temperature=temperature,
                response_format=response_format,
                max_output_tokens=max_output_tokens,
            )
        except Exception as err:
            runtime.metrics.increment_retry()
            runtime.events.log(
                "llm_call_error",
                {
                    "agent": meta.agent,
                    "stage": meta.stage,
                    "error": str(err),
                },
            )
            if attempt == attempts - 1:
                raise
            runtime.display.log_event(f"LLM error from {meta.agent}: {err}. Retrying.")
            prompt_messages = list(base_messages)
            continue
        try:
            payload = _extract_json_payload(text)
            structured = StructuredLLMOutput.model_validate_json(payload)
            parsed = parser(structured.output)
            return StructuredParseResult(
                content=parsed,
                thinking=structured.thinking,
                thinking_tokens=structured.thinking_tokens,
            )
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as err:
            runtime.metrics.increment_retry()
            correction_hint = "Previous output was invalid. Reply with ONLY valid JSON following the structured schema with `thinking`, `thinking_tokens`, and `output` keys."
            if isinstance(err, ValidationError):
                missing_fields = _extract_missing_fields(err)
                if missing_fields:
                    correction_hint = (
                        "Previous output was invalid because these fields were missing or malformed: "
                        f"{', '.join(sorted(set(missing_fields)))}. "
                        "Return valid JSON with all required fields populated (e.g., include `output.step_id`)."
                    )
            runtime.events.log(
                "json_parse_error",
                {
                    "agent": meta.agent,
                    "stage": meta.stage,
                    "error": str(err),
                    "snippet": text[:200],
                },
            )
            if attempt == attempts - 1:
                raise
            runtime.display.log_event(f"Invalid JSON from {meta.agent}, retrying.")
            prompt_messages = list(base_messages) + [
                HumanMessage(content=correction_hint)
            ]
    raise RuntimeError("JSON parsing failed")


def _voter_prompt(task: str, planner_step: PlannerStep, voter_index: int, show_rationale: bool, style_hint: str):
    rationale_instruction = (
        "Include a short rationale sentence." if show_rationale else "Omit the rationale field."
    )
    wrapper_hint = _structured_wrapper_instructions("Action")
    messages = [
        SystemMessage(
            content=(
                f"You are SwarmMaker worker #{voter_index}. Propose a single actionable JSON Action.\n"
                "Fields: step_id, action_type, args (object), optional rationale (<=1 sentence), confidence (0-1). "
                "Use the exact field names shown; do not invent alternatives like `confident`.\n"
                f"{wrapper_hint} Put any private reasoning inside `thinking`."
            )
        ),
        HumanMessage(
            content=(
                f"Task: {task}\n"
                f"Step goal: {planner_step.step_goal}\n"
                f"Stop condition: {planner_step.stop_condition}\n"
                f"Planner token cap per worker: {planner_step.worker_max_tokens} completion tokens (hard limit).\n"
                f"Style hint: {style_hint}\n"
                f"{rationale_instruction}\n"
                f"Set `output.step_id` to {planner_step.step_id} exactly. "
                "Place all required keys even if obvious. Respond with valid JSON only."
            )
        ),
    ]
    return messages


def run_swarm(
    *,
    task: str,
    config: SwarmConfig,
    runtime: RuntimeContext,
) -> RunResult:
    graph = build_graph(config)
    history_holder = runtime.history
    initial_state: GraphState = {
        "task": task,
        "config": config,
        "runtime": runtime,
        "history": history_holder,
        "steps_completed": 0,
        "step_counter": 0,
        "done": False,
    }
    compiled = graph.compile()
    final_state = compiled.invoke(initial_state)
    stats = runtime.metrics
    snapshot = stats.snapshot(final_state.get("steps_completed", 0))
    run_stats = RunStats(
        elapsed_s=snapshot.get("elapsed", 0.0),
        llm_calls=stats.llm_calls,
        tokens_in=stats.tokens_in,
        tokens_out=stats.tokens_out,
        retries=stats.retries,
        consensus_votes=stats.consensus_votes,
        aborted_reason=final_state.get("abort_reason"),
    )
    artifacts = RunArtifacts(
        langsmith_run_url=runtime.langsmith.run_url,
    )
    result = RunResult(
        task=task,
        final_answer=final_state.get("final_answer"),
        steps=list(history_holder),
        stats=run_stats,
        artifacts=artifacts,
    )
    return result
