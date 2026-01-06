"""Data contracts for the MAKER-style SwarmMaker runtime."""
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field


def canonical_json(obj: Any) -> str:
    """Return canonical JSON for deterministic signatures and logs."""

    def default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=default)


class StructuredMode(str, Enum):
    """Supported response-format modes for providers."""

    json_schema = "json_schema"
    json_object = "json_object"


class DecompositionProposal(BaseModel):
    """LLM-proposed decomposition of a problem into two subproblems."""

    model_config = ConfigDict(extra="forbid")

    subproblem_a: str = Field(min_length=1, description="First subproblem to solve.")
    subproblem_b: str = Field(min_length=1, description="Second subproblem to solve.")
    compose_fn: str = Field(min_length=1, description="Instructions to combine results.")
    is_atomic: bool = Field(
        description="True when the problem cannot be decomposed further and should be solved directly.",
    )
    rationale: str = Field(min_length=1, description="Why this decomposition is appropriate.")

    @property
    def signature(self) -> str:
        return canonical_json(
            {
                "subproblem_a": self.subproblem_a,
                "subproblem_b": self.subproblem_b,
                "compose_fn": self.compose_fn,
                "is_atomic": self.is_atomic,
            }
        )


class AtomicSolution(BaseModel):
    """Atomic solver output containing the proposed answer."""

    model_config = ConfigDict(extra="forbid")

    solution: str = Field(min_length=1, description="Candidate solution string.")
    confidence: float = Field(ge=0.0, le=1.0)
    work_shown: str = Field(min_length=1, description="Audit trail for the answer.")

    @property
    def signature(self) -> str:
        return canonical_json({"solution": self.solution.strip()})


class ProblemFact(BaseModel):
    """Compact fact captured from an accepted atomic solution."""

    model_config = ConfigDict(extra="forbid")

    problem: str
    solution: str
    work_shown: str
    confidence: float = Field(ge=0.0, le=1.0)
    depth: int = 0


class SupportPoint(BaseModel):
    """Structured support point describing a computed result."""

    model_config = ConfigDict(extra="forbid")

    label: str
    x: float = Field(default=0.0, description="X coordinate value.")
    y: float = Field(default=0.0, description="Y coordinate value.")
    statement: str = Field(default="", description="Description of this point.")


class FinalSupport(BaseModel):
    """Optional support payload carried with the final answer."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(default="", description="Brief summary of the support evidence.")
    equations: List[str] = Field(default_factory=list)
    points: List[SupportPoint] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """Final answer payload returned by the composer stage."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, description="Direct response to the original task.")
    confidence: float = Field(ge=0.0, le=1.0)
    support: FinalSupport = Field(default_factory=FinalSupport, description="Supporting evidence.")


class RequirementStatus(BaseModel):
    """Status of a single requirement from the task."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(description="The requirement extracted from the task.")
    status: Literal["ADDRESSED", "MISSING"] = Field(description="Whether this requirement is addressed.")
    reason: str = Field(description="Why the requirement is addressed or missing.")


class CompletenessResult(BaseModel):
    """Result of checking if all task requirements are addressed."""

    model_config = ConfigDict(extra="forbid")

    requirements: List[RequirementStatus] = Field(default_factory=list)
    complete: bool = Field(description="True if all requirements are addressed.")
    missing_work: List[str] = Field(
        default_factory=list,
        description="Specific tasks to perform to fill gaps, if any.",
    )


class TaskState(BaseModel):
    """Minimal state snapshot shared with the orchestrator."""

    task: str
    decomposition_tree: Dict[str, Any] = Field(default_factory=dict)
    solved_subproblems: Dict[str, str] = Field(default_factory=dict)
    facts: Dict[str, ProblemFact] = Field(default_factory=dict)
    current_problem: str
    depth: int = 0
    notes: List[str] = Field(default_factory=list)
    draft_answer: Optional[str] = None
    progress_version: int = 0
    progress_hash: str = ""


class StepTrace(BaseModel):
    """History entry captured for each decomposition or atomic step."""

    step_id: int
    kind: Literal["decomposition", "atomic"]
    problem: str
    depth: int
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    chosen: Optional[Dict[str, Any]] = None
    votes: Dict[str, int] = Field(default_factory=dict)
    notes: Optional[str] = None
    rejections: List[Dict[str, Any]] = Field(default_factory=list)


class RunStats(BaseModel):
    elapsed_s: float = 0.0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retries: int = 0
    consensus_votes: int = 0
    aborted_reason: Optional[str] = None


class RunArtifacts(BaseModel):
    result_md_path: Optional[Path] = None
    result_json_path: Optional[Path] = None
    events_path: Optional[Path] = None
    langsmith_run_url: Optional[str] = None


class RunResult(BaseModel):
    task: str
    final_answer: Optional[str]
    final_payload: Optional[FinalAnswer] = None
    steps: List[StepTrace]
    stats: RunStats
    artifacts: RunArtifacts
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SwarmConfig(BaseModel):
    """Runtime configuration for the MAKER orchestrator."""

    model_decomposer: str
    model_solver: str
    batch_size: int = 5
    ahead_by: int = 2
    max_rounds: int = 10
    max_depth: int = 6
    prefer_atomic_min_samples: int = 3
    prefer_atomic_ratio: float = 0.5
    max_total_tokens: int = 50_000
    timeout_seconds: int = 60
    temperature_decomposer: float = 0.3
    temperature_solver: float = 0.8
    temperature_composer: float = 0.2
    dry_run: bool = False
    log_dir: Path = Field(default_factory=lambda: Path("runs"))
    structured_mode: StructuredMode = StructuredMode.json_schema
    max_stagnant_rounds: int = 2
    max_non_final_steps: int = 4
    max_work_shown_chars: int = 1200
    direct_attempt_batch_size: int = 2
    direct_attempt_rounds: int = 1
    stagnation_temperature_scale: float = 0.6
    progress_summary_limit: int = 5
    state_summary_char_limit: int = 1200


@dataclass
class AgentCallMeta:
    """Metadata for tagging LLM calls."""

    agent: str
    stage: str
    step_id: int
    voter_id: Optional[int] = None


T = TypeVar("T")


@dataclass
class StructuredParseResult(Generic[T]):
    """Return type for structured LLM responses."""

    content: T
    raw_text: str


class SchemaValidationError(RuntimeError):
    """Raised when provider violates the agreed schema."""

    def __init__(self, *, agent: str, stage: str, payload: str, error: Exception):
        super().__init__("Provider did not honor structured output")
        self.agent = agent
        self.stage = stage
        self.payload = payload
        self.error = error
