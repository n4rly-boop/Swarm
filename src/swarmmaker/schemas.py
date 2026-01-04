"""Data contracts and helpers for SwarmMaker."""
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field


def canonical_json(obj: Any) -> str:
    """Return the canonical serialized representation used across the system."""

    def default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=default)


class StructuredMode(str, Enum):
    json_schema = "json_schema"
    json_object = "json_object"


class PlannerStep(BaseModel):
    """Planner output describing the next worker goal."""

    step_id: int
    step_goal: str = Field(min_length=1, description="Concrete instruction for workers.")
    stop_condition: Literal["continue", "done"] = "continue"
    worker_max_tokens: int = Field(
        ge=16,
        le=2048,
        description="Max completion tokens for each worker proposal.",
    )


ActionType = Literal["FINAL", "NOTE", "ASK_CLARIFY", "DO"]


class Action(BaseModel):
    """Worker action proposal."""

    step_id: int
    action_type: ActionType
    args: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = Field(
        default=None,
        max_length=280,
        description="Optional single sentence rationale.",
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @property
    def signature(self) -> str:
        """Canonical signature used for deduping, logging, and consensus."""

        return canonical_json({"action_type": self.action_type, "args": self.args})


class StepRecord(BaseModel):
    """History entry shown in result artifacts."""

    step_id: int
    planner_step: PlannerStep
    candidate_signatures: List[str] = Field(default_factory=list)
    chosen_signature: Optional[str] = None
    judge_used: bool = False
    retries: int = 0
    verifier_passed: bool = False
    notes_snapshot: List[str] = Field(default_factory=list)
    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None


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
    steps: List[StepRecord]
    stats: RunStats
    artifacts: RunArtifacts
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SwarmConfig(BaseModel):
    """Runtime configuration for the swarm."""

    model_planner: str
    model_worker: str
    model_judge: str
    swarm_size: int = 4
    ahead_by: int = 2
    max_steps: int = 8
    max_retries: int = 2
    max_total_tokens: int = 20_000
    max_wall_seconds: int = 180
    timeout_seconds: int = 60
    temperature_planner: float = 0.3
    temperature_worker: float = 0.5
    temperature_judge: float = 0.2
    seed_base: int = 42
    show_rationale: bool = False
    dry_run: bool = False
    stream: bool = True
    log_dir: Path = Field(default_factory=lambda: Path("runs"))
    project_name: str = "swarmmaker-mvp"
    structured_mode: StructuredMode = StructuredMode.json_schema


@dataclass
class AgentCallMeta:
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
