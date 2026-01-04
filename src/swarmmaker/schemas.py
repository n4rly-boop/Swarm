"""Data models for SwarmMaker."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PlannerStep(BaseModel):
    """Structured response from the planner agent."""

    step_id: int
    step_goal: str = Field(min_length=1)
    expected_action_schema: Literal["Action"] = "Action"
    stop_condition: Literal["continue", "done"] = "continue"


class Action(BaseModel):
    """Action proposed by a worker/voter."""

    step_id: int
    action_type: str
    args: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @property
    def signature(self) -> str:
        # Signature used for consensus and loop prevention
        args_repr = repr(sorted(self.args.items()))
        return f"{self.step_id}:{self.action_type}:{args_repr}"


class StepRecord(BaseModel):
    """History entry for a micro-step."""

    step_id: int
    planner_step: PlannerStep
    candidates_signatures: List[str] = Field(default_factory=list)
    chosen_signature: Optional[str] = None
    judge_used: bool = False
    retries: int = 0
    verifier_passed: bool = False


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
    swarm_size: int = 5
    ahead_by: int = 2
    max_steps: int = 20
    max_retries: int = 3
    max_total_tokens: int = 20_000
    max_cost_usd: Optional[float] = None
    max_wall_seconds: int = 180
    timeout_seconds: int = 60
    temperature_planner: float = 0.3
    temperature_worker: float = 0.8
    temperature_judge: float = 0.2
    seed_base: int = 42
    show_rationale: bool = False
    dry_run: bool = False
    log_dir: Path = Field(default_factory=lambda: Path("runs"))
    project_name: str = "swarmmaker-mvp"


@dataclass
class AgentCallMeta:
    agent: str
    stage: str
    step_id: int
    voter_id: Optional[int] = None

