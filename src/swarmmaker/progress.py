"""Simple CLI reporter for tracking MAKER progress."""


import hashlib
from typing import Any, Callable, Dict

from .schemas import SwarmConfig, TaskState, canonical_json


class RunReporter:
    """Streams textual updates about steps and metrics."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit

    def start(self, task: str, config: SwarmConfig) -> None:
        self.emit("")
        self.emit("=== SwarmMaker MAKER Run ===")
        self.emit(f"Task: {task}")
        self.emit(
            "Models: "
            f"decomposer={config.model_decomposer}, "
            f"solver={config.model_solver}, "
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
            "  tokens: "
            f"{int(snapshot.get('tokens_in', 0))}/{int(snapshot.get('tokens_out', 0))} "
            f"| llm_calls: {int(snapshot.get('llm_calls', 0))} "
            f"| votes: {int(snapshot.get('consensus_votes', 0))} "
            f"| retries: {int(snapshot.get('retries', 0))} "
            f"| steps logged: {int(snapshot.get('steps', 0))} "
            f"| budget remaining: {int(snapshot.get('budget_remaining', 0))}"
        )


class ProgressTracker:
    """Tracks whether the run state is making meaningful progress."""

    def __init__(self, *, max_stagnant_rounds: int) -> None:
        self.max_stagnant_rounds = max_stagnant_rounds
        self._last_hash = ""
        self._stagnant_rounds = 0

    def record(self, state: TaskState) -> Dict[str, Any]:
        payload = {
            "facts": {key: fact.solution for key, fact in state.facts.items()},
            "draft": state.draft_answer,
            "notes": state.notes[-5:],
            "solved": state.solved_subproblems.copy(),
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
