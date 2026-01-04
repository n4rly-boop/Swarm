"""Simple CLI reporter for tracking MAKER progress."""
from __future__ import annotations

from typing import Callable, Dict

from .schemas import SwarmConfig


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
            f"discriminator={config.model_discriminator}"
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
