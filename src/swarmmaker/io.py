"""IO helpers for SwarmMaker runs."""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import RunResult, canonical_json


def ensure_log_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class EventLogger:
    """Append-only JSONL logger with canonical serialization."""

    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_log_dir(self.path.parent)
        if not self.path.exists():
            self.path.touch()

    def log(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        step_id: Optional[int] = None,
        agent: Optional[str] = None,
        stage: Optional[str] = None,
        signature: Optional[str] = None,
        message: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "step_id": step_id,
            "agent": agent,
            "stage": stage,
            "signature": signature,
            "message": message,
            "model": model,
            "payload": payload or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(entry) + "\n")


def save_json_result(result: RunResult, path: Path) -> None:
    ensure_log_dir(path.parent)
    payload = result.model_dump(mode="python")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(payload))


def save_markdown_result(result: RunResult, path: Path) -> None:
    ensure_log_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(render_result_markdown(result))


def render_result_markdown(result: RunResult) -> str:
    stats = result.stats
    lines = [
        "## Task",
        result.task.strip(),
        "",
        "## Final Answer",
        (result.final_answer or "No final answer produced.").strip(),
        "",
        "## Stats",
        f"- Steps: {len(result.steps)}",
        f"- Elapsed: {stats.elapsed_s:.2f}s",
        f"- LLM calls: {stats.llm_calls}",
        f"- Tokens in/out: {stats.tokens_in}/{stats.tokens_out}",
        f"- Retries: {stats.retries}",
        f"- Consensus votes: {stats.consensus_votes}",
    ]
    if stats.aborted_reason:
        lines.append(f"- Aborted: {stats.aborted_reason}")
    if result.artifacts.langsmith_run_url:
        lines.append(f"- LangSmith: {result.artifacts.langsmith_run_url}")
    if result.artifacts.events_path:
        lines.append(f"- Events: {result.artifacts.events_path}")
    if result.artifacts.result_json_path:
        lines.append(f"- Result JSON: {result.artifacts.result_json_path}")
    return "\n".join(lines) + "\n"
