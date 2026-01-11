"""IO helpers for SwarmMaker runs."""
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import RunResult, canonical_json


def ensure_log_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class EventLogger:
    """Append-only JSONL logger with buffered writes."""

    def __init__(self, path: Path, buffer_size: int = 10) -> None:
        self.path = path
        self.buffer_size = buffer_size
        self._buffer: List[str] = []
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
        self._buffer.append(canonical_json(entry))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered events to file."""
        if not self._buffer:
            return
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(self._buffer) + "\n")
            self._buffer.clear()
        except OSError as e:
            print(f"[EventLogger] Flush failed: {e}", file=sys.stderr)
            self._buffer.clear()  # Avoid unbounded growth

    def __del__(self) -> None:
        """Flush remaining events on cleanup."""
        self.flush()


def save_json_result(result: RunResult, path: Path) -> bool:
    """Save JSON result to file. Returns True on success."""
    ensure_log_dir(path.parent)
    payload = result.model_dump(mode="python")
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
        return True
    except OSError as e:
        print(f"[save_json_result] Write failed: {e}", file=sys.stderr)
        return False


def save_markdown_result(result: RunResult, path: Path) -> bool:
    """Save markdown result to file. Returns True on success."""
    ensure_log_dir(path.parent)
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(render_result_markdown(result))
        return True
    except OSError as e:
        print(f"[save_markdown_result] Write failed: {e}", file=sys.stderr)
        return False


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
