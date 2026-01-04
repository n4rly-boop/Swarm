"""IO helpers for SwarmMaker runs."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .schemas import RunResult


def ensure_log_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class EventLogger:
    """Append-only JSONL event logger."""

    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_log_dir(self.path.parent)
        if not self.path.exists():
            self.path.touch()

    def log(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "event": event_type,
            "payload": payload or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_json_result(result: RunResult, path: Path) -> None:
    ensure_log_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(result.model_dump_json(indent=2, by_alias=True))


def save_markdown_result(result: RunResult, path: Path) -> None:
    ensure_log_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(render_result_markdown(result))


def render_result_markdown(result: RunResult) -> str:
    """Generate the result.md body."""
    stats = result.stats
    summary_lines = [
        f"- Steps: {len(result.steps)}",
        f"- Elapsed: {stats.elapsed_s:.2f}s",
        f"- LLM calls: {stats.llm_calls}",
        f"- Tokens in/out: {stats.tokens_in}/{stats.tokens_out}",
    ]
    if stats.aborted_reason:
        summary_lines.append(f"- Aborted: {stats.aborted_reason}")
    stats_lines = [
        f"- Tokens in: {stats.tokens_in}",
        f"- Tokens out: {stats.tokens_out}",
        f"- LLM calls: {stats.llm_calls}",
        f"- Retries: {stats.retries}",
        f"- Consensus votes: {stats.consensus_votes}",
    ]
    artifacts_lines = []
    if result.artifacts.langsmith_run_url:
        artifacts_lines.append(f"- LangSmith: {result.artifacts.langsmith_run_url}")
    if result.artifacts.events_path:
        artifacts_lines.append(f"- Events: {result.artifacts.events_path}")

    body = [
        "## Task",
        result.task.strip(),
        "",
        "## Final answer",
        (result.final_answer or "No final answer produced.").strip(),
        "",
        "## Execution summary",
        "\n".join(summary_lines),
        "",
        "## Stats",
        "\n".join(stats_lines + artifacts_lines),
        "",
    ]
    return "\n".join(body)
