"""Command-line entrypoint for SwarmMaker."""
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .callbacks import LiveSwarmDisplay
from .consensus import ConsensusEngine
from .config import settings
from .graph import LangSmithManager, RuntimeContext, run_swarm
from .io import EventLogger, ensure_log_dir, save_json_result, save_markdown_result
from .llm import LLMClient, MetricsTracker
from .schemas import (
    RunArtifacts,
    RunResult,
    RunStats,
    StructuredMode,
    SwarmConfig,
    canonical_json,
)
from .verify import ActionVerifier

app = typer.Typer(add_completion=False, help="SwarmMaker multi-agent CLI.")

DEFAULT_PLANNER_MODEL = "google/gemini-2.5-flash-preview-09-2025"
DEFAULT_WORKER_MODEL = "qwen/qwen2.5-coder-7b-instruct"
DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash-preview-09-2025"


@app.command()
def main(
    task: str = typer.Argument(..., help="User task or prompt for the swarm."),
    model_planner: str = typer.Option(DEFAULT_PLANNER_MODEL, "--model-planner", show_default=True),
    model_worker: str = typer.Option(DEFAULT_WORKER_MODEL, "--model-worker", show_default=True),
    model_judge: str = typer.Option(DEFAULT_JUDGE_MODEL, "--model-judge", show_default=True),
    swarm_size: int = typer.Option(5, "--swarm-size"),
    ahead_by: int = typer.Option(2, "--ahead-by", help="Early stop K votes ahead."),
    max_steps: int = typer.Option(15, "--max-steps"),
    max_retries: int = typer.Option(2, "--max-retries"),
    max_total_tokens: int = typer.Option(50_000, "--max-total-tokens"),
    max_wall_seconds: int = typer.Option(180, "--max-wall-seconds"),
    timeout_seconds: int = typer.Option(60, "--timeout-seconds"),
    temperature_planner: float = typer.Option(0.3, "--temperature-planner"),
    temperature_worker: float = typer.Option(0.8, "--temperature-worker"),
    temperature_judge: float = typer.Option(0.2, "--temperature-judge"),
    seed_base: int = typer.Option(42, "--seed", "--seed-base", help="Seed for stochastic prompts.", show_default=True),
    show_rationale: bool = typer.Option(False, "--show-rationale/--hide-rationale", show_default=True),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate deterministic mock outputs."),
    structured_mode: StructuredMode = typer.Option(
        StructuredMode.json_schema,
        "--structured-mode",
        case_sensitive=False,
        help="Provider response_format enforcement mode.",
    ),
    stream: bool = typer.Option(True, "--stream/--no-stream", show_default=True),
    log_dir: Optional[Path] = typer.Option(
        None, "--log-dir", help="Directory for logs (defaults to runs/<timestamp>)."
    ),
    project_name: str = typer.Option("swarmmaker-mvp", "--project-name"),
) -> None:
    """Run the SwarmMaker multi-agent flow for the provided task."""

    _ = settings
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    target_log_dir = log_dir or Path("runs") / timestamp
    target_log_dir = ensure_log_dir(target_log_dir)
    events_path = target_log_dir / "events.jsonl"
    result_json_path = target_log_dir / "result.json"
    result_md_path = target_log_dir / "result.md"

    config = SwarmConfig(
        model_planner=model_planner,
        model_worker=model_worker,
        model_judge=model_judge,
        swarm_size=swarm_size,
        ahead_by=ahead_by,
        max_steps=max_steps,
        max_retries=max_retries,
        max_total_tokens=max_total_tokens,
        max_wall_seconds=max_wall_seconds,
        timeout_seconds=timeout_seconds,
        temperature_planner=temperature_planner,
        temperature_worker=temperature_worker,
        temperature_judge=temperature_judge,
        seed_base=seed_base,
        show_rationale=show_rationale,
        dry_run=dry_run,
        stream=stream,
        log_dir=target_log_dir,
        project_name=project_name,
        structured_mode=structured_mode,
    )

    openrouter_key = settings.get("OPENROUTER_API_KEY")
    if not config.dry_run and not openrouter_key:
        typer.echo("OPENROUTER_API_KEY is required unless --dry-run is set.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Configuration: {canonical_json(config.model_dump(mode='python'))}")

    display = LiveSwarmDisplay(config.swarm_size, stream_enabled=config.stream)
    display.start()
    events = EventLogger(events_path)
    events.log("task", {"task": task}, message="task received")
    display.log_event(f"Logging to {target_log_dir}")

    metrics = MetricsTracker(config.max_total_tokens)
    llm_client = LLMClient(
        api_key=openrouter_key,
        base_url=settings.get("OPENROUTER_BASE_URL"),
        timeout=config.timeout_seconds,
        display=display,
        metrics=metrics,
        structured_mode=config.structured_mode,
        stream=config.stream,
        dry_run=config.dry_run,
    )
    consensus = ConsensusEngine(config.ahead_by)
    verifier = ActionVerifier()
    langsmith = LangSmithManager(
        enabled=bool(os.environ.get("LANGSMITH_API_KEY")),
        project_name=config.project_name,
        task=task,
    )
    runtime = RuntimeContext(
        llm=llm_client,
        consensus=consensus,
        verifier=verifier,
        display=display,
        events=events,
        metrics=metrics,
        langsmith=langsmith,
        config=config,
        history=[],
    )

    result: Optional[RunResult] = None
    try:
        result = run_swarm(
            task=task,
            config=config,
            runtime=runtime,
        )
    except KeyboardInterrupt:
        aborted_reason = "interrupted by user"
        events.log("aborted", {"reason": aborted_reason}, message=aborted_reason)
        display.log_event("Interrupted by user.")
        langsmith.complete(error=aborted_reason)
        run_stats = RunStats(
            elapsed_s=time.perf_counter() - metrics.start_time,
            llm_calls=metrics.llm_calls,
            tokens_in=metrics.tokens_in,
            tokens_out=metrics.tokens_out,
            retries=metrics.retries,
            consensus_votes=metrics.consensus_votes,
            aborted_reason=aborted_reason,
        )
        artifacts = RunArtifacts(events_path=events_path, result_md_path=result_md_path, langsmith_run_url=langsmith.run_url)
        result = RunResult(
            task=task,
            final_answer=None,
            steps=list(runtime.history),
            stats=run_stats,
            artifacts=artifacts,
        )
    finally:
        display.stop()

    if result is None:
        raise typer.Exit(1)

    result.artifacts.events_path = events_path
    result.artifacts.result_md_path = result_md_path
    result.artifacts.result_json_path = result_json_path
    if not result.artifacts.langsmith_run_url:
        result.artifacts.langsmith_run_url = langsmith.run_url

    save_json_result(result, result_json_path)
    save_markdown_result(result, result_md_path)

    typer.echo(f"Result saved to {result_json_path}")
    if result.artifacts.langsmith_run_url:
        typer.echo(f"LangSmith run: {result.artifacts.langsmith_run_url}")
