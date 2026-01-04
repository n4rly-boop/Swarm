"""Command-line entrypoint for the MAKER-aligned SwarmMaker runtime."""
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from .config import settings
from .decomposer import Decomposer
from .discriminator import DecompositionDiscriminator, SolutionDiscriminator
from .io import EventLogger, ensure_log_dir, save_json_result, save_markdown_result
from .llm import LLMClient, MetricsTracker
from .orchestrator import MakerOrchestrator, MakerRuntime
from .red_flag import RedFlagGuard
from .schemas import (
    RunArtifacts,
    RunResult,
    RunStats,
    StructuredMode,
    SwarmConfig,
    canonical_json,
)
from .solver import AtomicSolver
from .verify import StateVerifier

app = typer.Typer(add_completion=False, help="SwarmMaker MAKER orchestrator CLI.")

DEFAULT_DECOMPOSER_MODEL = "google/gemini-2.5-flash-preview-09-2025"
DEFAULT_SOLVER_MODEL = "qwen/qwen2.5-7b-instruct"
DEFAULT_DISCRIMINATOR_MODEL = "google/gemini-2.5-flash-preview-09-2025"


@app.command()
def main(
    task: str = typer.Argument(..., help="Task description to solve."),
    model_decomposer: str = typer.Option(DEFAULT_DECOMPOSER_MODEL, "--model-decomposer", show_default=True),
    model_solver: str = typer.Option(DEFAULT_SOLVER_MODEL, "--model-solver", show_default=True),
    model_discriminator: str = typer.Option(DEFAULT_DISCRIMINATOR_MODEL, "--model-discriminator", show_default=True),
    batch_size: int = typer.Option(4, "--batch-size", help="Samples per round for decomposition and solving."),
    ahead_by: int = typer.Option(2, "--ahead-by", help="Votes required beyond runner-up."),
    max_rounds: int = typer.Option(5, "--max-rounds", help="Max sampling rounds per stage."),
    max_depth: int = typer.Option(6, "--max-depth", help="Maximum recursion depth."),
    max_total_tokens: int = typer.Option(50_000, "--max-total-tokens", help="Budget for all LLM calls."),
    timeout_seconds: int = typer.Option(60, "--timeout-seconds"),
    temperature_decomposer: float = typer.Option(0.3, "--temperature-decomposer"),
    temperature_solver: float = typer.Option(0.8, "--temperature-solver"),
    temperature_discriminator: float = typer.Option(0.2, "--temperature-discriminator"),
    structured_mode: StructuredMode = typer.Option(
        StructuredMode.json_schema,
        "--structured-mode",
        case_sensitive=False,
        help="response_format enforcement mode",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Return deterministic mock results."),
    log_dir: Optional[Path] = typer.Option(
        None, "--log-dir", help="Directory for artifacts (defaults to runs/<timestamp>)."
    ),
) -> None:
    """Execute the MAKER step loop for the provided task."""

    _ = settings
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    target_log_dir = ensure_log_dir(log_dir or Path("runs") / timestamp)
    events_path = target_log_dir / "events.jsonl"
    result_json_path = target_log_dir / "result.json"
    result_md_path = target_log_dir / "result.md"

    config = SwarmConfig(
        model_decomposer=model_decomposer,
        model_solver=model_solver,
        model_discriminator=model_discriminator,
        batch_size=batch_size,
        ahead_by=ahead_by,
        max_rounds=max_rounds,
        max_depth=max_depth,
        max_total_tokens=max_total_tokens,
        timeout_seconds=timeout_seconds,
        temperature_decomposer=temperature_decomposer,
        temperature_solver=temperature_solver,
        temperature_discriminator=temperature_discriminator,
        dry_run=dry_run,
        log_dir=target_log_dir,
        structured_mode=structured_mode,
    )

    openrouter_key = settings.get("OPENROUTER_API_KEY")
    if not config.dry_run and not openrouter_key:
        typer.echo("OPENROUTER_API_KEY is required unless --dry-run is set.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Configuration: {canonical_json(config.model_dump(mode='python'))}")

    events = EventLogger(events_path)
    events.log("task", {"task": task}, message="task received")

    metrics = MetricsTracker(config.max_total_tokens)
    llm_client = LLMClient(
        api_key=openrouter_key,
        base_url=settings.get("OPENROUTER_BASE_URL"),
        timeout=config.timeout_seconds,
        metrics=metrics,
        structured_mode=config.structured_mode,
        dry_run=config.dry_run,
    )

    runtime = MakerRuntime(
        config=config,
        decomposer=Decomposer(llm_client, config),
        solver=AtomicSolver(llm_client, config),
        decomposition_discriminator=DecompositionDiscriminator(config.ahead_by),
        solution_discriminator=SolutionDiscriminator(config.ahead_by),
        red_flag=RedFlagGuard(),
        verifier=StateVerifier(),
        logger=events,
        metrics=metrics,
    )

    orchestrator = MakerOrchestrator(runtime)
    start = time.perf_counter()
    try:
        result = orchestrator.run(task)
    except KeyboardInterrupt:
        typer.echo("Interrupted by user.", err=True)
        raise typer.Exit(1)
    finally:
        elapsed = time.perf_counter() - start
        typer.echo(f"Elapsed: {elapsed:.2f}s")

    result.artifacts.events_path = events_path
    result.artifacts.result_md_path = result_md_path
    result.artifacts.result_json_path = result_json_path

    save_json_result(result, result_json_path)
    save_markdown_result(result, result_md_path)

    typer.echo(f"Result saved to {result_json_path}")
    typer.echo(f"Markdown summary: {result_md_path}")
