"""Rich UI components and streaming callbacks."""
import asyncio
import threading
from collections import deque
from typing import Any, Dict, Optional, Tuple

from langchain_core.callbacks import BaseCallbackHandler
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


PanelEvent = Tuple[str, Tuple[Any, ...]]


class LiveSwarmDisplay:
    """Single-threaded renderer that consumes an event queue for stability."""

    def __init__(self, swarm_size: int, *, stream_enabled: bool = True) -> None:
        self.console = Console()
        self.visible_voters = min(6, max(0, swarm_size))
        self.collapsed_voters = max(0, swarm_size - self.visible_voters)
        self.panel_content: Dict[str, str] = {}
        self.metrics: Dict[str, Any] = {}
        self.events = deque(maxlen=40)
        self.stream_enabled = stream_enabled

        self._queue_ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._live: Optional[Live] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional["asyncio.Queue[PanelEvent]"] = None

        self._init_panels()

    # Public API ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._queue_ready.clear()
        self._thread = threading.Thread(target=self._run_render_loop, daemon=True)
        self._thread.start()
        self._queue_ready.wait(timeout=2.0)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._enqueue("__shutdown__", tuple())
        self._thread.join(timeout=2.0)
        self._thread = None
        if self._live:
            self._live.stop()
            self._live = None
        self._loop = None
        self._queue = None

    def update_metrics(self, data: Dict[str, Any]) -> None:
        self._enqueue("metrics", (data,))

    def stream_token(self, stage: str, token: str) -> None:
        if not self.stream_enabled:
            return
        self._enqueue("token", (stage, token))

    def set_panel_text(self, stage: str, text: str) -> None:
        self._enqueue("set", (stage, text))

    def log_event(self, message: str) -> None:
        self._enqueue("event", (message,))

    # Internal helpers ---------------------------------------------------
    def _enqueue(self, event: str, payload: Tuple[Any, ...]) -> None:
        loop = self._loop
        queue = self._queue
        if loop is None or queue is None:
            return

        def _put() -> None:
            queue.put_nowait((event, payload))

        loop.call_soon_threadsafe(_put)

    def _init_panels(self) -> None:
        for name in ["PLANNER", "AGGREGATE", "JUDGE", "VERIFIER", "FINAL"]:
            self.panel_content[name] = ""
        for idx in range(1, self.visible_voters + 1):
            self.panel_content[f"VOTER#{idx}"] = ""
        if self.collapsed_voters:
            self.panel_content["VOTERS(+extra)"] = f"+{self.collapsed_voters} more voters"

    def _run_render_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        queue: "asyncio.Queue[PanelEvent]" = asyncio.Queue()
        self._loop = loop
        self._queue = queue
        self._queue_ready.set()
        try:
            loop.run_until_complete(self._render_loop(queue))
        finally:
            self._loop = None
            self._queue = None
            loop.close()

    async def _render_loop(self, queue: "asyncio.Queue[PanelEvent]") -> None:
        layout = self._render_layout()
        with Live(layout, console=self.console, refresh_per_second=8) as live:
            self._live = live
            while not self._stop_event.is_set():
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if event == "__shutdown__":
                    break
                self._handle_event(event, payload)
                live.update(self._render_layout(), refresh=True)
            live.update(self._render_layout(), refresh=True)

    def _handle_event(self, event: str, payload: Tuple[Any, ...]) -> None:
        if event == "metrics":
            (data,) = payload
            self.metrics.update(data)
        elif event == "token":
            stage, token = payload
            current = self.panel_content.get(stage, "")
            updated = (current + token)[-2000:]
            self.panel_content[stage] = updated
        elif event == "set":
            stage, text = payload
            self.panel_content[stage] = text[-2000:]
        elif event == "event":
            (message,) = payload
            self.events.appendleft(message)

    def _render_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._render_metrics_panel(), name="metrics", size=3),
            Layout(name="body", ratio=2),
        )
        layout["body"].split_row(
            Layout(self._render_agents_panel(), name="agents", ratio=3),
            Layout(self._render_events_panel(), name="events", ratio=1),
        )
        return layout

    def _render_metrics_panel(self) -> Panel:
        metrics = self.metrics or {}
        content = (
            f"elapsed: {metrics.get('elapsed', 0):.1f}s | "
            f"steps: {metrics.get('steps', 0)} | "
            f"calls: {metrics.get('llm_calls', 0)} | "
            f"tokens: {metrics.get('tokens_in', 0)}/{metrics.get('tokens_out', 0)} | "
            f"retries: {metrics.get('retries', 0)} | "
            f"votes: {metrics.get('consensus_votes', 0)} | "
            f"budget: {metrics.get('budget_remaining', 'n/a')}"
        )
        return Panel(content, title="Metrics", border_style="green")

    def _render_agents_panel(self) -> Columns:
        panels = [
            Panel(Text(text[-2000:]), title=stage, border_style="cyan")
            for stage, text in self.panel_content.items()
        ]
        return Columns(panels, expand=True)

    def _render_events_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        if not self.events:
            table.add_row(Text("No events yet."))
        else:
            for message in list(self.events)[:14]:
                table.add_row(Text(message))
        return Panel(table, title="Events", border_style="magenta")


class StreamingCallbackHandler(BaseCallbackHandler):
    """Streams provider tokens through the LiveSwarmDisplay."""

    def __init__(self, display: LiveSwarmDisplay, stage: str) -> None:
        self.display = display
        self.stage = stage

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # type: ignore[override]
        self.display.stream_token(self.stage, token)
