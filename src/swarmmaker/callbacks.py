"""Rich callbacks and Live UI for SwarmMaker."""
from collections import deque
from typing import Any, Dict, Optional

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from langchain_core.callbacks import BaseCallbackHandler


class LiveSwarmDisplay:
    """Live Rich dashboard showing agent streams and metrics."""

    def __init__(self, swarm_size: int) -> None:
        self.console = Console()
        self.visible_voters = min(6, swarm_size)
        self.collapsed_voters = max(0, swarm_size - self.visible_voters)
        self.panel_content: Dict[str, str] = {}
        self.metrics: Dict[str, Any] = {}
        self.events = deque(maxlen=30)
        self.live: Optional[Live] = None
        self._init_panels()

    def _init_panels(self) -> None:
        for name in ["PLANNER", "JUDGE", "FINAL", "VERIFIER"]:
            self.panel_content[name] = ""
        for idx in range(1, self.visible_voters + 1):
            self.panel_content[f"VOTER#{idx}"] = ""
        if self.collapsed_voters:
            self.panel_content["VOTERS(+extra)"] = f"+{self.collapsed_voters} more"

    def start(self) -> None:
        if self.live is None:
            self.live = Live(self._render_layout(), console=self.console, refresh_per_second=8)
            self.live.start()

    def stop(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None

    def update_metrics(self, data: Dict[str, Any]) -> None:
        self.metrics.update(data)
        self._refresh()

    def stream_token(self, stage: str, token: str) -> None:
        current = self.panel_content.get(stage, "")
        self.panel_content[stage] = current + token
        self._refresh()

    def set_panel_text(self, stage: str, text: str) -> None:
        self.panel_content[stage] = text
        self._refresh()

    def log_event(self, message: str) -> None:
        self.events.appendleft(message)
        self._refresh()

    def _refresh(self) -> None:
        if self.live is not None:
            self.live.update(self._render_layout(), refresh=True)

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
        if not self.metrics:
            content = "elapsed: 0s | steps: 0 | calls: 0 | tokens: 0/0"
        else:
            content = (
                f"elapsed: {self.metrics.get('elapsed', 0):.1f}s | "
                f"steps: {self.metrics.get('steps', 0)} | "
                f"calls: {self.metrics.get('llm_calls', 0)} | "
                f"tokens: {self.metrics.get('tokens_in', 0)}/"
                f"{self.metrics.get('tokens_out', 0)} | "
                f"retries: {self.metrics.get('retries', 0)} | "
                f"budget: {self.metrics.get('budget_remaining', 'n/a')}"
            )
        return Panel(content, title="Metrics", border_style="green")

    def _render_agents_panel(self):
        panels = []
        for stage, text in self.panel_content.items():
            panels.append(Panel(Text(text[-1000:]), title=stage, border_style="cyan"))
        return Columns(panels, expand=True)

    def _render_events_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        for event in list(self.events)[:10]:
            table.add_row(Text(event))
        if not self.events:
            table.add_row(Text("No events yet."))
        return Panel(table, title="Events", border_style="magenta")


class StreamingCallbackHandler(BaseCallbackHandler):
    """Streams tokens into the LiveSwarmDisplay."""

    def __init__(self, display: LiveSwarmDisplay, stage: str) -> None:
        self.display = display
        self.stage = stage

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # type: ignore[override]
        self.display.stream_token(self.stage, token)
