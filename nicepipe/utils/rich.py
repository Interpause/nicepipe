import logging
from contextlib import contextmanager

from rich import get_console
from rich.live import Live
from rich.layout import Layout
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, ProgressColumn


console = get_console()
"""fancy Rich console, please call enable_fancy_console() to fully utilize"""
console.quiet = True
console.record = True
console.height = 10


class TaskSpeed(ProgressColumn):
    def render(self, task):
        speed = task.speed or 0
        if 0 <= speed < 10:
            color = "red"
        elif 10 <= speed < 20:
            color = "yellow"
        else:
            color = "green"
        return f"[{color}]{speed:.1f}Hz"


fps_bar = Progress(
    SpinnerColumn(),
    TimeElapsedColumn(),
    TaskSpeed(),
    "{task.description}",
    speed_estimate_period=2,
    console=console,
)
"""used for tracking fps"""


def add_fps_task(name, total=float("inf"), advance=1):
    """Add task to track via the fps bar and returns update function."""
    task_id = fps_bar.add_task(name, total=total)
    return lambda: fps_bar.update(task_id, advance=advance)


layout = Layout()
"""used for layout"""
layout.split_row(
    Layout(Panel(fps_bar, title="Status"), name="Status", ratio=4),
    Layout(name="Info"),
)
layout["Info"].split_column(
    Layout(Panel("Ctrl+C/ESC to stop.")), Layout("No Status", name="Misc")
)


def update_status(text):
    """Update status text of live display."""
    layout["Info"]["Misc"].update(text)


live = Live(layout, console=console, refresh_per_second=10, transient=True)
"""used for live display"""


@contextmanager
def enable_fancy_console(start_live=True, use_hydra=True):
    """Globally takes over print(), logging and the console to be fancier using Rich."""

    console.quiet = False

    # don't use when using hydra to configure logging
    if not use_hydra:
        logging.basicConfig(
            # don't need to change as RichHandler adds its own defaults over it
            # level=logging.DEBUG,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, console=console)],
            force=True,
        )

    try:
        if start_live:
            live.start()
        yield live
    finally:
        live.stop()
