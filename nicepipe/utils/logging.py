import os
import logging
from logging import Formatter, LogRecord
from logging.handlers import WatchedFileHandler
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

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


# NOTE:
# In order to log EVERYTHING to a log file, root loggger level has to be DEBUG or NOTSET.
# See: https://docs.python.org/3/howto/logging.html#handlers
# The logger filters what is sent to handlers, including the file handler.
# However, for the stdout handler (Rich), we do not want to log everything of course.
# Hence, global logging level has to be set on the handler.
# Yet, if we want to log DEBUG for a specific package, the handler must be aware of it.
# This is the best solution I could think of.


def filter_by_logger(global_log_level=logging.INFO):
    """
    If a record has no ancestors with set log level, it should be filtered by global_log_level.
    However, getEffectiveLevel() includes the root logger. Hence we compare if the record's
    effective level matches the root (this is despite the fact root should be NOTSET to log all
    records to file).

    Therefore there is an edge case where this filter fails:

    - global_log_level = 50
    - root_level = 10
    - ancestor_level = 10

    Although the record has an ancestor with a set log level and therefore should be filtered
    by the ancestor_level, it will end up filtered by the global_log_level instead.

    This edge case is not a problem as long as root_level = 0 (logging.NOTSET).
    """

    def filter(record: LogRecord):
        root_level = logging.getLogger().level
        level = logging.getLogger(record.name).getEffectiveLevel()
        return (
            record.levelno >= max(global_log_level, root_level)
            if level == root_level
            else level
        )

    return filter


# Yes I know for the below context functions, it should reset the logger to its initial state but idc.


@contextmanager
def enable_fancy_console(start_live=True, log_level=logging.DEBUG):
    """Globally takes over print(), logging and the console to be fancier using Rich."""
    rich = RichHandler(
        rich_tracebacks=True,
        log_time_format="[%X]",
        console=console,
    )
    rich.setFormatter(Formatter(fmt="%(message)s", datefmt="[%X]"))
    rich.addFilter(filter_by_logger(log_level))

    root = logging.getLogger()
    root.addHandler(rich)
    root.setLevel(logging.NOTSET)

    console.quiet = False

    try:
        if start_live:
            live.start()
        yield live
    finally:
        live.stop()


ORIGINAL_CWD = Path.cwd()


@contextmanager
def change_cwd(path="logs", pattern="%Y%m%d%H%M%S", log_name="main.log"):
    folder = Path(path) / datetime.now().strftime(pattern)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(folder)
        file_handler = WatchedFileHandler(
            log_name,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(
            Formatter(
                fmt="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
                datefmt="%Y%m%d%H%M%S",
            )
        )
        file_handler.setLevel(logging.NOTSET)
        root = logging.getLogger()
        root.setLevel(logging.NOTSET)
        root.addHandler(file_handler)
        yield
    finally:
        os.chdir(ORIGINAL_CWD)
