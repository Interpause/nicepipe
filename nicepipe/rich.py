import logging
from contextlib import contextmanager

from rich import get_console
from rich.live import Live
from rich.layout import Layout
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, ProgressColumn


class TaskSpeed(ProgressColumn):
    def render(self, task):
        speed = task.speed or 0
        if 0 <= speed < 10:
            color = 'red'
        elif 10 <= speed < 20:
            color = 'yellow'
        else:
            color = 'green'
        return f"[{color}]{speed:.1f}Hz"


console = get_console()
'''fancy Rich console, please call enable_fancy_console() to fully utilize'''
console.quiet = True
console.record = True
console.height = 10

layout = Layout()
'''used for layout'''
live = Live(layout, console=console, refresh_per_second=10, transient=True)
'''used for live display'''
rate_bar = Progress(SpinnerColumn(), TimeElapsedColumn(), TaskSpeed(),
                    '{task.description}', speed_estimate_period=2, console=console)
'''add stuff to the fps footer'''


@contextmanager
def enable_fancy_console():
    '''globally takes over print(), logging and the console to be fancier using Rich'''

    console.quiet = False

    logging.basicConfig(
        # don't need to change as RichHandler adds its own defaults over it
        # level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
        force=True
    )

    layout.split_row(
        Layout(Panel(rate_bar, title="Status"), name="Status", ratio=4),
        Layout(name="Info")
    )

    layout['Info'].split_column(
        Layout(Panel('Ctrl+C/ESC to stop.')),
        Layout(name="Misc")
    )

    try:
        live.start()
        yield
    finally:
        live.stop()
