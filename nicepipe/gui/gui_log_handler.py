from collections import deque
import uuid
from logging import NOTSET, Handler, Formatter, getLogger
from textwrap import TextWrapper
import dearpygui.dearpygui as dpg

# TODO: Pretty logger
# - investigate Rich to_svg and to_html methods
# - else full ANSI code interpreter
# - log filtering
# - might be possible using tables
# - use set_width/height(-1) to make items fully expand
# - use textwrap for the message even when using tables

DEFAULT_FORMATTER = Formatter(
    fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y%m%d%H%M%S",
)


class GUILogHandler(Handler):
    tabsize = 2
    wrap_char = 120
    lines = 50
    wrap_indent = " " * 8

    def __init__(self, level=NOTSET, maxlen=500):
        super().__init__(level=level)
        self._msg_log = deque([], maxlen)
        """max messages to keep"""
        self._tag = f"logs-{uuid.uuid4()}"
        self._ready = False
        self._textwrapper = TextWrapper(
            width=self.wrap_char,
            tabsize=self.tabsize,
            subsequent_indent=self.wrap_indent,
        )

    def show(self):
        if not self._ready:
            with dpg.value_registry():
                dpg.add_string_value(default_value="loading...", tag=self._tag)
            self._textbox = dpg.add_input_text(
                multiline=True, readonly=True, source=self._tag
            )
            dpg.set_item_width(self._textbox, -1)
            dpg.set_item_height(self._textbox, -1)
        self._ready = True

    def update(self):
        if self._ready and dpg.is_dearpygui_running():
            dpg.set_value(self._tag, "\n".join(self._msg_log))
            size = dpg.get_text_size(
                "\n".join(["M" * (self.wrap_char + 1)] * self.lines), wrap_width=-1
            )

            dpg.set_item_width(self._textbox, size[0])
            dpg.set_item_height(self._textbox, size[1])

    def emit(self, record):
        try:
            msg = self.format(record)
            self._msg_log.append(self._textwrapper.fill(msg))
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)


def create_gui_log_handler(level=NOTSET, maxlen=500, formatter=DEFAULT_FORMATTER):
    handler = GUILogHandler(level=level, maxlen=maxlen)
    handler.setFormatter(formatter)
    getLogger().addHandler(handler)
    return handler
