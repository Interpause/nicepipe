import logging
import dearpygui.dearpygui as dpg
from contextlib import contextmanager

log = logging.getLogger(__name__)


@contextmanager
def create_gui():
    dpg.create_context()
    dpg.show_debug()
    dpg.create_viewport(title="nicepipe", width=1280, height=720, vsync=False)
    dpg.setup_dearpygui()

    with dpg.window(label="Example"):
        dpg.add_text("hello world")
        dpg.add_button(label="save")
        dpg.add_input_text(label="string", default_value="placeholder")
        dpg.add_slider_float(label="float", default_value=0.5, max_value=1)

    dpg.show_viewport()
    try:
        yield dpg.render_dearpygui_frame
    finally:
        dpg.stop_dearpygui()
        dpg.destroy_context()
