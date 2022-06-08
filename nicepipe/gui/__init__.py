import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager, ExitStack

import dearpygui.dearpygui as dpg
from nicepipe.gui.cam import GUIStreamer
from nicepipe.gui.gui_log_handler import create_gui_log_handler
from nicepipe.gui.fps_display import show_fps
from nicepipe.utils import add_fps_counter, cancel_and_join, RLLoop

log = logging.getLogger(__name__)

gui_log_handler = create_gui_log_handler()


@contextmanager
def create_gui():
    dpg.create_context()
    dpg.create_viewport(title="nicepipe", width=1280, height=720, vsync=False)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    try:
        yield dpg.render_dearpygui_frame
    finally:
        dpg.stop_dearpygui()
        dpg.destroy_context()


async def gui_loop(render, state):
    gui_loop = add_fps_counter("main: GUI")
    async for _ in RLLoop(30, update_func=gui_loop):
        render()
        if not dpg.is_dearpygui_running():
            state.is_ending = True
            break


async def logging_loop(*funcs):
    async for _ in RLLoop(5):
        for func in funcs:
            func()
        if not dpg.is_dearpygui_running():
            break


def show_camera():
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowPadding, 0, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowBorderSize, 0, category=dpg.mvThemeCat_Core
            )

    window = dpg.add_window(label="Cam", tag="cam_window", no_scrollbar=True)

    dpg.bind_item_theme(window, theme)

    sink = GUIStreamer(max_fps=30, lock_fps=False, window_tag=window)

    return sink, window


def main_gui():
    from nicepipe import __version__

    gui_sink, cam_window = show_camera()

    dpg.set_primary_window(cam_window, True)

    # is only their initial values; good enough for positioning
    win_height = dpg.get_viewport_client_height()
    win_width = dpg.get_viewport_client_width()

    dpg.add_window(label="Settings", tag="config_window", show=False)
    gui_sink.cfg_window_tag = "config_window"

    with dpg.window(label="Logs", tag="logs_window", autosize=True, show=False):
        gui_log_handler.show()

    with dpg.window(
        label="Loop FPS",
        tag="fps_window",
        pos=(0, win_height * 0.2),
        height=win_height * 0.4,
        width=win_width * 0.5,
        show=False,
    ):
        update_fps_bar = show_fps()

    with dpg.window(
        label="Credits",
        pos=(win_width * 0.3, win_height * 0.3),
        height=win_height * 0.4,
        width=win_width * 0.4,
        modal=True,
        show=False,
        tag="about_window",
    ):
        dpg.add_text(f"Nicepipe v{__version__}")
        dpg.add_text("Powered by JHTech")

    with dpg.viewport_menu_bar():
        dpg.add_menu_item(
            label="Settings", callback=lambda: dpg.show_item("config_window")
        )
        dpg.add_menu_item(label="Logs", callback=lambda: dpg.show_item("logs_window"))
        dpg.add_menu_item(label="FPS", callback=lambda: dpg.show_item("fps_window"))
        dpg.add_menu_item(label="DPG", callback=show_all_dpg_tools)
        dpg.add_menu_item(label="About", callback=lambda: dpg.show_item("about_window"))

    return gui_sink, update_fps_bar


def show_all_dpg_tools():
    dpg.show_documentation()
    dpg.show_style_editor()
    dpg.show_debug()
    dpg.show_about()
    dpg.show_metrics()
    dpg.show_font_manager()
    dpg.show_item_registry()


@asynccontextmanager
async def setup_gui(state):
    with ExitStack() as stack:
        render = stack.enter_context(create_gui())
        global_theme = stack.enter_context(dpg.theme())
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowPadding, 0, category=dpg.mvThemeCat_Core
            )

        dpg.bind_theme(global_theme)
        gui_sink, fps_counter_update = main_gui()

        task1 = asyncio.create_task(gui_loop(render, state))
        task2 = asyncio.create_task(
            logging_loop(fps_counter_update, gui_log_handler.update)
        )

        try:
            yield gui_sink
        finally:
            state.is_ending = True
            await cancel_and_join(task1, task2)
