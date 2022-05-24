import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager

import dearpygui.dearpygui as dpg
from nicepipe.gui.cam import GUIStreamer
from nicepipe.utils import add_fps_counter, cancel_and_join, RLLoop

log = logging.getLogger(__name__)


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


async def gui_loop(render):
    gui_loop = add_fps_counter("main: GUI")
    async for _ in RLLoop(30, update_func=gui_loop):
        render()
        if not dpg.is_dearpygui_running():
            continue


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


def show_all_dpg_tools():
    dpg.show_documentation()
    dpg.show_style_editor()
    dpg.show_debug()
    dpg.show_about()
    dpg.show_metrics()
    dpg.show_font_manager()
    dpg.show_item_registry()


@asynccontextmanager
async def setup_gui():
    with create_gui() as render:
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_theme(global_theme)

        task = asyncio.create_task(gui_loop(render))
        try:
            yield task
        finally:
            await cancel_and_join(task)
