import asyncio
import logging
import dearpygui.dearpygui as dpg
from dearpygui_ext.logger import mvLogger
from contextlib import asynccontextmanager, contextmanager

import numpy as np

import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose

from ..utils import add_fps_task, cancel_and_join, rlloop

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
    gui_loop = add_fps_task("gui loop")
    async for _ in rlloop(60, update_func=gui_loop):
        render()
        if not dpg.is_dearpygui_running():
            # for some reason, breaking here destroys some child process sufficiently
            # to crash the main process. It probably is dpg's process, not mine.
            continue


def show_camera():
    imbuffer = None
    """HWC, RGB, float32"""

    window = dpg.add_window(label="Cam")

    def initialize(width, height):
        nonlocal imbuffer
        imbuffer = np.zeros((height, width, 3), dtype=np.float32)

        with dpg.texture_registry():
            texture = dpg.add_raw_texture(
                width,
                height,
                imbuffer,
                format=dpg.mvFormat_Float_rgb,
            )

        dpg.add_image(texture, parent=window)

        dpg.set_value("logs", dpg.get_value("logs") + "\ncam received!")

    def update_imbuffer(results, img):
        if imbuffer is None:
            initialize(img.shape[1], img.shape[0])

        imbuffer[...] = img[..., ::-1] / 255
        mp_results = results.get("mp_pose")
        if not mp_results is None:
            mp_drawing.draw_landmarks(
                imbuffer,
                mp_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

    return update_imbuffer, window


def show_logger():
    with dpg.value_registry():
        dpg.add_string_value(default_value="hello", tag="logs")

    with dpg.window(label="Logs"):
        dpg.add_input_text(multiline=True, readonly=True, source="logs")

    # dearpygui's logger is too simplistic. also doesnt support copy and paste. might as well write my own.
    # probably will have to write it as a Handler that creates a window or smth
    # logger = mvLogger(parent=dpg.window(label="Logs"))


@asynccontextmanager
async def setup_gui():
    with create_gui() as render:
        # dpg.show_documentation()
        # dpg.show_style_editor()
        dpg.show_debug()
        # dpg.show_about()
        # dpg.show_metrics()
        # dpg.show_font_manager()
        # dpg.show_item_registry()

        show_logger()
        # show_camera()

        task = asyncio.create_task(gui_loop(render))
        try:
            yield task
        finally:
            await cancel_and_join([task])
