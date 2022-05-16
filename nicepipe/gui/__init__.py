import asyncio
import logging
import cv2
import dearpygui.dearpygui as dpg

from contextlib import asynccontextmanager, contextmanager

import numpy as np

import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose

from ..output import Sink
from ..utils import add_fps_counter, cancel_and_join, RLLoop

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


def draw_point(img, pt, color, size=2):
    x, y = int(pt[0]), int(pt[1])
    x1 = max(0, x - size)
    x2 = min(img.shape[1], x + size)
    y1 = max(0, y - size)
    y2 = min(img.shape[0], y + size)
    img[y1:y2, x1:x2, :] = color


def show_camera():
    imbuffer = None
    """HWC, RGB, float32"""
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

        cam = dpg.add_image(texture, parent=window)

        def resize_cam(_, window_tag):
            window_height = dpg.get_item_height(window_tag)
            window_width = dpg.get_item_width(window_tag)
            scale_factor = min(window_height / height, window_width / width)
            new_height = height * scale_factor
            new_width = width * scale_factor
            dpg.set_item_height(cam, new_height)
            dpg.set_item_width(cam, new_width)
            dpg.set_item_pos(
                cam,
                ((window_width - new_width) // 2, (window_height - new_height) // 2),
            )

        resize_cam(window, window)

        with dpg.item_handler_registry(tag="cam_resize_handler"):
            dpg.add_item_resize_handler(callback=resize_cam)
        dpg.bind_item_handler_registry("cam_window", "cam_resize_handler")

    # TODO: This must be refactored. especially as more components & their debug visuals are added
    class GUIStreamer(Sink):
        """dirty hack output sink for the GUI"""

        async def open(self, **_):
            self.visual_mp_pose = True
            self.visual_kp = True
            self.visual_tape = True

        async def close(self):
            pass

        def config_visuals(self, cfg):
            self.visual_mp_pose = cfg["visual_mp_pose"]
            self.visual_kp = cfg["visual_kp"]
            self.visual_tape = cfg["visual_tape"]

        def send(self, img, data):
            img = img[0]
            if imbuffer is None:
                initialize(img.shape[1], img.shape[0])

            imbuffer[...] = img[..., ::-1] / 255
            h, w = imbuffer.shape[:2]

            mp_results = data.get("mp_pose", None)
            if not mp_results is None and self.visual_mp_pose:
                mp_drawing.draw_landmarks(
                    imbuffer,
                    mp_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            kp_results = data.get("kp", None)
            if not kp_results is None and self.visual_kp:
                for (name, box) in kp_results["dets"]:
                    cv2.polylines(
                        imbuffer, [box.astype(np.int32)], True, (0, 255, 0), 2
                    )
                if "debug" in kp_results:
                    debug_kp = kp_results["debug"]
                    # for pt in debug_kp["all_kp"]:
                    #     draw_point(imbuffer, pt, (255, 0, 0))
                    for pt in debug_kp["matched_kp"]:
                        draw_point(imbuffer, pt, (0, 255, 0))

            tape_results = data.get("tape", None)
            if not tape_results is None and self.visual_tape:
                for det in tape_results:
                    x1, y1, x2, y2, conf, cls = det
                    cv2.rectangle(
                        imbuffer,
                        (int(x1 * w), int(y1 * h)),
                        (int(x2 * w), int(y2 * h)),
                        (255, 0, 0),
                        2,
                    )

    return GUIStreamer(), window


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
