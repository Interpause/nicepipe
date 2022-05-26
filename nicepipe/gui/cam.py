from __future__ import annotations
import asyncio
from dataclasses import dataclass, field

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from nicepipe.output.base import Sink
from nicepipe.utils import RLLoop, cancel_and_join


@dataclass
class GUIStreamer(Sink):
    """output sink for GUI"""

    visualizers: dict = field(default_factory=dict)
    visuals_enabled: dict = field(default_factory=dict)
    """visualizations enabled"""
    size: int = 360
    """buffer is sized by width"""
    window_tag: int | str = None
    """window to attach to"""

    def _resize_gui(self, _, tag):
        window_height = dpg.get_item_height(tag)
        window_width = dpg.get_item_width(tag)
        scale_factor = min(window_height / self.height, window_width / self.size)
        new_height = self.height * scale_factor
        new_width = self.size * scale_factor
        dpg.set_item_height(self._cam_tag, new_height)
        dpg.set_item_width(self._cam_tag, new_width)
        dpg.set_item_pos(
            self._cam_tag,
            ((window_width - new_width) // 2, (window_height - new_height) // 2),
        )

    def _init_gui(self, width, height):
        self.size = width
        self.height = height

        self._imbuf = np.zeros((height, width, 3), dtype=np.float32)
        """HWC, RGB, float32"""
        with dpg.texture_registry():
            texture = dpg.add_raw_texture(
                width, height, self._imbuf, format=dpg.mvFormat_Float_rgb
            )

        self._cam_tag = dpg.add_image(texture, parent=self.window_tag)

        cam_handlers = dpg.add_item_handler_registry()
        dpg.add_item_resize_handler(callback=self._resize_gui, parent=cam_handlers)
        dpg.bind_item_handler_registry(self.window_tag, cam_handlers)
        self._resize_gui(self.window_tag, self.window_tag)

    def _loop(self):
        for _ in RLLoop(self.max_fps):
            if self._is_closing:
                break
            if not self.visuals_enabled.get("_camgui", False):
                continue

            try:
                img, all_data = self._cur_data
            except (AttributeError, TypeError):
                continue

            try:
                if self.lock_fps and img[1] == self._prev_id:
                    continue
            except AttributeError:
                pass
            self._prev_id = img[1]

            img = cv2.resize(img[0][..., ::-1], (self.size, self.height)) / 255

            for name, visualizer in self.visualizers.items():
                data = all_data.get(name, None)
                if not data is None and self.visuals_enabled.get(name, False):
                    visualizer((img, data))

            self._imbuf[...] = img

    def send(self, img, data):
        if not self.visuals_enabled.get("_camgui", False):
            return

        if self._imbuf is None:
            self._init_gui(self.size, (self.size * img[0].shape[0]) // img[0].shape[1])
            self._task = asyncio.create_task(asyncio.to_thread(self._loop))

        self._cur_data = (img, data)

    async def open(self, visualizers=None, **_):
        self._imbuf = None
        self._is_closing = False
        self.visualizers = (
            visualizers if isinstance(visualizers, dict) else self.visualizers
        )
        self.visualizers["_camgui"] = None
        self._task = None

    async def close(self):
        self._is_closing = True
        if self._task:
            await cancel_and_join(self._task)
