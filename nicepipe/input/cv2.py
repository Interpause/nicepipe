from __future__ import annotations
from typing import Any, Literal
from dataclasses import dataclass
from asyncio import create_task, to_thread, sleep
import logging

import cv2
import numpy as np

from .base import Source
from ..utils import cancel_and_join, gather_and_reraise

log = logging.getLogger(__name__)


@dataclass
class cv2CapCfg:
    """https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html"""

    # https://github.com/omry/omegaconf/pull/381
    # source: Union[str, int] = 0
    source: Any = 0
    """cv2.VideoCapture source"""
    # default cv2 capture source on windows has an unsilenceable warning... but dshow (the alternative) lags..
    api: int = cv2.CAP_ANY
    """cv2.VideoCapture API"""
    size_wh: tuple[int, int] = (1280, 720)
    """cv2.VideoCapture resolution in width, height"""
    fps: int = 30
    """cv2.VideoCapture fps"""


@dataclass
class cv2CapSource(cv2CapCfg, Source):
    """cv2-based video source."""

    @property
    def is_open(self) -> bool:
        """whether the input source is open"""
        return self._cap.isOpened() and not self._is_closing

    @property
    def shape(self) -> tuple[int, int, Literal[3]]:
        """shape of output in HWC"""
        return (self.size_wh[1], self.size_wh[0], 3)

    def __post_init__(self):
        self._is_closing = False
        self._nframe = 0
        self._cap = cv2.VideoCapture()

    def _loop(self):
        while self.is_open:
            success = self._cap.read(self.frame)
            if not success:
                log.debug("Ignoring empty camera frame.")
                continue
            self._nframe += 1
            self.fps_callback()

    async def __anext__(self):
        if not self.is_open:
            raise StopAsyncIteration
        try:
            while self.is_open and self._prev_frame == self._nframe:
                # NOTE:
                # making a __next__ version that uses time.sleep actually performs worse
                # this is because of thread-switching being less efficient than tasks
                # when doing sleep(0)
                await sleep(0)  # how bad is this?
        except AttributeError:
            pass
        finally:
            self._prev_frame = self._nframe
            return self.frame, self._nframe

    def _init_cv2_cap(self):
        # for some reason, props can only be set AFTER opening
        self._cap.open(self.source, self.api)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size_wh[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size_wh[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    async def open(self):
        self._is_closing = False
        self._nframe = 0
        self.frame = np.empty(self.shape, dtype=np.uint8)
        await to_thread(self._init_cv2_cap)
        self._task = create_task(to_thread(self._loop))
        log.debug(
            "%s opened! source: %s, api: %s, size_wh: (%d, %d), fps: %.1f",
            type(self).__name__,
            self.source,
            self._cap.getBackendName(),
            self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self._cap.get(cv2.CAP_PROP_FPS),
        )

    async def close(self):
        self._is_closing = True
        log.debug("%s closing...", type(self).__name__)
        await gather_and_reraise(
            cancel_and_join(self._task),
            to_thread(self._cap.release),
        )
        log.debug("%s closed!", type(self).__name__)


def print_cv2_debug():
    log.debug(
        "See https://docs.opencv.org/4.x/dc/d3d/videoio_8hpp.html for more details."
    )
    log.debug(f"Camera Backends: {cv2.videoio_registry.getCameraBackends()}")
    log.debug(f"File Backends: {cv2.videoio_registry.getStreamBackends()}")
    log.debug(f"Writer Backends: {cv2.videoio_registry.getWriterBackends()}")
