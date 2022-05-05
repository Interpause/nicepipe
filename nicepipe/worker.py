from __future__ import annotations
from typing import Any, Tuple, Union
from dataclasses import dataclass, field

from collections import deque
import cv2
import asyncio

from .predict import predictCfg
from .predict.mp_pose import create_prediction_worker

from .utils import encodeImg, rlloop, add_fps_task
from .api.websocket import WebsocketServer

import logging

log = logging.getLogger(__name__)

# TODO:
# 3 loop types:
# - Receive loop (source FPS + user-set FPS)
# - Predict loop (user-set FPS per model) (DONE)
# - Output loop (user-set FPS)
# typically only 1 receive loop. Currently using openCV, ideally use pyAV
# 1 predict loop per model, cur_results is a dict of each model's output
# typically only 1 output loop (websocket server). As worker conceptually
# takes in video & outputs predictions, visualizing or saving not in scope
# predict loops should check if image is different before predicting


@dataclass
class cv2CapCfg:
    """https://docs.opencv.org/4.5.5/d8/dfe/classcv_1_1VideoCapture.html"""

    # https://github.com/omry/omegaconf/pull/381
    # source: Union[str, int] = 0
    source: Any = 0
    """cv2.VideoCapture source"""
    # default cv2 capture source on windows has an unsilenceable warning... but dshow (the alternative) lags..
    api: int = cv2.CAP_ANY
    """cv2.VideoCapture API"""
    size_wh: Tuple[int, int] = (1280, 720)
    """cv2.VideoCapture resolution in width, height"""
    fps: int = 30
    """cv2.VideoCapture fps"""


@dataclass
class cv2EncCfg:
    """https://docs.opencv.org/4.5.5/d4/da8/group__imgcodecs.html"""

    format: str = "jpeg"
    """cv2 encode format"""
    flags: list[int] = field(default_factory=list)
    """cv2 imencode flags"""


@dataclass
class wssCfg:
    """nicepipe.api.websocket"""

    host: str = "localhost"
    port: int = 8080


@dataclass
class workerCfg:
    cv2_cap: cv2CapCfg = field(default_factory=cv2CapCfg)
    cv2_enc: cv2EncCfg = field(default_factory=cv2EncCfg)
    wss: wssCfg = field(default_factory=wssCfg)
    predict: predictCfg = field(default_factory=predictCfg)


@dataclass
class Worker(workerCfg):
    """worker receives videos & outputs predictions"""

    def __post_init__(self):
        self._cur_data = {}
        self._cur_img = None
        self._pbar = {
            "main": add_fps_task("worker loop"),
            **{k: add_fps_task(f"predict:{k}") for k in self.predict},
        }
        self._wss = WebsocketServer(**self.wss)
        self._send_tasks = deque()  # dont limit deque, custom clearing behaviour
        self._queue_len = 60  # max amount of send tasks in queue
        self._encode = lambda im: encodeImg(im, **self.cv2_enc)

    async def _recv(self):
        # TODO: use pyAV instead of cv2. support webRTC.
        while self._cap.isOpened() and self._is_open:
            success, img = await asyncio.to_thread(self._cap.read)
            if not success:
                log.warn("Ignoring empty camera frame.")
                continue  # break if using video

            # pass image by reference by disabling write
            img.flags.writeable = False
            yield img

    async def _send(self):
        # TODO: fyi send webp over wss <<< send video chunks over anything
        # TODO: lag from encodeJPG is significant at higher res, hence use of to_thread()

        img = await asyncio.to_thread(self._encode, self._cur_img)

        preds = {}
        for name, result in self._cur_data.items():
            if result is None:
                continue
            preds[name] = await self._predictor[name].clean_output(
                result, img_encoder=self._encode
            )

        await self._wss.broadcast("frame", {"img": img, **preds})

    async def _clear_queue(self):
        """Clear queued up websocket send tasks to prevent memory leak."""
        tasks = []
        while len(self._send_tasks) > self._queue_len:
            task = self._send_tasks.popleft()
            task.cancel()
            tasks.append(task)

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except:
            pass

    async def _loop(self):
        # NOTE: dont modify extra downstream else it will affect this one
        # but interestingly enough...
        # even if i create a new dict each time (e.g. predict(img, {...}))
        # popping on that dict downstream, ocassionally it carries over to
        # the next loop?
        # I know Python does object caching... am I hitting the limits?
        try:
            async for img in rlloop(
                self.cv2_cap.fps * 2,  # doesnt matter cause we await read()
                iterator=self._recv(),
                update_func=self._pbar["main"],
            ):
                self._cur_img = img

                for name, predictor in self._predictor.items():
                    self._cur_data[name] = predictor.predict(img)

                self._send_tasks.append(asyncio.create_task(self._send()))
                await self._clear_queue()
                if not self._is_open:
                    break
        except KeyboardInterrupt:
            self._is_open = False

    def next(self):
        while self._is_open:
            yield self._cur_data, self._cur_img

    async def open(self):
        self._is_open = True

        # cv2 VideoCapture
        self._cap = cv2.VideoCapture(self.cv2_cap.source, self.cv2_cap.api)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cv2_cap.size_wh[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cv2_cap.size_wh[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.cv2_cap.fps)

        # TODO: how to not hardcode this?? Instantiate?
        self._predictor = {}
        if not self.predict.mp_pose is None:
            mp_pose_cfg = self.predict.mp_pose
            self._predictor["mp_pose"] = create_prediction_worker(
                fps_callback=self._pbar["mp_pose"], **mp_pose_cfg
            )

        await asyncio.gather(
            self._wss.open(), *[p.open() for p in self._predictor.values()]
        )
        self._loop_task = asyncio.create_task(self._loop())

    async def join(self):
        await self._loop_task

    async def close(self):
        self._is_open = False
        log.info("Waiting for input & output streams to close...")
        await asyncio.gather(
            self._loop_task,
            *self._send_tasks,
            self._wss.close(),
            *[p.close() for p in self._predictor.values()],
        )
        # should be called last to avoid being stuck in cap.read() & also so cv2.CAP_MSMF warning message doesnt interrupt the debug logs
        self._cap.release()

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
