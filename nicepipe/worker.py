from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass

from collections import deque
import cv2
import asyncio

from omegaconf import DictConfig

from .predict.mp_pose import (
    create_predictor_worker,
    prep_send_mp_results,
)
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

    source: str | int
    """cv2.VideoCapture source"""
    api: int
    """cv2.VideoCapture API"""
    size_wh: Tuple[int, int]
    """cv2.VideoCapture resolution in width, height"""
    fps: int
    """cv2.VideoCapture fps"""


@dataclass
class cv2EncCfg:
    """https://docs.opencv.org/4.5.5/d4/da8/group__imgcodecs.html"""

    format: str
    """cv2 encode format"""
    flags: list[int]
    """cv2 imencode flags"""


@dataclass
class wssCfg:
    """nicepipe.api.websocket"""

    host: str
    port: int


@dataclass
class Worker:
    """worker receives videos & outputs predictions"""

    predictors: DictConfig
    cv2_cap_cfg: cv2CapCfg
    cv2_enc_cfg: cv2EncCfg
    wss_cfg: wssCfg

    queue_len: int = 60
    """Max amount of send tasks in queue"""

    def __post_init__(self):
        self.cur_data = {}
        self.cur_img = None
        self.pbar = {k: add_fps_task(f"predict:{k}") for k in self.predictors}
        self.pbar["main"] = add_fps_task("main loop")
        self.wss = WebsocketServer(**self.wss_cfg)
        self.send_tasks = deque()  # dont limit deque, custom clearing behaviour
        self._encode = lambda im: encodeImg(im, **self.cv2_enc_cfg)

    async def _recv(self):
        # TODO: use pyAV instead of cv2. support webRTC.
        while self.cap.isOpened() and self.is_open:
            success, img = await asyncio.to_thread(self.cap.read)
            if not success:
                log.warn("Ignoring empty camera frame.")
                continue  # break if using video

            # pass image by reference by disabling write
            img.flags.writeable = False
            yield img

    async def _send(self):
        # TODO: fyi send webp over wss <<< send video chunks over anything
        # TODO: lag from encodeJPG is significant at higher res, hence use of to_thread()

        img = await asyncio.to_thread(self._encode, self.cur_img)

        # TODO: HOW TO NOT HARDCODE THIS???
        mp_pose = None
        if "mp_pose" in self.cur_data:
            mp_pose = await prep_send_mp_results(
                self.cur_data["mp_pose"], img_encoder=self._encode
            )

        await self.wss.broadcast("frame", {"img": img, "mp_pose": mp_pose})

    async def _clear_queue(self):
        """Clear queued up websocket send tasks to prevent memory leak."""
        while len(self.send_tasks) > self.queue_len:
            task = self.send_tasks.popleft()
            task.cancel()
            try:
                await task
            except:
                pass

    async def _loop(self):
        # NOTE: dont modify extras downstream else it will affect this one
        # but interestingly enough...
        # even if i create a new dict each time (e.g. predict(img, {...}))
        # popping on that dict downstream, ocassionally it carries over to
        # the next loop?
        # I know Python does object caching... am I hitting the limits?
        try:
            async for img in rlloop(
                self.cv2_cap_cfg.fps * 2,  # doesnt matter cause we await read()
                iterator=self._recv(),
                update_func=self.pbar["main"],
            ):
                self.cur_img = img

                for name, cfg in self.predictors.items():
                    self.cur_data[name] = self._predict[name].predict(
                        img, {"downscale_size": cfg.downscale_wh}
                    )

                self.send_tasks.append(asyncio.create_task(self._send()))
                await self._clear_queue()
                if not self.is_open:
                    break
        except KeyboardInterrupt:
            self.is_open = False

    def next(self):
        while self.is_open:
            yield self.cur_data, self.cur_img

    async def open(self):
        self.is_open = True

        # cv2 VideoCapture
        self.cap = cv2.VideoCapture(self.cv2_cap_cfg.source, self.cv2_cap_cfg.api)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cv2_cap_cfg.size_wh[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cv2_cap_cfg.size_wh[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.cv2_cap_cfg.fps)

        # TODO: how to not hardcode this?? Instantiate?
        self._predict = {}
        if "mp_pose" in self.predictors:
            mp_pose_cfg = self.predictors.mp_pose
            self._predict["mp_pose"] = create_predictor_worker(
                cfg=mp_pose_cfg.cfg,
                max_fps=mp_pose_cfg.max_fps,
                lock_fps_to_input=mp_pose_cfg.lock_fps,
                fps_callback=self.pbar["mp_pose"],
            )

        await asyncio.gather(
            self.wss.open(), *[p.open() for p in self._predict.values()]
        )
        self.loop_task = asyncio.create_task(self._loop())

    async def join(self):
        await self.loop_task

    async def close(self):
        self.is_open = False
        log.info("Waiting for input & output streams to close...")
        await asyncio.gather(
            self.loop_task,
            *self.send_tasks,
            self.wss.close(),
            *[p.close() for p in self._predict.values()],
        )
        # should be called last to avoid being stuck in cap.read() & also so cv2.CAP_MSMF warning message doesnt interrupt the debug logs
        self.cap.release()

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
