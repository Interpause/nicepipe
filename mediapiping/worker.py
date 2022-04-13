from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy

import cv2
import asyncio
import websockets
# import google.protobuf.json_format as pb_json

import mediapiping.mp_pose_process as mp_pose_process
from mediapiping.utils import rlloop
from tqdm import tqdm
import time


from logging import Logger
log = Logger(__name__)

# https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options
DEFAULT_MP_POSE_CFG = dict(
    static_image_mode=False,
    model_complexity=1,  # 0, 1 or 2 (0 or 1 is okay)
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


@dataclass
class Worker:
    '''worker to receive video & run inference'''

    source: str | int = 0
    '''cv2.VideoCapture source'''
    mp_pose_cfg: dict = field(
        default_factory=lambda: deepcopy(DEFAULT_MP_POSE_CFG))
    '''kwargs to configure mediapipe Pose'''
    max_fps: int = 30

    def __post_init__(self):
        self.prev_results = None
        self.prev_img = None
        self.pbar = tqdm(position=2)
        self.pbar.set_description('worker loop')
        '''debounce async prediction tasks'''

    def _recv(self):
        # TODO: use pyAV instead of cv2. support webRTC.
        # self.cap.read() should be async to allow other things to run...
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                log.warn("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # pass image by reference by disabling write
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yield img

    def _send(self):
        # either protobuf format:
        #   landmarks.SerializeToString()
        # or json format:
        #   pb_json.MessageToJson(landmarks)
        # i think ill treat it as a blackbox & use protobuf format
        pass

    async def _handle_wss(self, ws):
        pass

    async def _predict(self, img):
        # TODO: execute multiple models concurrently & combine results
        results = await self._mp_predict(img)
        return results

    async def _loop(self):
        async def set_prediction(img):
            results = await self._predict(img)
            # prediction process will return None when still debouncing
            if not results is None:
                self.prev_results = results

        async for img in rlloop(self.max_fps, iter=self._recv(), update_func=self.pbar.update):
            self.prev_img = img
            # bottleneck is opencv image yield rate lmao
            # print(img)
            asyncio.create_task(set_prediction(img))
            self._send()

    def next(self):
        while True:
            yield self.prev_results, self.prev_img

    async def open(self):
        self.cap = cv2.VideoCapture(self.source)
        self.mp_proc, self._mp_predict = mp_pose_process.start(
            self.mp_pose_cfg)
        self.wss = await websockets.serve(self._handle_wss, 'localhost', 8080)
        self.loop_task = asyncio.create_task(self._loop())

    async def close(self):
        self.loop_task.cancel()
        self.cap.release()
        self.wss.close()
        self.mp_proc.terminate()
        await asyncio.gather(self.wss.wait_closed(), self.mp_proc.coro_join())

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
