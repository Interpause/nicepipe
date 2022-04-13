from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy

import cv2
import asyncio
import websockets
# import google.protobuf.json_format as pb_json

import mediapiping.mp_pose_process as mp_pose_process
from tqdm import tqdm


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

    def __post_init__(self):
        self.prev_results = None
        self.pbar = tqdm(position=2)
        self.pbar.set_description('send loop')
        '''debounce async prediction tasks'''

    def _recv(self):
        # TODO: use pyAV instead of cv2. support webRTC.
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
        # TODO: use multiple models & combine results
        results = await self._mp_predict(img)
        # prediction process will return None when still debouncing
        if not results is None:
            self.prev_results = results
        return results


# '''
# open() should start image recv, predict & send loop in background
# replace next() with function that gets current image & prediction
# and also gives an id that can be used to check if it has changed
# since the last time it was called
# '''

    def next(self):
        for img in self._recv():
            # async effects dont show for CPU models
            # bottleneck is opencv image yield rate lmao
            asyncio.create_task(self._predict(img))
            yield self.prev_results, img

    async def open(self):
        self.cap = cv2.VideoCapture(self.source)
        self.mp_proc, self._mp_predict = mp_pose_process.start(
            self.mp_pose_cfg)
        self.wss = await websockets.serve(self._handle_wss, 'localhost', 8080)

    async def close(self):
        self.cap.release()
        self.wss.close()
        self.mp_proc.terminate()
        await asyncio.gather(self.wss.wait_closed(), self.mp_proc.coro_join())

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
