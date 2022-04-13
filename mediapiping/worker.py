from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy

import cv2
import base64
import asyncio
import google.protobuf.json_format as pb_json

import mediapiping.mp_pose_process as mp_pose_process
from mediapiping.utils import rlloop
from mediapiping.websocket import WebsocketServer
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
    '''worker to receive video & run inference & broadcast it'''

    source: str | int = 0
    '''cv2.VideoCapture source'''
    mp_pose_cfg: dict = field(
        default_factory=lambda: deepcopy(DEFAULT_MP_POSE_CFG))
    '''kwargs to configure mediapipe Pose'''
    max_fps: int = 30

    def __post_init__(self):
        self.cur_data = None
        self.cur_img = None
        self.pbar = tqdm(position=2)
        self.wss = WebsocketServer()
        self.pbar.set_description('worker loop')

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
            yield img

    def _send(self):
        # TODO: fyi send webp over wss <<< send video chunks over anything
        # and wss << webrtc
        # encoding the image like this has a performance hit
        # https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
        # webp is slower tho it has better quality for size
        # success, buf = cv2.imencode('.webp', self.cur_img, [
        #                            cv2.IMWRITE_WEBP_QUALITY, 100])

        success, buf = cv2.imencode('.jpg', self.cur_img)
        if not success:
            raise 'image encode failed'

        img = base64.b64encode(buf).decode('ascii')
        pose = pb_json.MessageToDict(self.cur_data.pose_landmarks)[
            'landmark'] if not self.cur_data is None else None

        # below checks if too many frames are being skipped
        # if hasattr(self, 'ptest'):
        #     if(self.ptest == pose):
        #         print('prediction reused')
        # self.ptest = pose

        asyncio.create_task(self.wss.broadcast('frame', {
            'img': img,
            'pose': pose
        }))

    async def _predict(self, img):
        # TODO: execute multiple models concurrently & combine results
        results = await self._mp_predict(img)
        return results

    async def _loop(self):
        async def set_prediction(img):
            results = await self._predict(img)
            # prediction process will return None when still debouncing
            if not results is None:
                self.cur_data = results

        async for img in rlloop(self.max_fps, iter=self._recv(), update_func=self.pbar.update):
            self.cur_img = img
            # bottleneck is opencv image yield rate lmao
            asyncio.create_task(set_prediction(img))
            self._send()

    def next(self):
        while True:
            yield self.cur_data, self.cur_img

    async def open(self):
        self.cap = cv2.VideoCapture(self.source)
        self.mp_proc, self._mp_predict = mp_pose_process.start(
            self.mp_pose_cfg)
        await asyncio.gather(self.wss.open())
        self.loop_task = asyncio.create_task(self._loop())

    async def close(self):
        self.cap.release()
        self.mp_proc.terminate()
        await asyncio.gather(self.loop_task, self.mp_proc.coro_join(), self.wss.close())

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
