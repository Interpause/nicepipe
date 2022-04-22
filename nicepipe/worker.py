from __future__ import annotations
from logging import getLogger
from dataclasses import dataclass, field
from copy import deepcopy

import cv2
import asyncio
import google.protobuf.json_format as pb_json

import nicepipe.mp_pose_process as mp_pose_process
from nicepipe.utils import encodeJPG, rlloop
from nicepipe.rich import rate_bar
from nicepipe.websocket import WebsocketServer

log = getLogger(__name__)

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

# TODO: asyncio reduces boilerplate & bugs when doing concurrency
# 3 loop types:
# - Receive loop (source FPS + user-set FPS)
# - Predict loop (user-set FPS per model)
# - Output loop (user-set FPS)
# typically only 1 receive loop. Currently using openCV, ideally use pyAV
# 1 predict loop per model, cur_results is a dict of each model's output
# typically only 1 output loop (websocket server). As worker conceptually
# takes in video & outputs predictions, visualizing or saving not in scope
# predict loops should check if image is different before predicting
#
# Using asyncio's thread/process pool executors, possible that above
# can be very very efficient


@dataclass
class Worker:
    '''worker receives videos & outputs predictions'''

    cv2_args: list = field(default_factory=lambda: [0])
    '''cv2.VideoCapture args'''
    cv2_height: int = 480
    cv2_width: int = 640
    mp_pose_cfg: dict = field(
        default_factory=lambda: deepcopy(DEFAULT_MP_POSE_CFG))
    '''kwargs to configure mediapipe Pose'''
    max_fps: int = 30
    wss_host: str = 'localhost'
    wss_port: int = 8080

    def __post_init__(self):
        self.cur_data = None
        self.cur_img = None
        self.pbar = [rate_bar.add_task("worker loop", total=float(
            'inf')), rate_bar.add_task("predict loop", total=float('inf'))]
        self.wss = WebsocketServer(self.wss_host, self.wss_port)

    def _recv(self):
        # TODO: use pyAV instead of cv2. support webRTC.
        while self.cap.isOpened():
            success, img = self.cap.read()
            if not success:
                log.warn("Ignoring empty camera frame.")
                continue  # break if using video

            # pass image by reference by disabling write
            img.flags.writeable = False
            yield img

    def _send(self):
        # TODO: fyi send webp over wss <<< send video chunks over anything

        img = encodeJPG(self.cur_img)

        mask = None
        pose = None
        if not self.cur_data is None:
            pose = pb_json.MessageToDict(
                self.cur_data.pose_landmarks)['landmark']
            if hasattr(self.cur_data, 'segmentation_mask'):
                mask = encodeJPG(self.cur_data.segmentation_mask)

        asyncio.create_task(self.wss.broadcast('frame', {
            'img': img,
            'mask': mask,
            'pose': pose
        }))

    async def _predict(self, img, wait=False, cancel=False):
        while wait and self._mp_predict.is_busy:
            if cancel and not self.cur_img is img:
                return 'busy'
            await asyncio.sleep(0)

        results = await self._mp_predict(img)
        return results

    async def _loop(self):
        async def set_prediction(img):
            results = await self._predict(img, wait=True, cancel=True)
            # prediction process will return 'busy' when still debouncing
            if results != 'busy':
                self.cur_data = results
                rate_bar.update(self.pbar[1], advance=1)

        async for img in rlloop(self.max_fps, iter=self._recv(), update_func=lambda: rate_bar.update(self.pbar[0], advance=1)):
            self.cur_img = img
            # bottleneck is opencv image yield rate lmao
            asyncio.create_task(set_prediction(img))
            self._send()

    def next(self):
        while True:
            yield self.cur_data, self.cur_img

    async def open(self):
        self.cap = cv2.VideoCapture(*self.cv2_args)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cv2_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cv2_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
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
