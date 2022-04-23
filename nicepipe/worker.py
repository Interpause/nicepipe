from __future__ import annotations
from logging import getLogger
from dataclasses import dataclass, field

from copy import deepcopy
import cv2
import asyncio
import google.protobuf.json_format as pb_json

from nicepipe.mp_pose import DEFAULT_MP_POSE_CFG, create_predictor_worker
from nicepipe.utils import encodeJPG, rlloop
from nicepipe.rich import rate_bar
from nicepipe.websocket import WebsocketServer

log = getLogger(__name__)

# TODO: asyncio reduces boilerplate & bugs when doing concurrency
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
class Worker:
    '''worker receives videos & outputs predictions'''

    cv2_args: list = (0,)
    '''cv2.VideoCapture args'''
    cv2_height: int = 480
    cv2_width: int = 640
    mp_pose_cfg: dict = field(
        default_factory=lambda: deepcopy(DEFAULT_MP_POSE_CFG))
    '''MediaPipe Pose config'''
    max_fps: int = 30
    wss_host: str = 'localhost'
    wss_port: int = 8080

    def __post_init__(self):
        self.cur_data = None
        self.cur_img = None
        self.pbar = [rate_bar.add_task("main loop", total=float(
            'inf')), rate_bar.add_task("predict loop", total=float('inf'))]
        self.wss = WebsocketServer(self.wss_host, self.wss_port)

    def _recv(self):
        # TODO: use pyAV instead of cv2. support webRTC.
        while self.cap.isOpened() and self.is_open:
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

    async def _loop(self):
        try:
            async for img in rlloop(self.max_fps, iterator=self._recv(), update_func=lambda: rate_bar.update(self.pbar[0], advance=1)):
                self.cur_img = img
                # bottleneck is opencv image yield rate lmao
                self.cur_data = self._mp_predict.predict(img)
                # TODO: lag at higher resolution is from here...
                # webrtc time? output worker?
                self._send()
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
        self.cap = cv2.VideoCapture(*self.cv2_args)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cv2_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cv2_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)

        self._mp_predict = create_predictor_worker(
            cfg=self.mp_pose_cfg,
            # TODO: feels like asyncio has some sort of priority system...
            # probably will be fixed once openCV is read in another thread instead
            # after all asyncio doesnt work with non-asyncio blocking io
            max_fps=self.max_fps*4,
            fps_callback=lambda: rate_bar.update(self.pbar[1], advance=1)
        )

        await asyncio.gather(self.wss.open(), self._mp_predict.open())
        self.loop_task = asyncio.create_task(self._loop())

    async def close(self):
        self.is_open = False
        self.cap.release()
        await asyncio.gather(self.loop_task, self.wss.close(), self._mp_predict.close())

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
