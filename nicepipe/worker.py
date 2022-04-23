from __future__ import annotations
from typing import Tuple, Union
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
class Worker:
    '''worker receives videos & outputs predictions'''

    cv2_source: Union[int, str] = 0
    '''cv2.VideoCapture source'''
    cv2_cap_api: int = cv2.CAP_ANY
    '''cv2.VideoCapture API'''
    cv2_size_wh: Tuple[int, int] = (640, 360)
    '''cv2.VideoCapture resolution in width, height'''
    mp_pose_cfg: dict = field(
        default_factory=lambda: deepcopy(DEFAULT_MP_POSE_CFG))
    '''MediaPipe Pose config'''
    mp_size_wh: Tuple[int, int] = (640, 360)
    '''size to downscale input to for mediapipe pose'''
    max_fps: int = 30
    '''Max FPS of PredictionWorkers and VideoCapture FPS. PredictionWorkers may exceed input FPS. cv2 rounds down FPS it doesn't support.'''
    wss_host: str = 'localhost'
    wss_port: int = 8080

    def __post_init__(self):
        self.cur_data = None
        self.cur_img = None
        self.pbar = [rate_bar.add_task("main loop", total=float(
            'inf')), rate_bar.add_task("predict loop", total=float('inf'))]
        self.wss = WebsocketServer(self.wss_host, self.wss_port)

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

        img = await asyncio.to_thread(encodeJPG, self.cur_img)

        mask = None
        pose = None
        if not self.cur_data is None:
            pose = pb_json.MessageToDict(
                self.cur_data.pose_landmarks)['landmark']
            if hasattr(self.cur_data, 'segmentation_mask'):
                mask = await asyncio.to_thread(encodeJPG, self.cur_data.segmentation_mask)

        await self.wss.broadcast('frame', {
            'img': img,
            'mask': mask,
            'pose': pose
        })

    async def _loop(self):
        # NOTE: dont modify extra downstream else it will affect this one
        # but interestingly enough...
        # even if i create a new dict each time (e.g. predict(img, {...}))
        # popping on that dict downstream, ocassionally it carries over to
        # the next loop?
        # I know Python does object caching... am I hitting the limits?
        extras = {'downscale_size': self.mp_size_wh}
        try:
            async for img in rlloop(self.max_fps, iterator=self._recv(), update_func=lambda: rate_bar.update(self.pbar[0], advance=1)):
                self.cur_img = img
                self.cur_data = self._mp_predict.predict(
                    img, extras)
                asyncio.create_task(self._send())
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
        self.cap = cv2.VideoCapture(self.cv2_source, self.cv2_cap_api)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cv2_size_wh[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cv2_size_wh[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)

        self._mp_predict = create_predictor_worker(
            cfg=self.mp_pose_cfg,
            max_fps=self.max_fps,
            fps_callback=lambda: rate_bar.update(self.pbar[1], advance=1)
        )

        await asyncio.gather(self.wss.open(), self._mp_predict.open())
        self.loop_task = asyncio.create_task(self._loop())

    async def close(self):
        self.is_open = False
        log.info('Waiting for input & output streams to close...')
        await asyncio.gather(self.loop_task, self.wss.close(), self._mp_predict.close())
        # should be called last to avoid being stuck in cap.read() & also so cv2.CAP_MSMF warning message doesnt interrupt the debug logs
        self.cap.release()

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()