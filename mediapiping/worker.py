from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy

import cv2
# import asyncio
from mediapipe.python.solutions.pose import Pose as MpPose

from logging import Logger
log = Logger(__name__)

# https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options
DEFAULT_MP_POSE_CFG = dict(
    static_image_mode=False,
    model_complexity=0,  # 0, 1 or 2 (0 or 1 is okay)
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

    # async def _predict(self, img):
    #     # TODO: use multiple models & combine results
    #     # await asyncio.sleep(2)
    #     results = self.mp_pose.process(img)
    #     self.prev_results = results
    #     return results

    # def next(self):
    #     for img in self._recv():
    #         # async effects dont show for CPU models
    #         # also what happens as tasks accumulate...
    #         # maybe its a bad idea to async after all?
    #         # bottleneck is opencv image yield rate lmao
    #         asyncio.create_task(self._predict(img))
    #         yield self.prev_results, img

    def _predict(self, img):
        # TODO: use multiple models & combine results
        results = self.mp_pose.process(img)
        self.prev_results = results
        return results

    def next(self):
        for img in self._recv():
            results = self._predict(img)
            yield results, img

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        self.mp_pose = MpPose(**self.mp_pose_cfg)

    def close(self):
        self.cap.release()
        self.mp_pose.close()
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_t, exc_v, exc_tb):
        self.close()
