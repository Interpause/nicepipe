from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import NamedTuple
from types import SimpleNamespace

import numpy as np
from aioprocessing import AioConnection as Connection, AioPipe as Pipe, AioProcess as Process
from mediapipe.python.solutions.pose import Pose as MpPose
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2

# TODO: Abstractify this into:
# Serializer func, Deserializer func, Initialization func & inference function (including preprocessing)

log = logging.getLogger(__name__)


def serialize_mp_results(results: NamedTuple):
    # {k: v.SerializeToString() for k, v in vars(results).items() if hasattr(v, 'SerializeToString')}
    obj = {}
    if not results.pose_landmarks is None:
        obj['pose_landmarks'] = results.pose_landmarks.SerializeToString()

    if not results.segmentation_mask is None:
        # np.array of (H,W)
        obj['segmentation_mask'] = results.segmentation_mask
    return obj


def deserialize_mp_results(results: dict):
    # no way to dynamically determine the type of protobuf msg used :(
    obj = {}
    if 'pose_landmarks' in results:
        # create protobuf message
        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        # load contents into message...
        pose_landmarks.ParseFromString(results['pose_landmarks'])
        # yeah f**k google why is Protobuf so user-unfriendly
        obj['pose_landmarks'] = pose_landmarks
    if 'segmentation_mask' in results:
        obj['segmentation_mask'] = results['segmentation_mask']

    if obj == {}:
        return None
    else:
        return SimpleNamespace(**obj)


def childtask(pipe: Connection, mpp: MpPose):
    while True:
        img = pipe.recv()
        img = img[..., ::-1]  # BGR to RGB
        # print('child recv: ', time.time_ns())
        results = mpp.process(img)
        serialized = serialize_mp_results(results)
        # print('child send: ', time.time_ns())
        pipe.send(serialized)


def childproc(pipe: Connection, cfg: dict):
    mpp = MpPose(**cfg)
    childtask(pipe, mpp)


@dataclass
class Predictor:
    pipe: Connection
    is_busy = False

    async def __call__(self, img: np.ndarray):
        '''send an BGR image to the child process for prediction. Returns None when debouncing.'''
        if self.is_busy:
            return None
        self.is_busy = True
        # print('parent send: ', time.time_ns())
        await self.pipe.coro_send(img)
        self.is_busy = False
        # tested: no broken pipe if clear debounce before recv is done
        reply = await self.pipe.coro_recv()

        # print('parent recv: ', time.time_ns())
        return deserialize_mp_results(reply)


def start(cfg: dict):
    parent_pipe, child_pipe = Pipe()

    proc = Process(target=childproc, args=(child_pipe, cfg), daemon=True)
    proc.start()

    return proc, Predictor(parent_pipe)
