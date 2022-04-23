from __future__ import annotations
from dataclasses import dataclass
import logging
from pickle import UnpicklingError
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
        results = mpp.process(img)
        serialized = serialize_mp_results(results)
        pipe.send(serialized)


def childproc(pipe: Connection, cfg: dict):
    # KeyboardInterrupt is separate per process
    try:
        mpp = MpPose(**cfg)
        childtask(pipe, mpp)
    except KeyboardInterrupt:
        pass


@dataclass
class Predictor:
    cfg: dict
    is_busy = False

    def open(self):
        parent_pipe, child_pipe = Pipe()
        self.proc = Process(target=childproc, args=(
            child_pipe, self.cfg), daemon=True)
        self.proc.start()
        self.pipe = parent_pipe

    def close(self):
        self.proc.terminate()
        self.proc.join()

    async def __call__(self, img: np.ndarray):
        '''send an BGR image to the child process for prediction. Returns 'busy' when debouncing.'''
        if self.is_busy:
            return 'busy'
        self.is_busy = True
        try:
            await self.pipe.coro_send(img)
            self.is_busy = False
            reply = await self.pipe.coro_recv()
            return deserialize_mp_results(reply)
        # both EOFError & BrokenPipeError occur when asyncio loop gets terminated
        # for speed reasons we dont wait for receiving results before sending next image
        # hence UnpicklingError may ocassionally occur
        # on windows this doesn't seem fatal, but seems to have been crashing the process
        # when using linux? further testing needed
        except (EOFError, BrokenPipeError, UnpicklingError, AssertionError):
            return 'busy'
        finally:
            # turns out in python, finally runs even when continuing/breaking/returning!
            self.is_busy = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_t, exc_v, exc_tb):
        self.close()
