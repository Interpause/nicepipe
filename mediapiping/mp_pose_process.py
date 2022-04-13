from __future__ import annotations
from typing import NamedTuple
from types import SimpleNamespace

import numpy as np
from aioprocessing import AioConnection as Connection, AioPipe as Pipe, AioProcess as Process
from mediapipe.python.solutions.pose import Pose as MpPose
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
from tqdm import tqdm
# import time

# TODO: Abstractify this into:
# Serializer func, Deserializer func, Initialization func & inference function (including preprocessing)


def serialize_mp_results(results: NamedTuple):
    return {k: v.SerializeToString() for k, v in vars(results).items() if hasattr(v, 'SerializeToString')}


def deserialize_mp_results(results: dict):
    # no way to dynamically determine the type of protobuf msg used :(
    if 'pose_landmarks' in results:
        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks.ParseFromString(results['pose_landmarks'])
        return SimpleNamespace(pose_landmarks=pose_landmarks)
    else:
        return None


def childtask(pipe: Connection, mpp: MpPose):
    pbar = tqdm(position=1)
    pbar.set_description('mp proc loop')
    while True:
        pbar.update()
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


def start(cfg: dict):
    parent_pipe, child_pipe = Pipe()

    proc = Process(target=childproc, args=(child_pipe, cfg), daemon=True)
    proc.start()

    is_busy = False

    async def predict(img: np.ndarray):
        '''send an BGR image to the child process for prediction. Returns None when debouncing.'''
        nonlocal is_busy
        if is_busy:
            return None
        is_busy = True
        # print('parent send: ', time.time_ns())
        await parent_pipe.coro_send(img)
        is_busy = False
        # tested: no broken pipe if clear debounce before recv is done
        reply = await parent_pipe.coro_recv()

        # print('parent recv: ', time.time_ns())
        return deserialize_mp_results(reply)

    return proc, predict
