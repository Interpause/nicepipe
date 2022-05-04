from __future__ import annotations
from typing import NamedTuple
from dataclasses import dataclass

from types import SimpleNamespace
import cv2
from mediapipe.python.solutions.pose import Pose as MpPose
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2

from .base import BasePredictor, PredictionWorker


# https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options
DEFAULT_MP_POSE_CFG = dict(
    static_image_mode=False,
    # NOTE: all 3 models have to be run at least once to download their files
    model_complexity=1,  # 0, 1 or 2 (0 or 1 is okay)
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


async def process_input(img, extra):
    res = extra.get("downscale_size", (640, 360))
    return cv2.resize(img, res), {}


def serialize_mp_results(results: NamedTuple):
    obj = {}
    if not results.pose_landmarks is None:
        obj["pose_landmarks"] = results.pose_landmarks.SerializeToString()

    if not results.segmentation_mask is None:
        # np.array of (H,W)
        obj["segmentation_mask"] = results.segmentation_mask
    return obj


async def deserialize_mp_results(results: dict):
    obj = {}
    if "pose_landmarks" in results:
        # create protobuf message
        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        # load contents into message...
        pose_landmarks.ParseFromString(results["pose_landmarks"])
        # yeah google why is Protobuf so user-unfriendly
        obj["pose_landmarks"] = pose_landmarks
    if "segmentation_mask" in results:
        obj["segmentation_mask"] = results["segmentation_mask"]

    if obj == {}:
        return None
    else:
        return SimpleNamespace(**obj)


@dataclass
class MPPosePredictor(BasePredictor):
    cfg: dict
    """MediaPipe Pose config, see https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options."""

    def init(self):
        self.mpp = MpPose(**self.cfg)

    def cleanup(self):
        self.mpp.close()

    def predict(self, img, extra):
        img = img[..., ::-1]  # BGR to RGB
        results = self.mpp.process(img)
        serialized = serialize_mp_results(results)
        return serialized


def create_predictor_worker(cfg: dict = DEFAULT_MP_POSE_CFG, **kwargs):
    return PredictionWorker(
        MPPosePredictor(cfg),
        process_input=process_input,
        process_output=deserialize_mp_results,
        **kwargs,
    )
