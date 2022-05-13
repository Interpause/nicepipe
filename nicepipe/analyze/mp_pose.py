from __future__ import annotations
from typing import Optional, Tuple
from types import SimpleNamespace
from dataclasses import dataclass, field
import asyncio

import cv2
from mediapipe.python.solutions.pose import Pose as MpPose
import mediapipe.framework.formats.landmark_pb2 as landmark_pb2
import google.protobuf.json_format as pb_json

from .base import BaseAnalyzer, AnalysisWorker, AnalysisWorkerCfg
from ..utils import encodeJPG


@dataclass
class mpPoseCfg:
    """
    See https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options.

    NOTE: All 3 model complexities have to be run at least once to download their files.
    """

    static_image_mode: bool = False
    model_complexity: int = 1
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    smooth_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.7


@dataclass
class mpPoseWorkerCfg(AnalysisWorkerCfg):
    cfg: mpPoseCfg = field(default_factory=mpPoseCfg)
    scale_wh: Optional[Tuple[int, int]] = (640, 360)


def serialize_mp_results(results: SimpleNamespace):
    obj = {}
    if not results.pose_landmarks is None:
        obj["pose_landmarks"] = results.pose_landmarks.SerializeToString()

    if not results.segmentation_mask is None:
        # np.array of (H,W)
        obj["segmentation_mask"] = results.segmentation_mask
    return obj


async def deserialize_mp_results(results: dict, **_):
    obj = {}
    if "pose_landmarks" in results:
        # create protobuf message
        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        # load contents into message...
        await asyncio.to_thread(
            pose_landmarks.ParseFromString, results["pose_landmarks"]
        )
        # yeah google why is Protobuf so user-unfriendly
        obj["pose_landmarks"] = pose_landmarks
    if "segmentation_mask" in results:
        obj["segmentation_mask"] = results["segmentation_mask"]

    if obj == {}:
        return None
    else:
        return SimpleNamespace(**obj)


async def prep_send_mp_results(results: SimpleNamespace, img_encoder=encodeJPG, **_):
    """prepare mp results for sending over network"""
    mask = None
    pose = None
    if not results is None:
        pose = (await asyncio.to_thread(pb_json.MessageToDict, results.pose_landmarks))[
            "landmark"
        ]
        if hasattr(results, "segmentation_mask"):
            mask = await asyncio.to_thread(img_encoder, results.segmentation_mask)
    return {"mask": mask, "pose": pose}


@dataclass
class MPPosePredictor(BaseAnalyzer):
    cfg: dict
    """MediaPipe Pose config, see https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options."""

    def init(self):
        self.mpp = MpPose(**self.cfg)

    def cleanup(self):
        self.mpp.close()

    def analyze(self, img, **_):
        img = img[..., ::-1]  # BGR to RGB
        results = self.mpp.process(img)
        serialized = serialize_mp_results(results)
        return serialized


def create_mp_pose_worker(cfg=mpPoseCfg(), scale_wh=mpPoseWorkerCfg.scale_wh, **kwargs):
    async def process_input(img, **extra):
        if scale_wh is None:
            return img, extra
        return await asyncio.to_thread(cv2.resize, img, scale_wh), extra

    return AnalysisWorker(
        analyzer=MPPosePredictor(cfg),
        process_input=process_input,
        process_output=deserialize_mp_results,
        format_output=prep_send_mp_results,
        **kwargs,
    )