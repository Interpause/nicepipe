from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass, field

# from nicepipe.predict.yolo import create_yolo_worker
from nicepipe.predict.kp import kpDetCfg, create_kp_worker
from nicepipe.predict.mp_pose import mpPoseWorkerCfg, create_mp_pose_worker
from ..utils import add_fps_counter
from .base import PredictionWorker

__all__ = [
    "predictCfg",
    "create_predictors",
    "PredictionWorker",
    "create_mp_pose_worker",
    "create_kp_worker",
]


@dataclass
class predictCfg:
    """Predictors available."""

    clothes: Optional[Any] = field(default_factory=dict)
    mp_pose: Optional[mpPoseWorkerCfg] = field(default_factory=mpPoseWorkerCfg)
    kp: Optional[kpDetCfg] = field(default_factory=kpDetCfg)


def create_predictors(
    cfg: predictCfg, add_fps_counters=True
) -> dict[str, PredictionWorker]:
    # TODO: dont hardcode this, use _target_ instantiation
    predictors: dict[str, PredictionWorker] = {}
    if cfg.mp_pose:
        predictors["mp_pose"] = create_mp_pose_worker(**cfg.mp_pose)
    # if cfg.clothes:
    #     predictors["clothes"] = create_yolo_worker(**cfg.clothes)
    if cfg.kp:
        predictors["kp"] = create_kp_worker(**cfg.kp)

    if add_fps_counters:
        for name, predictor in predictors.items():
            predictor.fps_callback = add_fps_counter(f"predict: {name}")

    return predictors
