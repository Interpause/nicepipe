from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

from ..utils import add_fps_counter
from .mp_pose import mpPoseWorkerCfg, create_mp_pose_worker
from .base import PredictionWorker

__all__ = [
    "predictCfg",
    "create_predictors",
    "PredictionWorker",
    "create_mp_pose_worker",
]


@dataclass
class predictCfg:
    """Predictors available."""

    clothes = None
    mp_pose: Optional[mpPoseWorkerCfg] = field(default_factory=mpPoseWorkerCfg)


def create_predictors(
    cfg: predictCfg, add_fps_counters=True
) -> dict[str, PredictionWorker]:
    # TODO: dont hardcode this, use _target_ instantiation
    predictors: dict[str, PredictionWorker] = {}
    if cfg.mp_pose:
        predictors["mp_pose"] = create_mp_pose_worker(**cfg.mp_pose)

    if add_fps_counters:
        for name, predictor in predictors.items():
            predictor.fps_callback = add_fps_counter(f"predict: {name}")

    return predictors
