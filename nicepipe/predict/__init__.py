from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field


from .mp_pose import mpPoseWorkerCfg, create_mp_pose_worker
from .base import PredictionWorker

__all__ = ["predictCfg", "PredictionWorker", "create_mp_pose_worker"]


@dataclass
class predictCfg:
    """Predictors available."""

    clothes = None
    mp_pose: Optional[mpPoseWorkerCfg] = field(default_factory=mpPoseWorkerCfg)
