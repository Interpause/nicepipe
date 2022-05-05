from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field


from .mp_pose import mpPoseWorkerCfg

__all__ = ["predictCfg"]


@dataclass
class predictCfg:
    """Predictors available."""

    clothes = None
    mp_pose: Optional[mpPoseWorkerCfg] = field(default_factory=mpPoseWorkerCfg)
