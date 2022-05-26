from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass, field

from nicepipe.analyze.escape_hatch import create_tape_worker, tapeCfg

from nicepipe.analyze.yolo import create_yolo_worker
from nicepipe.analyze.kp import kpDetCfg, create_kp_worker
from nicepipe.analyze.mp_pose import mpPoseWorkerCfg, create_mp_pose_worker
from ..utils import add_fps_counter
from .base import AnalysisWorker

__all__ = [
    "analysisCfg",
    "create_analyzers",
    "AnalysisWorker",
    "create_mp_pose_worker",
    "create_kp_worker",
]


@dataclass
class analysisCfg:
    """Analyzers available."""

    clothes: Optional[Any] = field(default_factory=dict)
    mp_pose: Optional[mpPoseWorkerCfg] = field(default_factory=mpPoseWorkerCfg)
    kp: Optional[kpDetCfg] = field(default_factory=kpDetCfg)
    tape: Optional[tapeCfg] = field(default_factory=tapeCfg)


def create_analyzers(
    cfg: analysisCfg, add_fps_counters=True
) -> dict[str, AnalysisWorker]:
    # TODO: dont hardcode this, use _target_ instantiation
    analyzers: dict[str, AnalysisWorker] = {}
    if not cfg.mp_pose is None:
        analyzers["mp_pose"] = create_mp_pose_worker(**cfg.mp_pose)
    if not cfg.clothes is None:
        analyzers["clothes"] = create_yolo_worker(**cfg.clothes)
    if not cfg.kp is None:
        analyzers["kp"] = create_kp_worker(**cfg.kp)
    if not cfg.tape is None:
        analyzers["tape"] = create_tape_worker(**cfg.tape)

    if add_fps_counters:
        for name, analyzer in analyzers.items():
            analyzer.fps_callback = add_fps_counter(f"analyze: {name}")

    return analyzers
