from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..utils import WithFPSCallback


@dataclass
class baseSinkCfg:
    max_fps: int = 30
    """Max output rate of Sink."""
    lock_fps: bool = True
    """Whether to lock output rate to input rate."""


@dataclass
class Sink(ABC, baseSinkCfg, WithFPSCallback):
    """Abstract class for output sources."""

    @abstractmethod
    def send(img: np.ndarray[np.uint8], preds: dict[str, Any]):
        pass

    @abstractmethod
    async def open(self):
        pass

    @abstractmethod
    async def close(self):
        pass

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
