from __future__ import annotations
from typing import Tuple, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..utils import WithFPSCallback


@dataclass
class Source(ABC, WithFPSCallback):
    """Abstract class for input sources."""

    frame: np.ndarray[np.uint8] = field(default=np.empty([], dtype=np.uint8))
    """current video frame"""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        return False

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, Literal[3]]:
        return (0, 0, 3)

    @abstractmethod
    async def __anext__(self) -> Tuple[np.ndarray[np.uint8], int]:
        return self.frame, 0

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

    def __aiter__(self):
        return self
