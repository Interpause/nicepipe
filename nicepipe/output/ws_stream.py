from __future__ import annotations
import asyncio
import logging
from typing import Any, Tuple
from dataclasses import dataclass, field

import numpy as np

from ..api.websocket import WebsocketServer, wssCfg
from ..utils import cancel_and_join, encodeImg, cv2EncCfg, set_interval
from ..predict import PredictionWorker
from .base import Sink, baseSinkCfg

log = logging.getLogger(__name__)


@dataclass
class wsStreamCfg(baseSinkCfg):
    cv2_enc: cv2EncCfg = field(default_factory=cv2EncCfg)
    wss: wssCfg = field(default_factory=wssCfg)


@dataclass
class WebsocketStreamer(Sink, wsStreamCfg):
    """Stream output over websocket."""

    formatters: dict[str, PredictionWorker] = field(default_factory=dict)
    """formatters are needed in order to jsonify predictions from each."""

    async def _loop(self):
        try:
            img, preds = self._cur_data
        except (AttributeError, TypeError):
            return
        try:
            if self.lock_fps and img[1] == self._prev_id:
                return
        except AttributeError:
            pass
        self._prev_id = img[1]
        img = await asyncio.to_thread(self._encode, img[0])

        # We running into the Python object caching issue again!
        out = {}
        for name, data in preds.items():
            if data is None:
                continue

            out[name] = await self.formatters[name](data, img_encoder=self._encode)

        await self._wss.broadcast("frame", {"img": img, "preds": out})
        self.fps_callback()

    def send(self, img: Tuple[np.ndarray, int], preds: dict[str, Any]):
        self._cur_data = (img, preds)

    async def open(self, formatters=None, **_):
        self._cur_data = None
        self.formatters = (
            formatters if isinstance(formatters, dict) else self.formatters
        )
        self._encode = lambda im: encodeImg(im, **self.cv2_enc)
        self._wss = WebsocketServer(**self.wss)
        await self._wss.open()
        self._task = set_interval(self._loop, self.max_fps)

    async def close(self):
        await asyncio.gather(cancel_and_join(self._task), self._wss.close())
