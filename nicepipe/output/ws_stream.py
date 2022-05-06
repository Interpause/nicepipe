from __future__ import annotations
import asyncio
import logging
from typing import Any
from dataclasses import dataclass, field

import numpy as np

from ..api.websocket import WebsocketServer, wssCfg
from ..utils import cancel_and_join, encodeImg, rlloop, cv2EncCfg
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

    predictors: dict[str, PredictionWorker] = field(default_factory=dict)
    """predictors are needed in order to jsonify predictions from each."""

    async def _loop(self):
        async for _ in rlloop(self.max_fps):
            if self._cur_data is None:
                continue

            img, preds = self._cur_data
            img = await asyncio.to_thread(self._encode, img)

            # We running into the Python object caching issue again!
            out = {}
            for name, data in preds.items():
                if data is None:
                    continue

                if isinstance(data, dict):
                    log.error(f"Python failed us! {data}")

                out[name] = await self.predictors[name].clean_output(
                    data, img_encoder=self._encode
                )

            await self._wss.broadcast("frame", {"img": img, "preds": out})
            self.fps_callback()

    def send(self, img: np.ndarray[np.uint8], preds: dict[str, Any]):
        self._cur_data = (img, preds)

    async def open(self):
        self._is_closing = False
        self._cur_data = None
        self._encode = lambda im: encodeImg(im, **self.cv2_enc)
        self._wss = WebsocketServer(**self.wss)
        await self._wss.open()
        self._task = asyncio.create_task(self._loop())

    async def close(self):
        self._is_closing = True
        await asyncio.gather(cancel_and_join(self._task), self._wss.close())
