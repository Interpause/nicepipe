from __future__ import annotations
import asyncio
import logging
from typing import Any, Tuple
from dataclasses import dataclass, field

import numpy as np
from socketio import AsyncNamespace

from ..utils import cancel_and_join, encodeImg, cv2EncCfg, set_interval
from ..predict import PredictionWorker
from .base import Sink, baseSinkCfg

log = logging.getLogger(__name__)


@dataclass
class sioStreamCfg(baseSinkCfg):
    cv2_enc: cv2EncCfg = field(default_factory=cv2EncCfg)
    room_name: str = "stream_channel"
    namespace: str = "/stream"


@dataclass
class SioStreamer(Sink, sioStreamCfg, AsyncNamespace):
    """Stream output over socketio. See https://python-socketio.readthedocs.io/en/latest/server.html#class-based-namespaces."""

    predictors: dict[str, PredictionWorker] = field(default_factory=dict)
    """predictors are needed in order to jsonify predictions from each."""

    def on_connect(self, sid, environ, auth):
        # TODO: authenticate client; check if sufficient rights to VIEW
        pass

    def on_disconnect(self, sid):
        self.on_unsub_stream(sid, None)

    def on_sub_stream(self, sid, data):
        if self.is_open:
            self.enter_room(sid, self.room_name)
            return 200  # OK
        return 503  # Service Unavailable

    def on_unsub_stream(self, sid, data):
        self.leave_room(sid, self.room_name)
        return 200

    def on_bing(self, sid, data):
        log.warning(sid)
        return "bong"

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

            out[name] = await self.predictors[name].clean_output(
                data, img_encoder=self._encode
            )

        await self.emit("frame", {"img": img, "preds": out}, room=self.room_name)
        self.fps_callback()

    def send(self, img: Tuple[np.ndarray, int], preds: dict[str, Any]):
        self._cur_data = (img, preds)

    async def open(self):
        self.is_open = True
        self._cur_data = None
        self._encode = lambda im: encodeImg(im, **self.cv2_enc)
        self._task = set_interval(self._loop, self.max_fps)

    async def close(self):
        self.is_open = False
        await self.emit("close", room=self.room_name)
        await asyncio.gather(
            self.close_room(self.room_name), cancel_and_join(self._task)
        )
