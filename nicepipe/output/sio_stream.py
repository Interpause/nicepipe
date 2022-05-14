from __future__ import annotations
import asyncio
import logging
from typing import Any, Tuple
from dataclasses import dataclass, field

import numpy as np
from socketio import AsyncNamespace
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    VideoStreamTrack,
)
from av import VideoFrame

from ..utils import (
    cancel_and_join,
    encodeImg,
    cv2EncCfg,
    gather_and_reraise,
    set_interval,
)
from ..analyze import AnalysisWorker
from .base import Sink, baseSinkCfg

log = logging.getLogger(__name__)

# NOTE: socket.io volatile packets are explicitly not going to be implemented
# the guy suggests to just add a callback to check if clients respond and dont
# send to non-responsive clients.
# Also socket.io-msgpack-parser isn't supported either yet.
# Thats both easy performance gains off the table.


class LiveStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.height = 480
        self.width = 640
        self.counter = 0

    def send_frame(self, img):
        self.frame = VideoFrame.from_ndarray(img, format="bgr24")

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.frame
        frame.pts = pts
        frame.time_base = time_base
        self.counter += 1
        return frame


@dataclass
class sioStreamCfg(baseSinkCfg):
    cv2_enc: cv2EncCfg = field(default_factory=cv2EncCfg)
    room_name: str = "stream_channel"
    namespace: str = "/stream"


@dataclass
class SioStreamer(Sink, sioStreamCfg, AsyncNamespace):
    """Stream output over socketio. See https://python-socketio.readthedocs.io/en/latest/server.html#class-based-namespaces."""

    formatters: dict[str, AnalysisWorker] = field(default_factory=dict)
    """formatters are needed in order to jsonify analysis from each."""
    livestream_track: LiveStreamTrack = field(default_factory=LiveStreamTrack)

    def on_connect(self, sid, environ, auth):
        # TODO: authenticate client; check if sufficient rights to VIEW
        pass

    async def on_disconnect(self, sid):
        conn = self._rtc_conns.pop(sid, None)
        self.on_unsub_stream(sid)
        if conn:
            await conn.close()

    def on_sub_stream(self, sid):
        if self.is_open:
            self.enter_room(sid, self.room_name)
            return 200  # OK
        return 503  # Service Unavailable

    def on_unsub_stream(self, sid):
        self.leave_room(sid, self.room_name)
        return 200

    async def on_sub_rtc(self, sid, sdp):
        # log.debug("%s\t%s", sid, sdp)
        client_sdp = RTCSessionDescription(**sdp)

        conn = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        # chn = conn.createDataChannel("data")
        conn.addTrack(self.livestream_track)
        self._rtc_conns[sid] = conn

        # @chn.on("open")
        # def on_open():
        #     pass

        await conn.setRemoteDescription(client_sdp)
        await conn.setLocalDescription(await conn.createAnswer())

        server_sdp = {
            "type": conn.localDescription.type,
            "sdp": conn.localDescription.sdp,
        }

        # SDP is a blackbox namecard sent by both sides describing what sort of video, audio & data codecs they support
        return 200, server_sdp

    async def on_new_ice_candidate(self, sid, candidate):
        # Each ICE candidate is a proposed connection: TCP or UDP, IP address & port, TURN or direct
        # Throughout the lifespan of a connection, candidates are exchanged to find the optimal connection

        # ICE candidates can be included in the SDP
        # notably aiortc doesnt trickle ICE candidates and include all of its at one go
        # but all browser implementations of webRTC trickle by default
        # requiring a workaround to force the browser to send it all at one go
        # Ultimately, ICE trickle increases initial connection speed among other benefits
        # aiortc at least can receive ICE trickle, but its very recent
        # client side workaround was needed due to JSON representation incompatibility

        conn = self._rtc_conns.get(sid, None)
        if conn:  # conn might not be established yet, or errant client
            await conn.addIceCandidate(RTCIceCandidate(**candidate))

    def on_bing(self, sid):
        log.warning(sid)
        return "bong"

    def _prepare_output(self, img, data):
        img = self._encode(img[0])

        # We running into the Python object caching issue again!
        out = {}
        for name, datum in data.items():
            if datum is None:
                continue
            out[name] = self.formatters[name](datum, img_encoder=self._encode)

        return img, out

    async def _loop(self):
        try:
            img, data = self._cur_data
        except (AttributeError, TypeError):
            return
        try:
            if self.lock_fps and img[1] == self._prev_id:
                return
        except AttributeError:
            pass
        self._prev_id = img[1]
        self.livestream_track.width = img[0].shape[1]
        self.livestream_track.height = img[0].shape[0]
        self.livestream_track.send_frame(img[0])
        img, out = await asyncio.to_thread(self._prepare_output, img, data)
        await self.emit("frame", {"img": img, "data": out}, room=self.room_name)
        self.fps_callback()

    def send(self, img: Tuple[np.ndarray, int], data: dict[str, Any]):
        self._cur_data = (img, data)

    async def open(self, formatters=None, **_):
        self.is_open = True
        self._cur_data = None
        self.formatters = (
            formatters if isinstance(formatters, dict) else self.formatters
        )
        self._rtc_conns: dict[str, RTCPeerConnection] = {}
        self._encode = lambda im: encodeImg(im, **self.cv2_enc)
        self._task = set_interval(self._loop, self.max_fps)
        log.debug(f"{type(self).__name__} opened!")

    async def close(self):
        self.is_open = False
        log.debug(f"{type(self).__name__} closing...")
        await self.emit("close", room=self.room_name)
        await gather_and_reraise(
            self.close_room(self.room_name),
            cancel_and_join(self._task),
            *(c.close() for c in self._rtc_conns.values()),
        )
        log.debug(f"{type(self).__name__} closed!")
