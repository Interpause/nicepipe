from __future__ import annotations
import asyncio
from collections import deque
import logging
import time
from typing import Any, Tuple
from dataclasses import dataclass, field
import msgpack

import numpy as np
from socketio import AsyncNamespace
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    VideoStreamTrack,
    RTCDataChannel,
)
import aiortc.codecs
from av import VideoFrame

from ..utils import RLLoop, cancel_and_join, encodeImg, cv2EncCfg, gather_and_reraise
from ..analyze import AnalysisWorker
from .base import Sink, baseSinkCfg

log = logging.getLogger(__name__)

# NOTE: socket.io volatile packets are explicitly not going to be implemented
# the guy suggests to just add a callback to check if clients respond and dont
# send to non-responsive clients.
# Also socket.io-msgpack-parser isn't supported either yet.
# Thats both easy performance gains off the table.

# NOTE: LAG ISSUE: Socket.io is less efficient than pure websockets. Furthermore,
# given the lack of volatile packets, or a way to monkey-patch it in, we have a bad
# situation. The FPS of the main thread (GUI and this) tanks when socket.io is
# saturated. Heck it tanks when the client is unresponsive (during a refresh). This
# might mean we either have to switch back to websocket for streaming or go all the way
# to webRTC.

# NOTE: Obvious in retrospect, theres supposed to be one VideoStreamTrack per connection,
# since the track is supposed to adjust dynamically to connection...
# well lag for anything more than 1 client I guess.

# NOTE: use chrome://webrtc-internals/. its very useful

# NOTE:
# setting the default will slightly force it to try a higher bitrate
# but if its too high, video jitters a lot & quality suffers anyways
# best is to set MAX_BITRATE, in which both browser & aiortc try & hit the highest
# bitrate they can (Ive tested it will do this).
# that said, the bottleneck is likely pyAV's encoder.
# pyAV has to be built from source with ffmpeg 4.0 to enable nvenc encoders,
# then aiortc has to be modified or monkey-patched to use said encoders.
# Good luck with: https://pyav.org/docs/develop/overview/installation.html#build-on-windows

# aiortc.codecs.h264.DEFAULT_BITRATE = aiortc.codecs.vpx.DEFAULT_BITRATE = int(6e6)
aiortc.codecs.h264.MAX_BITRATE = aiortc.codecs.vpx.MAX_BITRATE = int(20e6)


class LiveStreamTrack(VideoStreamTrack):
    def __init__(self, height=480, width=640):
        super().__init__()
        self.height = height
        self.width = width
        self.counter = 0
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[..., :3] = (0, 255, 0)
        self.frame = VideoFrame.from_ndarray(img, format="bgr24")

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
    height: int = 480
    width: int = 640

    def on_connect(self, sid, environ, auth):
        # TODO: authenticate client; check if sufficient rights to VIEW
        pass

    async def on_disconnect(self, sid):
        conn = self._rtc_conns.pop(sid, None)
        chn = self._data_chns.pop(sid, None)
        track = self._live_tracks.pop(sid, None)
        # self.on_unsub_stream(sid)
        if track:
            track.stop()
        if chn:
            chn.close()
        if conn and conn != "waiting":
            await conn.close()

    def on_sub_stream(self, sid):
        # we no longer support this due to socket.io saturation issue
        return 503
        if self.is_open:
            self.enter_room(sid, self.room_name)
            return 200  # OK
        return 503  # Service Unavailable

    def on_unsub_stream(self, sid):
        self.leave_room(sid, self.room_name)
        return 200

    async def on_negotiate_channel(self, sid, channel_id):
        while self._rtc_conns.get(sid, None) == "waiting":
            await asyncio.sleep(0)  # is this bad?

        conn = self._rtc_conns.get(sid, None)
        if not conn:
            return 500

        try:
            chn = conn.createDataChannel(
                "analysis",
                ordered=False,
                negotiated=True,
                id=channel_id,
                maxRetransmits=0,
            )
        except Exception as e:
            return 500

        @chn.on("open")
        def on_open():
            self._data_chns[sid] = chn

        @chn.on("close")
        @chn.on("closing")
        def on_close():
            self._data_chns.pop(sid, None)

        return 200

    async def on_sub_rtc(self, sid, sdp):
        # log.debug("%s\t%s", sid, sdp)
        self._rtc_conns[sid] = "waiting"
        conn = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self._live_tracks[sid] = tracks = LiveStreamTrack(
            height=self.height, width=self.width
        )
        conn.addTrack(tracks)

        self._rtc_conns[sid] = conn  # cannot receive ICE until a certain state?

        client_sdp = RTCSessionDescription(**sdp)
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

        while self._rtc_conns.get(sid, None) == "waiting":
            await asyncio.sleep(0)  # is this bad?
        conn = self._rtc_conns.get(sid, None)
        if conn:  # conn might not be established yet, or errant client
            await conn.addIceCandidate(RTCIceCandidate(**candidate))

    def on_bing(self, sid):
        log.warning(sid)
        return "bong"

    def _prepare_output(self, data):
        # We running into the Python object caching issue again!
        out = {}
        # if(data['mp_pose'] is None): print('a', time.time())
        out = {"time_sent": time.time()}
        for name, datum in data.items():
            if datum is None:
                continue
            out[name] = self.formatters[name](datum, img_encoder=self._encode)

        # if(out.get('mp_pose', None) is None): print('b', time.time())
        # else: print(out['mp_pose']['pose'][0])
        return msgpack.dumps(out)

    async def _loop(self):
        async for _ in RLLoop(self.max_fps):
            try:
                img, data = self._cur_data
            except (AttributeError, TypeError):
                continue
            try:
                if self.lock_fps and img[1] == self._prev_id:
                    continue
            except AttributeError:
                pass
            self._prev_id = img[1]
            self.height = img[0].shape[0]
            self.width = img[0].shape[1]

            # img, out = await asyncio.gather(
            #     asyncio.to_thread(self._encode, img[0]),
            #     asyncio.to_thread(self._prepare_output, data),
            # )
            # await self.emit("frame", {"img": img, "data": out}, room=self.room_name)

            # enc = await asyncio.to_thread(self._prepare_output, data)

            # output thread should have higher priority
            # dont await = save thread comm time + dont give up to other tasks
            # assert not data['mp_pose'] is None
            enc = self._prepare_output(data)

            # There are a few errors that might happen here due to things closing during a disconnect

            # TODO: Figure out how to run this on a separate process
            # aiortc needs the event loop so we cant run the entire loop in another thread
            send_coros = deque([])
            for chn in self._data_chns.values():
                try:
                    # print('sending to ', id)
                    chn.send(enc)
                    send_coros.append(
                        chn._RTCDataChannel__transport._data_channel_flush()
                    )
                    send_coros.append(chn._RTCDataChannel__transport._transmit())
                    # print('sent to', id)

                except Exception as e:
                    log.warning(e)
            for track in self._live_tracks.values():
                try:
                    track.send_frame(img[0])
                except:
                    log.warning(e)
            await asyncio.gather(
                *send_coros, return_exceptions=True
            )  # TODO: log these maybe
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
        self._data_chns: dict[str, RTCDataChannel] = {}
        self._live_tracks: dict[str, LiveStreamTrack] = {}
        self._encode = lambda im: encodeImg(im, **self.cv2_enc)
        self._task = asyncio.create_task(self._loop())
        log.debug(f"{type(self).__name__} opened!")

    async def close(self):
        self.is_open = False
        log.debug(f"{type(self).__name__} closing...")
        await self.emit("close", room=self.room_name)

        for c in self._data_chns.values():
            c.close()

        await gather_and_reraise(
            self.close_room(self.room_name),
            cancel_and_join(self._task),
            *(c.close() for c in self._rtc_conns.values()),
        )
        log.debug(f"{type(self).__name__} closed!")
