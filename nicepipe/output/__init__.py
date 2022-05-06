from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from ..utils import add_fps_counter
from .base import Sink
from .ws_stream import wsStreamCfg, WebsocketStreamer
from .sio_stream import sioStreamCfg, SioStreamer

__all__ = [
    "outputCfg",
    "create_sinks",
    "Sink",
    "wsStreamCfg",
    "WebsocketStreamer",
    "sioStreamCfg",
    "SioStreamer",
]


@dataclass
class outputCfg:
    ws_stream: Optional[wsStreamCfg] = field(default_factory=wsStreamCfg)
    sio_stream: Optional[sioStreamCfg] = field(default_factory=sioStreamCfg)


def create_sinks(cfg: outputCfg, add_fps_counters=True) -> dict[str, Sink]:
    # TODO: dont hardcode this either
    sinks: dict[str, Sink] = {}
    if cfg.ws_stream:
        sinks["ws"] = WebsocketStreamer(**cfg.ws_stream)
    if cfg.sio_stream:
        sinks["sio"] = SioStreamer(**cfg.sio_stream)

    if add_fps_counters:
        for name, sink in sinks.items():
            sink.fps_callback = add_fps_counter(f"output: {name}")

    return sinks
