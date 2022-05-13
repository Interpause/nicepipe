from __future__ import annotations
from dataclasses import dataclass, field
import logging
import asyncio

from .cfg import nicepipeCfg
from .input import cv2CapSource
from .analyze import AnalysisWorker, create_analyzers
from .input import Source
from .output import Sink, create_sinks
from .utils import WithFPSCallback, cancel_and_join, add_fps_counter, gather_and_reraise

log = logging.getLogger(__name__)

# TODO: detect when workers die, close and await, print out the errors, and attempt to restart


@dataclass
class Worker(WithFPSCallback):
    source: Source = field(default_factory=Source)
    """video source"""
    analyzers: dict[str, AnalysisWorker] = field(default_factory=dict)
    """analyzers to run"""
    sinks: dict[str, Sink] = field(default_factory=dict)
    """output sinks"""

    async def _loop(self):
        # NOTE: while the analyze extra feature remains unused till multi-analyzers are
        # implemented, there is a notable glitch to be aware of.
        # Even if a seemingly new dict is passed into __call__ each time, if downstream
        # modifies the dict, it fairly often carries onto the next "new" dict.
        # This is likely due to Python's object caching, and the fact we are hitting its
        # limits by doing realtime video processing.

        async for img in self.source:
            # This loop is locked to the input FPS, but its FPS has to be monitored in case
            # improper implementations of analyzers or sinks slow it down.
            self.fps_callback()
            data = {name: a(img) for name, a in self.analyzers.items()}
            for s in self.sinks.values():
                s.send(img, data)
            if self._is_closing:
                break
        log.debug(f"{type(self).__name__} loop ended!")

    async def open(self):
        self._is_closing = False
        self._formatters = {n: p.format_output for n, p in self.analyzers.items()}

        await gather_and_reraise(
            self.source.open(),
            *(p.open() for p in self.analyzers.values()),
            *(s.open(formatters=self._formatters) for s in self.sinks.values()),
        )
        self._task = asyncio.create_task(self._loop())
        log.debug(f"{type(self).__name__} opened!")

    async def close(self):
        self._is_closing = True
        log.debug(f"{type(self).__name__} closing...")
        try:
            await gather_and_reraise(
                cancel_and_join(self._task),
                self.source.close(),
                *(p.close() for p in self.analyzers.values()),
                *(s.close() for s in self.sinks.values()),
            )
        except Exception as e:
            log.error(e, exc_info=True)
        log.debug(f"{type(self).__name__} closed!")

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()


def create_worker(cfg: nicepipeCfg):
    input_counter = add_fps_counter("input: cv2")
    worker_counter = add_fps_counter("main: worker")

    source = cv2CapSource(fps_callback=input_counter, **cfg.input)

    analyzers = create_analyzers(cfg.analyze)
    sinks = create_sinks(cfg.output)

    return Worker(
        fps_callback=worker_counter, source=source, analyzers=analyzers, sinks=sinks
    )
