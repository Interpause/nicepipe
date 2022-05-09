from __future__ import annotations
from dataclasses import dataclass, field
import logging
import asyncio

from .cfg import nicepipeCfg
from .input import cv2CapSource
from .predict import PredictionWorker, create_predictors
from .input import Source
from .output import Sink, create_sinks
from .utils import WithFPSCallback, cancel_and_join, add_fps_counter

log = logging.getLogger(__name__)

# TODO: detect when workers die, close and await, print out the errors, and attempt to restart


@dataclass
class Worker(WithFPSCallback):
    source: Source = field(default_factory=Source)
    """video source"""
    predictors: dict[str, PredictionWorker] = field(default_factory=dict)
    """predictors to run"""
    sinks: dict[str, Sink] = field(default_factory=dict)
    """output sinks"""

    async def _loop(self):
        # NOTE: while the predict extra feature remains unused till multi-predictors are
        # implemented, there is a notable glitch to be aware of.
        # Even if a seemingly new dict is passed into predict each time, if downstream
        # modifies the dict, it fairly often carries onto the next "new" dict.
        # This is likely due to Python's object caching, and the fact we are hitting its
        # limits by doing realtime video processing.

        async for img in self.source:
            # This loop is locked to the input FPS, but its FPS has to be monitored in case
            # improper implementations of predictors or sinks slow it down.
            self.fps_callback()
            preds = {name: p.predict(img) for name, p in self.predictors.items()}
            for s in self.sinks.values():
                s.send(img, preds)
            if self._is_closing:
                break
        log.debug(f"{type(self).__name__} loop ended!")

    async def open(self):
        self._is_closing = False
        self._formatters = {n: p.format_output for n, p in self.predictors.items()}

        await asyncio.gather(
            self.source.open(),
            *(p.open() for p in self.predictors.values()),
            *(s.open(formatters=self._formatters) for s in self.sinks.values()),
        )
        self._task = asyncio.create_task(self._loop())
        log.debug(f"{type(self).__name__} opened!")

    async def close(self):
        self._is_closing = True
        log.debug(f"{type(self).__name__} closing...")
        await asyncio.gather(
            cancel_and_join(self._task),
            self.source.close(),
            *(p.close() for p in self.predictors.values()),
            *(s.close() for s in self.sinks.values()),
        )
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

    predictors = create_predictors(cfg.predict)
    sinks = create_sinks(cfg.output)

    return Worker(
        fps_callback=worker_counter, source=source, predictors=predictors, sinks=sinks
    )
