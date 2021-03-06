from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterable, Callable, Deque

import time
import asyncio
from asyncio import CancelledError, Task
from base64 import b64encode
from urllib.parse import quote_from_bytes
from numpy import ndarray
from cv2 import imencode
from .logging import add_fps_counter, update_status, enable_fancy_console, change_cwd

__all__ = [
    "RLLoop",
    "encodeJPG",
    "encodeImg",
    "trim_task_queue",
    "reraise_errors",
    "cancel_and_join",
    "WithFPSCallback",
    "add_fps_counter",
    "update_status",
    "enable_fancy_console",
    "change_cwd",
]


class RLLoop:
    """rate-limited loop in Hz. Loop is infinite if iterator is None."""

    def __init__(self, rate, iterator=None, update_func=lambda: 0):
        self.is_async = isinstance(iterator, AsyncIterable)
        self._next = (
            iterator.__aiter__().__anext__
            if self.is_async
            else iter(int, 1).__next__  # infinite iterator if None
            if iterator is None
            else iterator.__iter__().__next__
        )
        self.period = 1.0 / rate
        self._update = update_func

    def __iter__(self):
        if self.is_async:
            raise RuntimeError(
                "Generally a design flaw to run async iterator as synchronous."
            )
        t = 0
        try:
            # dont ask me why yield has to be after measurement
            while True:
                t = time.perf_counter()
                i = self._next()
                self._update()
                time.sleep(max(0, self.period - time.perf_counter() + t))
                yield i
        except StopIteration:
            pass

    async def __aiter__(self):
        if not self.is_async:

            async def wrap(f=self._next):
                try:
                    return f()  # to_thread might be useful if iterator is heavy
                except StopIteration:
                    raise StopAsyncIteration

            self._next = wrap

        t = 0
        try:
            while True:
                t = time.perf_counter()
                i = await self._next()
                self._update()
                await asyncio.sleep(max(0, self.period - time.perf_counter() + t))
                yield i
        except StopAsyncIteration:
            pass


@dataclass
class WithFPSCallback:
    fps_callback: Callable = field(default=lambda: 0)
    """callback for updating FPS counter"""


async def gather_and_reraise(*tasks: Task, ignored=[]):
    """Wraps asyncio.gather to make sure all tasks complete before reraising errors."""
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = tuple(
        r
        for r in results
        if isinstance(r, Exception) and not any(isinstance(r, e) for e in ignored)
    )
    if len(errors) == 0:
        return results
    elif len(errors) == 1:
        raise errors[0]
    else:
        raise Exception(*errors)


async def cancel_and_join(
    *tasks: Task, reraise=True, ignored=[CancelledError, KeyboardInterrupt]
):
    """Cancel and await all tasks. if reraise is true, raises a list of all exceptions from tasks excluding CancelledError and KeyboardInterrupt."""
    if len(tasks) == 0:
        return
    for task in tasks:
        task.cancel()
    if reraise:
        await gather_and_reraise(*tasks, ignored=ignored)
    else:
        await asyncio.gather(*tasks, return_exceptions=True)


async def trim_task_queue(tasks: Deque[Task], maxlen: int):
    """Clear queued up tasks which may have gotten stuck."""
    popped = []
    while len(tasks) > maxlen:
        popped.append(tasks.popleft())

    return await cancel_and_join(*popped)


def set_interval(afunc, fps, maxlen=10, args=[], kwargs={}):
    """Runs async function at fps as a task. Queues up the tasks and clears the queue when needed."""

    async def loop():
        tasks = deque()
        try:
            async for _ in RLLoop(fps):
                tasks.append(asyncio.create_task(afunc(*args, **kwargs)))
                await trim_task_queue(tasks, maxlen)
        except CancelledError:
            pass
        finally:
            await cancel_and_join(*tasks)

    return asyncio.create_task(loop())


# opencv options available for encoding
# https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
# NOTE:
# - opencv's webp compressor is slower than jpeg no matter the options used
# - base64 is 2x smaller than percent-encoded bytes
# - sending video chunks will always be more efficient cause videos only deal with differences between frames
@dataclass
class cv2EncCfg:
    """https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html"""

    format: str = "jpeg"
    """cv2 encode format"""
    flags: list[int] = field(default_factory=list)
    """cv2 imencode flags"""


def encodeImg(im: ndarray, format: str, b64=True, flags=[]) -> str:
    """Encodes image using opencv into string safe for sending."""
    _, enc = imencode(f".{format}", im, flags)
    data = b64encode(enc).decode("ascii") if b64 else quote_from_bytes(enc.tobytes())
    return f'data:image/{format}{";base64" if b64 else ""},{data}'


def encodeJPG(im: ndarray) -> str:
    """Uses opencv to encode ndarray img into base64 jpg. Opencv default is 95% quality & compression level 1 (fastest)."""
    return encodeImg(im, "jpg")
