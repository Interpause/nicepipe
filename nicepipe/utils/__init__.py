from __future__ import annotations
from typing import AsyncIterable, Deque

import time
import asyncio
from asyncio import Task
from base64 import b64encode
from urllib.parse import quote_from_bytes
from numpy import ndarray
from cv2 import imencode
from .logging import add_fps_task, update_status, enable_fancy_console, change_cwd

__all__ = [
    "rlloop",
    "encodeJPG",
    "encodeImg",
    "trim_task_queue",
    "add_fps_task",
    "update_status",
    "enable_fancy_console",
    "change_cwd",
]


async def rlloop(rate, iterator=None, update_func=lambda: 0):
    """rate-limited loop in Hz. Loop is infinite if iterator is None."""

    is_async = isinstance(iterator, AsyncIterable)
    if is_async:
        iterator = iterator.__aiter__()
    else:
        # set iterator to be infinite if None
        iterator = iter(int, 1) if iterator is None else iterator.__iter__()

    p = 1.0 / rate
    t = 0
    while True:
        t = time.perf_counter()
        try:
            i = (await iterator.__anext__()) if is_async else iterator.__next__()
        except (StopIteration, StopAsyncIteration):
            break

        update_func()
        await asyncio.sleep(max(1e-3, p - time.perf_counter() + t))
        # yield comes after the time measurement
        # might be how async loops work, but i found that including yield
        # into the time measurement roughly doubles FPS
        yield i


async def trim_task_queue(tasks: Deque[Task], maxlen: int):
    """Clear queued up tasks which may have gotten stuck."""
    popped = []
    while len(tasks) > maxlen:
        task = tasks.popleft()
        task.cancel()
        popped.append(task)

    return await asyncio.gather(*popped, return_exceptions=True)


# opencv options available for encoding
# https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
# NOTE:
# - opencv's webp compressor is slower than jpeg no matter the options used
# - base64 is 2x smaller than percent-encoded bytes
# - sending video chunks will always be more efficient cause videos only deal with differences between frames


def encodeImg(im: ndarray, format: str, b64=True, flags=[]) -> str:
    """Encodes image using opencv into string safe for sending."""
    _, enc = imencode(f".{format}", im, flags)
    data = b64encode(enc).decode("ascii") if b64 else quote_from_bytes(enc.tobytes())
    return f'data:image/{format}{";base64" if b64 else ""},{data}'


def encodeJPG(im: ndarray) -> str:
    """Uses opencv to encode ndarray img into base64 jpg. Opencv default is 95% quality & compression level 1 (fastest)."""
    return encodeImg(im, "jpg")
