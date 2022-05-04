from __future__ import annotations
from typing import AsyncIterable

import time
import asyncio
from base64 import b64encode
from urllib.parse import quote_from_bytes
from numpy import ndarray
from cv2 import imencode
from .rich import add_fps_task, update_status, enable_fancy_console

__all__ = [
    "rlloop",
    "encodeJPG",
    "encodeImg",
    "add_fps_task",
    "update_status",
    "enable_fancy_console",
]


async def rlloop(rate, iterator=None, update_func=lambda: 0):
    """rate-limited loop in Hz. Loop is infinite if iterator is None."""
    t = time.perf_counter()

    is_async = isinstance(iterator, AsyncIterable)
    if is_async:
        iterator = iterator.__aiter__()
    else:
        # set iterator to be infinite if None
        iterator = iter(int, 1) if iterator is None else iterator.__iter__()

    while True:
        try:
            i = (await iterator.__anext__()) if is_async else iterator.__next__()
        except (StopIteration, StopAsyncIteration):
            break
        await asyncio.sleep(max(1e-3, 1.0 / rate + t - time.perf_counter()))
        t = time.perf_counter()
        update_func()
        yield i


# opencv options available for encoding
# https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
# NOTE:
# - opencv's webp compressor is slower than jpeg no matter the options used
# - base64 is 2x smaller than percent-encoded bytes
# - sending video chunks will always be more efficient cause videos only deal with differences between frames


def encodeImg(im: ndarray, format: str, b64=True, opts=[]) -> str:
    """Encodes image using opencv into string safe for sending."""
    _, enc = imencode(f".{format}", im, opts)
    data = b64encode(enc).decode("ascii") if b64 else quote_from_bytes(enc.tobytes())
    return f'data:image/{format}{";base64" if b64 else ""},{data}'


def encodeJPG(im: ndarray) -> str:
    """Uses opencv to encode ndarray img into base64 jpg. Opencv default is 95% quality & compression level 1 (fastest)."""
    return encodeImg(im, "jpg")
