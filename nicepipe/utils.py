from __future__ import annotations
import time
import asyncio
from typing import AsyncIterable
from numpy import ndarray
from cv2 import imencode
from base64 import b64encode


async def rlloop(rate, iterator=None, update_func=lambda: 0):
    '''rate-limited loop in Hz. Loop is infinite if iterator is None.'''
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
        await asyncio.sleep(max(1e-3, 1./rate + t - time.perf_counter()))
        t = time.perf_counter()
        update_func()
        yield i


def encodeJPG(im: ndarray) -> str:
    '''uses opencv to encode ndarray img into base64 jpg'''
    # https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
    # webp is slower
    return b64encode(imencode('.jpg', im)[1]).decode('ascii')
