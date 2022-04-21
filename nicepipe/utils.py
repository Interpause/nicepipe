from __future__ import annotations
import time
import asyncio
from numpy import ndarray
from cv2 import imencode
from base64 import b64encode


async def rlloop(rate, iter=None, update_func=lambda: 0):
    '''rate-limited loop in Hz'''
    t = time.perf_counter()

    if iter is None:
        while True:
            await asyncio.sleep(max(1e-3, 1./rate + t - time.perf_counter()))
            t = time.perf_counter()
            update_func()
            yield

    else:
        for i in iter:
            await asyncio.sleep(max(1e-3, 1./rate + t - time.perf_counter()))
            t = time.perf_counter()
            update_func()
            yield i


def encodeJPG(im: ndarray) -> str:
    # https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
    # webp is slower
    return b64encode(imencode('.jpg', im)[1]).decode('ascii')
