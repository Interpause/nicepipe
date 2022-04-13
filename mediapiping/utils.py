import time
import asyncio


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
