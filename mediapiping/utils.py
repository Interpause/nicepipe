import time
import asyncio


async def rlloop(rate, iter=None, update_func=lambda: 0):
    '''rate-limited loop in Hz'''
    t = time.perf_counter()

    if iter is None:
        while True:
            yield
            update_func()
            await asyncio.sleep(max(1e-3, 1./rate + t - time.perf_counter()))
            t = time.perf_counter()

    else:
        for i in iter:
            yield i
            update_func()
            await asyncio.sleep(max(1e-3, 1./rate + t - time.perf_counter()))
            t = time.perf_counter()
