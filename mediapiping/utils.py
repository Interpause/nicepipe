import glob
import time
import os
import nvidia
import ctypes
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


def force_cuda_load():
    '''force cuda to load'''
    dlls = []
    glob_paths = [os.path.join(
        nvidia.__path__[0], '**/*.dll'), os.path.join('cudnn', '**/*.dll')]

    for g in glob_paths:
        for path in glob.glob(g, recursive=True):
            try:
                dlls.append(ctypes.WinDLL(path))
            except:
                pass
    return dlls
