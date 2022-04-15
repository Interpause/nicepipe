import time
import asyncio
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, ProgressColumn


class TaskSpeed(ProgressColumn):
    def render(self, task):
        return f"{( task.speed or 0 ):.1f}Hz"


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

rate_bar = Progress(SpinnerColumn(), TimeElapsedColumn(),
                    '{task.description}', TaskSpeed(), speed_estimate_period=2)
