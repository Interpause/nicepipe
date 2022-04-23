from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Coroutine, Tuple, Callable
from dataclasses import dataclass, field

from numpy import ndarray
from asyncio import run as async_run, gather, create_task
from aioprocessing import AioConnection as Connection, AioPipe as Pipe, AioProcess as Process
from aioprocessing.process import AioProcess

from nicepipe.utils import rlloop


async def passthrough(*args):
    '''placeholder async passthrough function'''
    return args


class BasePredictor(ABC):
    '''Abstract class to run models as separate processes.'''

    @abstractmethod
    def init(self):
        '''Initializes predictor.'''
        pass

    @abstractmethod
    def predict(self, img: ndarray, extra: dict) -> Any:
        '''Receives img & extra and returns results in a picklable format.'''
        pass

    @abstractmethod
    def cleanup(self):
        '''Clean up predictor's resources.'''
        pass

    def begin(self, pipe: Connection):
        '''Begins predictor IO as target of Process().'''
        try:
            self.pipe = pipe
            self.init()
            async_run(self._loop())
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    async def _loop(self):
        while True:
            # send & receive concurrently for performance
            # have measured over 1 min averaging period that async is faster
            input = await self.pipe.coro_recv()
            results = self.predict(*input)
            # don't await, unlikely to accumulate
            self.pipe.coro_send(results)


@dataclass
class PredictionWorker:
    '''Worker to manage running models in a separate process.

    \nExecution pipeline:
    \n1. __call__ -> process_input -> Predictor
    \n2. Predictor -> predict -> main thread
    \n3. main thread -> process_output -> return
    '''

    predictor: BasePredictor
    '''predictor used'''

    # data processing
    process_input: Callable[[ndarray, dict], Coroutine[Tuple[ndarray, dict]]] = field(
        default=passthrough)
    '''Used for input preprocessing on the main thread, notably ensuring input is picklable.'''
    process_output: Callable[[Any], Coroutine[Any]] = field(
        default=passthrough)
    '''Used for output postprocessing in the child process, notably deserializing output.'''

    # fps related
    max_fps: int = 30
    '''max io rate of Predictor in Hz'''
    fps_callback: Callable = field(default=lambda: 0)
    '''function to call every loop, useful for debugging.'''

    # variables
    current_input: Tuple[ndarray, dict] = None
    '''current input'''
    current_output: Any = None
    '''current output'''
    is_closing: bool = False
    '''flag to break loop'''
    loop_tasks: list = field(default_factory=list)
    '''loop tasks'''

    # multiprocessing
    process: AioProcess = None
    '''child process to run unpack_in, run & prepare_out.'''
    pipe: Connection = None
    '''Connection used to communicate with child process.'''

    async def _in_loop(self):
        '''loop for sending inputs to child process'''
        async for _ in rlloop(self.max_fps):
            if self.is_closing:
                break
            if self.current_input is None:
                continue
            input = await self.process_input(*self.current_input)
            # await this or it accumulates
            await self.pipe.coro_send(input)

    async def _set_output(self, output: Any):
        self.current_output = await self.process_output(output)

    async def _out_loop(self):
        '''loop for receiving outputs from child process'''
        async for _ in rlloop(self.max_fps, update_func=self.fps_callback):
            if self.is_closing:
                break
            output = await self.pipe.coro_recv()
            create_task(self._set_output(output))

    def predict(self, img: ndarray, extra: dict = {}):
        '''returns latest prediction & scheldules img & extra for the next'''
        self.current_input = (img, extra)
        return self.current_output

    async def open(self):
        self.pipe, child_pipe = Pipe()
        self.process = Process(target=self.predictor.begin,
                               args=(child_pipe,), daemon=True)
        self.process.start()
        self.loop_tasks = [create_task(
            self._in_loop()), create_task(self._out_loop())]

    async def close(self):
        self.is_closing = True
        await gather(*self.loop_tasks)
        self.process.terminate()
        await self.process.coro_join()

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
