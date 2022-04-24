from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Coroutine, Tuple, Callable
from dataclasses import dataclass, field

from collections import deque
from numpy import ndarray
from asyncio import Task, run as async_run, gather, create_task, to_thread
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from nicepipe.utils import rlloop
import nicepipe.uvloop  # use uvloop for child process asyncio loop


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
            input = await to_thread(self.pipe.recv)
            results = await to_thread(self.predict, *input)
            # don't await, unlikely to accumulate
            create_task(to_thread(self.pipe.send, results))


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
    tasks: deque[Task] = field(default_factory=lambda: deque(maxlen=600))
    '''deque tracking misc tasks'''
    loop_task: Task = None
    '''main task for both IO loops'''
    input_num: int = 0
    '''Number of unique inputs so far, used to limit prediction by input rate.'''
    lock_fps_to_input: bool = True
    '''Whether to lock prediction rate to input rate.'''

    # multiprocessing
    process: Process = None
    '''child process to run unpack_in, run & prepare_out.'''
    pipe: Connection = None
    '''Connection used to communicate with child process.'''

    async def _in_loop(self):
        '''loop for sending inputs to child process'''
        prev_num = self.input_num
        async for _ in rlloop(self.max_fps):
            if self.is_closing:
                break
            # limit fps by input rate, also handily skips initial None input
            if self.input_num == prev_num:
                continue
            if self.lock_fps_to_input:
                prev_num = self.input_num
            input = await self.process_input(*self.current_input)
            # await this or it accumulates
            await to_thread(self.pipe.send, input)

    async def _set_output(self, output: Any):
        self.current_output = await self.process_output(output)

    async def _out_loop(self):
        '''loop for receiving outputs from child process'''
        async for _ in rlloop(self.max_fps, update_func=self.fps_callback):
            if self.is_closing:
                break
            output = await to_thread(self.pipe.recv)
            self.tasks.append(create_task(self._set_output(output)))

    def predict(self, img: ndarray, extra: dict = None):
        '''returns latest prediction & scheldules img & extra for the next'''
        new_input = (img, {} if extra is None else extra)
        if self.current_input is None or not img is self.current_input[0] or extra != self.current_input[1]:
            self.current_input = new_input
            self.input_num += 1
        return self.current_output

    async def open(self):
        self.pipe, child_pipe = Pipe()
        self.process = Process(target=self.predictor.begin,
                               args=(child_pipe,), daemon=True)
        self.process.start()
        self.loop_task = gather(self._in_loop(), self._out_loop())

    async def close(self):
        self.is_closing = True
        await gather(self.loop_task, *self.tasks)
        self.process.terminate()
        await to_thread(self.process.join)

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
