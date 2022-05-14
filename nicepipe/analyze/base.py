"""Base/template for analyzers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Generic, Protocol, Tuple, TypeVar, Union
from dataclasses import dataclass, field
import logging

from numpy import ndarray
from asyncio import Task, run as async_run, create_task, to_thread
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from ..utils import cancel_and_join, RLLoop, WithFPSCallback, trim_task_queue
import nicepipe.utils.uvloop  # use uvloop for child process asyncio loop

log = logging.getLogger(__name__)

IT = TypeVar("IT")
OT = TypeVar("OT")
JSONPrimitives = Union[int, str, dict, list, tuple]


class CallableWithExtra(Generic[IT, OT], Protocol):
    """placeholder async function & type for functions that take kwargs"""

    def __call__(self, input: IT, **extra) -> OT:
        return input, extra


def passthrough_extra(input, **extra):
    return input, extra


def passthrough(input, **_):
    return input


class BaseAnalyzer(ABC):
    """Abstract class to run analysis in separate process."""

    @abstractmethod
    def init(self):
        """Initializes analyzer."""
        pass

    @abstractmethod
    def analyze(self, img: ndarray, **extra) -> Any:
        """Receives img & extra and returns results in a picklable format."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up analyzer's resources."""
        pass

    def begin(self, pipe: Connection):
        """Begins analyzer IO as target of Process()."""
        try:
            self.pipe = pipe
            self.init()
            async_run(self._loop())
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    async def _loop(self):
        tasks = deque()
        # output only when there is input, and be ready for new input asap
        # making it have an input & output loop is hence senseless
        while True:
            img, extra = await to_thread(self.pipe.recv)
            try:
                results = (0, await to_thread(self.analyze, img, **extra))
            except Exception as e:
                results = (1, e)  # pass exception back

            # send & receive concurrently for performance
            # have measured over 1 min averaging period that async is faster
            # potential memory leak if not trimmed
            tasks.append(create_task(to_thread(self.pipe.send, results)))
            await trim_task_queue(tasks, 60)


@dataclass
class AnalysisWorkerCfg:
    """The serializable portions of AnalysisWorker that can be configured."""

    # fps related
    max_fps: int = 60
    """max io rate of Analyzer in Hz"""
    lock_fps: bool = False
    """Whether to lock analysis rate to input rate."""


@dataclass
class AnalysisWorker(AnalysisWorkerCfg, WithFPSCallback):
    """Worker to manage running models in a separate process.

    Execution pipeline:

    1. __call__ -> process_input -> Analyzer
    2. Analyzer -> analyze -> main thread
    3. main thread -> process_output -> return
    """

    analyzer: BaseAnalyzer = field(default_factory=BaseAnalyzer)
    """analyzer used"""

    # data processing
    process_input: CallableWithExtra[ndarray, Tuple[ndarray, dict]] = field(
        default=passthrough_extra
    )
    """Used for input preprocessing on the main thread, notably ensuring input is picklable."""
    process_output: CallableWithExtra[Any, Any] = field(default=passthrough)
    """Used for output postprocessing in the child process, notably deserializing output."""
    format_output: CallableWithExtra[Any, JSONPrimitives] = field(default=passthrough)
    """Used to ensure output can be JSON serialized at least."""

    # variables
    current_input: Tuple[Tuple[ndarray, int], dict] = None
    """current input"""
    current_output: Any = None
    """current output"""
    is_closing: bool = False
    """flag to break loop"""
    loop_tasks: list[Task] = None
    """main task for both IO loops"""

    # multiprocessing
    process: Process = None
    """child process to run unpack_in, run & prepare_out."""
    pipe: Connection = None
    """Connection used to communicate with child process."""

    def _in_loop(self):
        """loop for sending inputs to child process"""
        prev_id = -1
        for _ in RLLoop(self.max_fps):
            if self.is_closing:
                break
            try:
                img, extra = self.current_input
                if self.lock_fps and img[1] == prev_id:
                    continue
                prev_id = img[1]
            except TypeError:  # current_input is initially None
                continue
            input = self.process_input(img[0], **extra)
            self.pipe.send(input)

    def _out_loop(self):
        """loop for receiving outputs from child process"""
        while not self.is_closing:
            err, output = self.pipe.recv()
            if err:
                log.error(f"{type(self.analyzer).__name__} error", exc_info=output)
                continue
            self.current_output = self.process_output(output)
            self.fps_callback()

    def __call__(self, img: Tuple[ndarray, int], **extra):
        """returns latest prediction/analysis & scheldules img & extra for the next"""
        self.current_input = (img, extra)
        return self.current_output

    async def open(self):
        self.pipe, child_pipe = Pipe()
        self.process = Process(
            target=self.analyzer.begin,
            args=(child_pipe,),
            daemon=True,
            name=type(self.analyzer).__name__,
        )
        self.process.start()
        self.loop_tasks = (
            create_task(to_thread(self._in_loop)),
            create_task(to_thread(self._out_loop)),
        )
        log.debug(f"{type(self.analyzer).__name__} worker opened!")

    async def close(self):
        self.is_closing = True
        log.debug(f"{type(self.analyzer).__name__} worker closing...")
        await cancel_and_join(*self.loop_tasks)
        self.process.terminate()
        # await to_thread(self.process.join)
        log.debug(f"{type(self.analyzer).__name__} worked closed!")

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
