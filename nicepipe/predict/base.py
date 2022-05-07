"""Base/template for predictors."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, Tuple, TypeVar, Union
from dataclasses import dataclass, field

from numpy import ndarray
from asyncio import Task, run as async_run, gather, create_task, to_thread
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from ..utils import rlloop, WithFPSCallback
import nicepipe.utils.uvloop  # use uvloop for child process asyncio loop

IT = TypeVar("IT")
OT = TypeVar("OT")
JSONPrimitives = Union[int, str, dict, list, tuple]


class CallableWithExtra(Generic[IT, OT], Protocol):
    """placeholder async function & type for functions that take kwargs"""

    async def __call__(self, input: IT, **extra) -> OT:
        return input, extra


class BasePredictor(ABC):
    """Abstract class to run models as separate processes."""

    @abstractmethod
    def init(self):
        """Initializes predictor."""
        pass

    @abstractmethod
    def predict(self, img: ndarray, **extra) -> Any:
        """Receives img & extra and returns results in a picklable format."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up predictor's resources."""
        pass

    def begin(self, pipe: Connection):
        """Begins predictor IO as target of Process()."""
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
            img, extra = await to_thread(self.pipe.recv)
            results = await to_thread(self.predict, img, **extra)
            # don't await, unlikely to accumulate
            create_task(to_thread(self.pipe.send, results))


@dataclass
class predictionWorkerCfg:
    """
    The serializable portions of PredictionWorker that can be configured.\n
    Necessary to declare this duplicate to ensure serializability and
    because of quirks with dataclass inheritance.
    """

    # fps related
    max_fps: int = 60
    """max io rate of Predictor in Hz"""
    lock_fps: bool = False
    """Whether to lock prediction rate to input rate."""


@dataclass
class PredictionWorker(predictionWorkerCfg, WithFPSCallback):
    """Worker to manage running models in a separate process.

    Execution pipeline:

    1. __call__ -> process_input -> Predictor
    2. Predictor -> predict -> main thread
    3. main thread -> process_output -> return
    """

    predictor: BasePredictor = field(default_factory=BasePredictor)
    """predictor used"""

    # data processing
    process_input: CallableWithExtra[ndarray, Tuple[ndarray, dict]] = field(
        default_factory=CallableWithExtra
    )
    """Used for input preprocessing on the main thread, notably ensuring input is picklable."""
    process_output: CallableWithExtra[Any, Any] = field(
        default_factory=CallableWithExtra
    )
    """Used for output postprocessing in the child process, notably deserializing output."""
    format_output: CallableWithExtra[Any, JSONPrimitives] = field(
        default_factory=CallableWithExtra
    )
    """Used to ensure output can be JSON serialized at least."""

    # variables
    current_input: Tuple[Tuple[ndarray, int], dict] = None
    """current input"""
    current_output: Any = None
    """current output"""
    is_closing: bool = False
    """flag to break loop"""
    loop_task: Task = None
    """main task for both IO loops"""

    # multiprocessing
    process: Process = None
    """child process to run unpack_in, run & prepare_out."""
    pipe: Connection = None
    """Connection used to communicate with child process."""

    async def _in_loop(self):
        """loop for sending inputs to child process"""
        prev_id = -1
        async for _ in rlloop(self.max_fps):
            if self.is_closing:
                break
            try:
                img, extra = self.current_input
                if self.lock_fps and img[1] == prev_id:
                    continue
                prev_id = img[1]
            except TypeError:  # current_input is initially None
                continue
            input = await self.process_input(img[0], **extra)
            await to_thread(self.pipe.send, input)

    async def _out_loop(self):
        """loop for receiving outputs from child process"""
        while not self.is_closing:
            output = await to_thread(self.pipe.recv)
            self.current_output = await self.process_output(output)
            self.fps_callback()

    def predict(self, img: Tuple[ndarray, int], **extra):
        """returns latest prediction & scheldules img & extra for the next"""
        self.current_input = (img, extra)
        return self.current_output

    async def open(self):
        self.pipe, child_pipe = Pipe()
        self.process = Process(
            target=self.predictor.begin, args=(child_pipe,), daemon=True
        )
        self.process.start()
        self.loop_task = gather(self._in_loop(), self._out_loop())

    async def close(self):
        self.is_closing = True
        await self.loop_task
        self.process.terminate()
        await to_thread(self.process.join)

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
