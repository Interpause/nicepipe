"""
Backend CV Worker. Worker receives video feed from client then returns analysis. 
Being a remote worker, a client can connect to multiple workers, and workers can 
send analysis to multiple clients. A single client can act as the master and 
configure the worker's behaviour.
"""
from multiprocessing import freeze_support

# needed on windows for multiprocessing
freeze_support()

from .worker import Worker, create_worker

__version__ = "0.8.1"
__all__ = ["Worker", "create_worker"]


def run(cfg):
    from .__main__ import main

    main(cfg)
