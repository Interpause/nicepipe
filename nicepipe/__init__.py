"""
Backend CV Worker. Worker receives video feed from client then returns predictions. 
Being a remote worker, a client can connect to multiple workers, and workers can 
send predictions to multiple clients. A single client can act as the master and 
configure the worker's behaviour.
"""
from multiprocessing import freeze_support

# needed on windows for multiprocessing
freeze_support()

from .worker import *
from .utils import *

__version__ = "0.3.1"


def run():
    from .__main__ import main

    main()
