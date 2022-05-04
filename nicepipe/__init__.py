"""
Backend CV Worker. Worker receives video feed from client then returns predictions. 
Being a remote worker, a client can connect to multiple workers, and workers can 
send predictions to multiple clients. A single client can act as the master and 
configure the worker's behaviour.
"""

from .worker import *
from .utils import *

__version__ = "0.2.2"

app = 0


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")


if __name__ == "__main__":
    main()
