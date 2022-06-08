from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import logging
from datetime import datetime

import socketio
from blacksheep import Application
from uvicorn import Config, Server

from ..utils import cancel_and_join

log = logging.getLogger(__name__)


def setup_app():
    app = Application()
    app.use_cors(
        allow_methods="*",
        allow_origins="*",
        allow_headers="* Authorization",
        max_age=300,
    )

    @app.route("/")
    def home():
        return f"TODO: web-based control panel? {datetime.now()}"

    return app


def setup_sio():
    sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")

    @sio.event
    def connect(sid, environ, auth):
        # root namespace not implemented. So not connectable.
        # raise ConnectionRefusedError(501)  # Not Implemented
        pass

    @sio.event
    def disconnect(sid):
        pass

    @sio.event
    def bing(sid, data):
        log.warning(sid)
        return "bong"

    return sio


async def serve_uvicorn(app, log_level):
    """had to remove their signal handlers to use my own"""

    config = Config(app, log_config=None, log_level=log_level, lifespan="off")
    server = Server(config)

    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)
    await server.startup()

    return server, server.main_loop()


@asynccontextmanager
async def start_api(log_level=logging.INFO):
    http_app = setup_app()
    sio = setup_sio()
    app = socketio.ASGIApp(sio, http_app)

    server, server_loop = await serve_uvicorn(app, log_level)
    task = asyncio.create_task(server_loop)
    try:
        yield app, sio
    finally:
        await server.shutdown()
        await cancel_and_join(task)
