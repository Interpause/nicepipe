from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import logging
from datetime import datetime

import socketio
from sanic import Sanic
from sanic_cors import CORS
from sanic.response import text
from uvicorn import Config, Server

from ..utils import cancel_and_join

log = logging.getLogger(__name__)


def setup_app():
    app = Sanic("app", log_config=None)
    app.config.MOTD = False
    app.config["CORS_SUPPORTS_CREDENTIALS"] = True
    CORS(app)

    @app.route("/")
    def home(request):
        return text(f"TODO: web-based control panel? {datetime.now()}")

    return app


def setup_sio():
    sio = socketio.AsyncServer(async_mode="sanic", cors_allowed_origins=[])

    @sio.event
    def connect(sid, environ, auth):
        # root namespace not implemented. So not connectable.
        raise ConnectionRefusedError(501)  # Not Implemented

    @sio.event
    def disconnect(sid):
        pass

    return sio


@asynccontextmanager
async def start_api(log_level=logging.INFO):
    app = setup_app()
    sio = setup_sio()
    sio.attach(app)
    config = Config(app, log_config=None, log_level=log_level)
    server = Server(config)
    task = asyncio.create_task(server.serve())
    try:
        yield app, sio
    finally:
        await cancel_and_join(task)
