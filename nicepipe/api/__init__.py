from __future__ import annotations
from dataclasses import dataclass
import logging
from datetime import datetime

import socketio
from sanic import Sanic
from sanic.response import text
from uvicorn import Config, Server

log = logging.getLogger(__name__)


def setup_app():
    app = Sanic("app", log_config=None)
    app.config.MOTD = False

    @app.route("/")
    def home(request):
        return text(f"TODO: web-based control panel? {datetime.now()}")

    return app


def setup_sio():
    sio = socketio.AsyncServer()
    return sio


async def start_api(log_level=logging.INFO):
    app = setup_app()
    sio = setup_sio()
    sio.attach(app)
    config = Config(app, log_config=None, log_level=log_level)
    server = Server(config)
    await server.serve()
