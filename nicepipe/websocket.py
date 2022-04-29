from __future__ import annotations
from dataclasses import dataclass

import json
import asyncio
import websockets


@dataclass
class WebsocketServer:
    '''object-based wrapper around websockets'''
    host: str = 'localhost'
    port: int = 8080
    heartbeat_interval: int = 1

    def __post_init__(self):
        self.clients = set()
        self.tasks = []

    async def _heartbeat(self, ws):
        while ws.open:
            await self.send_client(ws, 'pong', None)
            await asyncio.sleep(self.heartbeat_interval)

    async def _register_client(self, ws):
        self.clients.add(ws)
        try:
            self.tasks.append(asyncio.create_task(self._heartbeat(ws)))
            # while not useful in this context, rate-limiting is possible
            async for msg in ws:
                try:
                    obj = json.loads(msg)
                    # TODO: interface for remote worker configure & control
                    # self.handler()
                    pass
                except:
                    pass
        finally:
            self.clients.remove(ws)

    async def send_client(self, ws, event, obj):
        try:
            await ws.send(json.dumps({
                'event': event,
                'data': obj
            }))
        except websockets.ConnectionClosed:
            pass

    async def broadcast(self, event, obj):
        await asyncio.gather(*[self.send_client(ws, event, obj) for ws in self.clients])

    async def open(self):
        self.wss = await websockets.serve(
            ws_handler=self._register_client,
            host=self.host,
            port=self.port
        )
        await self.wss.start_serving()

    async def close(self):
        self.wss.close()
        await asyncio.gather(self.wss.wait_closed(), *self.tasks)

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self.close()
