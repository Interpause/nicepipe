import asyncio
from nicepipe.utils import enable_fancy_console

from .gui import init_gui

# import uvicorn


async def main():
    # uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
    pass


if __name__ == "__main__":
    with enable_fancy_console(start_live=False):
        asyncio.run(main())
