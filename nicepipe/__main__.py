import asyncio
import dearpygui.dearpygui as dpg
from nicepipe.utils import enable_fancy_console

from .gui import create_gui

# import uvicorn


async def main():
    # uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
    with create_gui():
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
    pass


if __name__ == "__main__":
    with enable_fancy_console(start_live=False):
        asyncio.run(main())
