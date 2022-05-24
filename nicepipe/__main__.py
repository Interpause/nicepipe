from __future__ import annotations
import sys
import os
import logging
import asyncio


from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import OmegaConfBaseException


import dearpygui.dearpygui as dpg
from rich.prompt import Confirm
from nicepipe.api import start_api
from nicepipe.gui.fps_display import show_fps
from nicepipe.gui.gui_log_handler import create_gui_log_handler
from nicepipe.input.cv2 import print_cv2_debug


import nicepipe.utils.uvloop
from nicepipe.cfg import get_config, nicepipeCfg
from nicepipe.utils import (
    cancel_and_join,
    enable_fancy_console,
    RLLoop,
    change_cwd,
)
from nicepipe.gui import setup_gui, show_all_dpg_tools, show_camera
from nicepipe.worker import create_worker


log = logging.getLogger(__name__)

gui_log_handler = create_gui_log_handler()

rich_live_display = None


def prompt_test_cuda():
    try:
        import nicepipe.utils.cuda

        if nicepipe.utils.cuda.CUDA_ENABLED:
            if Confirm.ask("Run CUDA Test?", default=False):
                import tensorflow as tf  # noqa

                # import torch # torch.cuda.is_available()
                log.debug(f"DLLs: {nicepipe.utils.cuda.DLLs}")
                log.info(
                    f'Torch CUDA: disabled, Tensorflow CUDA: {len(tf.config.list_physical_devices("GPU")) > 0}'
                )
    # means CUDA & Tensorflow disabled
    except Exception as e:
        if not isinstance(e, ModuleNotFoundError):
            log.warning(e)


async def resume_live_display():
    """exists solely to prevent tflite's log messages from interrupting the fancy logs"""
    await asyncio.sleep(4)
    rich_live_display.start()


async def loop(cfg: nicepipeCfg):
    from nicepipe import __version__

    async with setup_gui():
        gui_sink, cam_window = show_camera()

        def _config_toggle(checkbox, value, user_data):
            gui_sink.visuals_enabled[user_data] = value

        dpg.set_primary_window(cam_window, True)

        # is only their initial values; good enough for positioning
        win_height = dpg.get_viewport_client_height()
        win_width = dpg.get_viewport_client_width()

        with dpg.window(label="Settings", tag="config_window", show=False):
            dpg.add_checkbox(
                label="Show Cam",
                user_data="_camgui",
                callback=_config_toggle,
                default_value=gui_sink.visuals_enabled.get("_camgui", False),
            )
            dpg.add_checkbox(
                label="MP Pose Visualization",
                user_data="mp_pose",
                callback=_config_toggle,
                default_value=gui_sink.visuals_enabled.get("mp_pose", False),
            )
            dpg.add_checkbox(
                label="Keypoint Matcher Visualization",
                user_data="kp",
                callback=_config_toggle,
                default_value=gui_sink.visuals_enabled.get("kp", False),
            )
            dpg.add_checkbox(
                label="Duct Tape Visualization",
                user_data="tape",
                callback=_config_toggle,
                default_value=gui_sink.visuals_enabled.get("tape", False),
            )

        with dpg.window(label="Logs", tag="logs_window", autosize=True, show=False):
            gui_log_handler.show()

        with dpg.window(
            label="Loop FPS",
            tag="fps_window",
            pos=(0, win_height * 0.2),
            height=win_height * 0.4,
            width=win_width * 0.5,
            show=False,
        ):
            update_fps_bar = show_fps()

        with dpg.window(
            label="Credits",
            pos=(win_width * 0.3, win_height * 0.3),
            height=win_height * 0.4,
            width=win_width * 0.4,
            modal=True,
            show=False,
            tag="about_window",
        ):
            dpg.add_text(f"Nicepipe v{__version__}")
            dpg.add_text("Powered by JHTech")

        with dpg.viewport_menu_bar():
            dpg.add_menu_item(
                label="Settings", callback=lambda: dpg.show_item("config_window")
            )
            dpg.add_menu_item(
                label="Logs", callback=lambda: dpg.show_item("logs_window")
            )
            dpg.add_menu_item(label="FPS", callback=lambda: dpg.show_item("fps_window"))
            dpg.add_menu_item(label="DPG", callback=show_all_dpg_tools)
            dpg.add_menu_item(
                label="About", callback=lambda: dpg.show_item("about_window")
            )

        async with start_api(log_level=cfg.misc.log_level) as (app, sio):
            worker = create_worker(cfg)
            worker.sinks["gui"] = gui_sink
            sio.register_namespace(worker.sinks["sio"])

            async with worker:
                if cfg.misc.console_live_display:
                    resume_task = asyncio.create_task(resume_live_display())

                async for _ in RLLoop(5):
                    update_fps_bar()
                    gui_log_handler.update()
                    if not dpg.is_dearpygui_running():
                        break
                if cfg.misc.console_live_display:
                    await cancel_and_join(resume_task)
                rich_live_display.stop()


def main(cfg: DictConfig):
    try:
        from nicepipe import __version__  # only way to avoid cyclic dependency

        with enable_fancy_console(
            start_live=False, log_level=cfg.misc.log_level
        ) as live:
            global rich_live_display
            rich_live_display = live

            # nicepipe & __main__ logs are more important to see
            logging.getLogger("nicepipe").setLevel(logging.DEBUG)
            log.setLevel(logging.DEBUG)

            # these 2 are really noisy
            logging.getLogger("aioice.ice").setLevel(
                max(logging.WARNING, cfg.misc.log_level)
            )
            logging.getLogger("aiortc").setLevel(max(logging.INFO, cfg.misc.log_level))

            log.info(
                f":smiley: hewwo world! :eggplant: JHTech's nicepipe [red]v{__version__}[/red]!",
                extra={"markup": True, "highlighter": None},
            )

            log.debug(f"Config: {cfg}")
            log.debug(f"Cwd: {os.getcwd()}")
            print_cv2_debug()

            if sys.platform.startswith("win"):
                log.warning(
                    "Windows detected! Disable Windows Game Mode else worker will lag when not in foreground!"
                )

            if not cfg.misc.skip_tests:
                prompt_test_cuda()

            asyncio.run(loop(cfg))
    finally:
        log.info("Stopped!")


if __name__ == "__main__":
    try:
        cfg = get_config()
        if cfg.misc.save_logs:
            with change_cwd():
                OmegaConf.save(cfg, "config.yml")
                main(cfg)
        else:
            main(cfg)
    except KeyboardInterrupt:
        pass
    # stacktrace for omegaconf errors is useless
    except OmegaConfBaseException as e:
        log.error(e)
    except Exception as e:
        log.error(e, exc_info=e)
    finally:
        input("Press enter to continue...")
        sys.exit(0)
