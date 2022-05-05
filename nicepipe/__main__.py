from __future__ import annotations
from collections import deque
import sys
import os
import logging
import asyncio

import numpy as np
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import OmegaConfBaseException

# import uvicorn
import dearpygui.dearpygui as dpg
from rich.prompt import Confirm

import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose

import nicepipe.utils.uvloop
from nicepipe.cfg import get_config
from nicepipe.utils import (
    enable_fancy_console,
    add_fps_task,
    rlloop,
    change_cwd,
    trim_task_queue,
)
from nicepipe.gui import create_gui
from nicepipe.worker import Worker


log = logging.getLogger(__name__)

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


async def gui_loop(imbuffer):
    gui_loop = add_fps_task("gui loop")
    with create_gui() as render:
        with dpg.texture_registry(show=True):
            dpg.add_raw_texture(
                imbuffer.shape[1],
                imbuffer.shape[0],
                imbuffer,
                tag="cam_texture",
                format=dpg.mvFormat_Float_rgb,
            )

        with dpg.window(label="Cam", tag="cam"):
            dpg.add_image("cam_texture")
        dpg.set_primary_window("cam", True)
        dpg.set_viewport_vsync(False)

        async for _ in rlloop(60, update_func=gui_loop):
            render()
            if not dpg.is_dearpygui_running():
                break


async def loop(cfg: DictConfig):
    async with Worker(**cfg.worker) as worker:
        vid_loop = add_fps_task("video loop")
        resume_task = asyncio.create_task(resume_live_display())

        w, h = cfg.worker.cv2_cap.size_wh
        # HWC, RGBA, float32
        imbuffer = np.zeros((h, w, 3), dtype=np.float32)
        gui_task = asyncio.create_task(gui_loop(imbuffer))

        def update_imbuffer(results, img):
            imbuffer[...] = img[..., ::-1] / 255
            mp_results = results.get("mp_pose")
            if not mp_results is None:
                mp_drawing.draw_landmarks(
                    imbuffer,
                    mp_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

        tasks = deque()
        async for results, img in rlloop(
            cfg.worker.cv2_cap.fps, iterator=worker.next(), update_func=vid_loop
        ):
            if not dpg.is_dearpygui_running():
                break
            if img is None:
                continue
            tasks.append(
                asyncio.create_task(asyncio.to_thread(update_imbuffer, results, img))
            )
            await trim_task_queue(tasks, 30)

        try:
            resume_task.cancel()
            await resume_task
        except:
            pass
        finally:
            rich_live_display.stop()

        await gui_task

    # uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")


# TODO: Configuration System
# copy template config folder to parent directory
# add template config folder to search path
# compose app config using both builtin and external config groups


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

            log.info(
                f":smiley: hewwo world! :eggplant: JHTech's nicepipe [red]v{__version__}[/red]!",
                extra={"markup": True, "highlighter": None},
            )

            log.debug(f"Config: {cfg}")
            log.debug(f"Cwd: {os.getcwd()}")

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


# TODO: Given HydraConf wont work. Test omegaConf first. then write using omegaConf a shallow shadow of Hydra (similar to Odyssey lmao)
# At least for the next version of Odyssey, given portable trainers arent in scope yet, Hydra can be used.

# TODO: investigate Rich to_svg and to_html methods for a potentially easy way for making the log panel instead of going full-on terminal emulator
