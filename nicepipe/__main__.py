from __future__ import annotations
import sys
import os
import logging
import asyncio

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
from nicepipe.utils import enable_fancy_console, add_fps_task, rlloop, change_cwd
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
                log.debug(f"DLLs:\t{nicepipe.utils.cuda.DLLs}")
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


async def loop(cfg: DictConfig):
    async with Worker(**cfg.worker) as worker:
        demo_loop = add_fps_task("gui loop")

        asyncio.create_task(resume_live_display())
        async for results, img in rlloop(
            60,
            iterator=worker.next(),
            update_func=demo_loop,
        ):
            if img is None:
                continue
            mp_results = results.get("mp_pose")
            if not mp_results is None:
                img.flags.writeable = True
                mp_drawing.draw_landmarks(
                    img,
                    mp_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow("MediaPipe Pose", cv2.flip(img, 1))
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
        rich_live_display.stop()

    # uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
    # with create_gui():
    #     while dpg.is_dearpygui_running():
    #         dpg.render_dearpygui_frame()
    #         return


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

            log.debug(f"Config:\t{cfg}")
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
