import os
import sys
import yaml

import cv2
import asyncio
from multiprocessing import freeze_support
from rich.prompt import Confirm

import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose

from nicepipe import Worker, __version__
from nicepipe.utils import enable_fancy_console, add_fps_task, update_status, rlloop
import nicepipe.utils.uvloop

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# enable/disable the local preview/test window
LOCAL_TEST_ENABLED = False
# enable/disable tensorflow/CUDA functionality
CUDA_ENABLED = True


# TODO: proper config read/write (maybe using omegaconf or hydra) and actually merge defaults or smth
def get_config():
    if not os.path.exists("./config.yml"):
        log.warning("config.yml not found! creating...")
        cfg = dict(
            # https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options
            mp_cfg=dict(
                static_image_mode=False,
                # NOTE: run each model complexity once to download model files
                model_complexity=1,  # 0, 1 or 2 (0 or 1 is okay)
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ),
            # https://docs.python.org/3/library/logging.html#logging-levels
            log_level=20,
            worker_cfg=dict(
                cv2_source=0,
                # default cv2 capture source on windows has an unsilenceable warning... but dshow (the alternative) lags..
                cv2_cap_api=cv2.CAP_ANY,
                cv2_size_wh=(1920, 1080),
                cv2_enc_flags=[],
                cv2_enc_format="jpeg",
                mp_size_wh=(640, 360),
                max_fps=60,
                lock_fps=True,
                wss_host="localhost",
                wss_port=8080,
            ),
            main_fps=60,
            no_local_test=False,
        )
        with open("./config.yml", "w") as f:
            yaml.safe_dump(cfg, f)
            log.warning("Program will now exit to allow you to edit config.yml.")
            raise KeyboardInterrupt

    with open("./config.yml") as f:
        cfg = yaml.safe_load(f)
    log.debug("config.yml loaded:")
    log.debug(cfg)
    logging.getLogger().setLevel(cfg["log_level"])
    return cfg


async def main(cfg, live):
    async def restart_live_console():
        """exists solely to prevent tflite's log messages from interrupting the fancy logs"""
        await asyncio.sleep(4)
        live.console.line(live.console.height)
        live.transient = True
        live.start()

    fancy = asyncio.create_task(restart_live_console())

    log.info(
        f":smiley: hewwo world! :eggplant: JHTech's nicepipe [red]v{__version__}[/red]!",
        extra={"markup": True, "highlighter": None},
    )
    async with Worker(
        cv2_source=cfg["worker_cfg"]["cv2_source"],
        cv2_cap_api=cfg["worker_cfg"]["cv2_cap_api"],
        cv2_size_wh=cfg["worker_cfg"]["cv2_size_wh"],
        mp_size_wh=cfg["worker_cfg"]["mp_size_wh"],
        cv2_enc_flags=cfg["worker_cfg"]["cv2_enc_flags"],
        cv2_enc_format=cfg["worker_cfg"]["cv2_enc_format"],
        mp_pose_cfg=cfg["mp_cfg"],
        max_fps=cfg["worker_cfg"]["max_fps"],
        lock_fps=cfg["worker_cfg"]["lock_fps"],
        wss_host=cfg["worker_cfg"]["wss_host"],
        wss_port=cfg["worker_cfg"]["wss_port"],
    ) as worker:
        if LOCAL_TEST_ENABLED:
            demo_loop = add_fps_task("demo loop")

            async for results, img in rlloop(
                cfg["main_fps"],
                iterator=worker.next(),
                update_func=demo_loop,
            ):
                if img is None:
                    continue
                if not results is None:
                    # landmarks are normalized to [0,1]
                    if landmarks := results.pose_landmarks:
                        # mediapipe attempts to predict pose even outside of image 0_0
                        # either can check if it exceeds image bounds or visibility

                        ley = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
                        lwy = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
                        if 0 < ley < 1 and 0 < lwy < 1:
                            dy = (
                                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
                                - landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
                            )

                            if abs(dy) < 0.2:
                                update_status(
                                    f"arm up, left_elbow: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y}, left_wrist: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y}"
                                )

                            else:
                                update_status(
                                    f"arm down, left_elbow: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y}, left_wrist: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y}"
                                )
                        else:
                            update_status("arm out of bounds")
                    else:
                        update_status("no human")

                    img.flags.writeable = True
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow("MediaPipe Pose", cv2.flip(img, 1))
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    return
        update_status("Nice Logs")
        await asyncio.gather(worker.join(), fancy)


if __name__ == "__main__":
    freeze_support()  # needed on windows for multiprocessing
    with enable_fancy_console(start_live=False) as live:
        try:
            if sys.platform.startswith("win"):
                log.warning(
                    "Windows detected! Disable Windows Game Mode else worker will lag when not in foreground!"
                )

            cfg = get_config()

            try:
                import nicepipe.utils.cuda  # noqa

                if not cfg["no_local_test"]:
                    if Confirm.ask("Run CUDA Test?", default=False):
                        import tensorflow as tf  # noqa

                        # import torch # torch.cuda.is_available()
                        log.debug(f"DLLs loaded: {nicepipe.utils.cuda.dlls}")
                        log.info(
                            f'Torch CUDA: disabled, Tensorflow CUDA: {len(tf.config.list_physical_devices("GPU")) > 0}'
                        )
            # means CUDA & Tensorflow disabled
            except Exception as e:
                CUDA_ENABLED = False
                if not isinstance(e, ModuleNotFoundError):
                    log.warning(e)

            if not cfg["no_local_test"]:
                LOCAL_TEST_ENABLED = Confirm.ask("Run Local Test?", default=False)

            asyncio.run(main(cfg, live))
        except KeyboardInterrupt:
            pass
        finally:
            log.info("Stopped!")
