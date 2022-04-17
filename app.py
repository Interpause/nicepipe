# NOTE: IF RUNNING ON WINDOWS, DISABLE GAME MODE!!! ELSE LAG WHEN SERVER IS NOT FOREGROUND

import logging
from rich import print
from rich.markup import escape
from rich.prompt import Confirm
from multiprocessing import freeze_support

import yaml
import os
import cv2
import asyncio
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose

import mediapiping
from mediapiping import Worker, rlloop
from mediapiping.rich import enable_fancy_console, rate_bar, layout, live, console


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def get_config():
    if not os.path.exists('./config.yml'):
        log.debug('config.yml not found! creating...')
        cfg = dict(
            # https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options
            mp_cfg=dict(
                static_image_mode=False,
                # NOTE: run each model complexity once to download model files
                model_complexity=1,  # 0, 1 or 2 (0 or 1 is okay)
                smooth_landmarks=True,
                enable_segmentation=True,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ),
            # https://docs.python.org/3/library/logging.html#logging-levels
            log_level=20,
            worker_cfg=dict(
                cv2_args=[0],
                cv2_height=480,
                cv2_width=640,
                max_fps=30,
                wss_host='localhost',
                wss_port=8080
            ),
            main_fps=60,
            no_local_test=False,
        )
        with open('./config.yml', 'w') as f:
            yaml.safe_dump(cfg, f)

    with open('./config.yml') as f:
        cfg = yaml.safe_load(f)
    log.debug('config.yml loaded')
    log.debug(cfg)
    logging.getLogger().setLevel(cfg['log_level'])
    return cfg


# enable/disable the local preview/test window
LOCAL_TEST_ENABLED = False
# enable/disable tensorflow/CUDA functionality
CUDA_ENABLED = True


async def main(cfg):
    try:
        async def restart_live_console():
            await asyncio.sleep(4)
            console.line(16)
            # console.clear()
            live.transient = False
            live.start()
        asyncio.create_task(restart_live_console())
        log.info(f":smiley: hewwo world! :eggplant: JHTech's Mediapiping [red]v{mediapiping.__version__}[/red]!", extra={
                 "markup": True, "highlighter": None})
        async with Worker(
            cv2_args=cfg['worker_cfg']['cv2_args'],
            cv2_height=cfg['worker_cfg']['cv2_height'],
            cv2_width=cfg['worker_cfg']['cv2_width'],
            mp_pose_cfg=cfg['mp_cfg'],
            max_fps=cfg['worker_cfg']['max_fps'],
            wss_host=cfg['worker_cfg']['wss_host'],
            wss_port=cfg['worker_cfg']['wss_port'],
        ) as worker:
            if not LOCAL_TEST_ENABLED:
                while True:
                    await asyncio.sleep(0)
            else:
                main_loop = rate_bar.add_task("main loop", total=float('inf'))

                async for results, img in rlloop(cfg['main_fps'], iter=worker.next(), update_func=lambda: rate_bar.update(main_loop, advance=1)):
                    if results is None:
                        continue

                    # landmarks are normalized to [0,1]
                    if landmarks := results.pose_landmarks:
                        # mediapipe attempts to predict pose even outside of image 0_0
                        # either can check if it exceeds image bounds or visibility

                        ley = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
                        lwy = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
                        if 0 < ley < 1 and 0 < lwy < 1:
                            dy = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y - \
                                landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y

                            if(abs(dy) < 0.2):
                                layout['Info']['Misc'].update(
                                    f'arm up, left_elbow: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y}, left_wrist: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y}')

                            else:
                                layout['Info']['Misc'].update(
                                    f'arm down, left_elbow: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y}, left_wrist: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y}')
                        else:
                            layout['Info']['Misc'].update('arm out of bounds')
                    else:
                        layout['Info']['Misc'].update('no human')

                    img.flags.writeable = True
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    # Flip the image horizontally for a selfie-view display.
                    cv2.imshow('MediaPipe Pose', cv2.flip(img, 1))
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    freeze_support()  # needed on windows for multiprocessing
    with enable_fancy_console():
        live.stop()
        # uvloop only available on unix platform
        try:
            import uvloop  # type: ignore
            uvloop.install()
        # means we on windows
        except ModuleNotFoundError:
            log.warning(
                'Windows detected! Disable Windows Game Mode else worker will lag when not in foreground!')
            pass

        cfg = get_config()

        # if Confirm.ask("Run CUDA Test?", default=False):
        #     import torch
        #     print(
        #         f'Torch CUDA: {torch.cuda.is_available()}, Tensorflow CUDA: {len(tf.config.list_physical_devices("GPU")) > 0}')

        try:
            import mediapiping.cuda  # noqa
            if not cfg['no_local_test']:
                if Confirm.ask("Run CUDA Test?", default=False):
                    log.debug(f'DLLs loaded: {mediapiping.cuda.dlls}')
                    import tensorflow as tf  # noqa
                    log.info(
                        f'Torch CUDA: disabled, Tensorflow CUDA: {len(tf.config.list_physical_devices("GPU")) > 0}')
        # means CUDA & Tensorflow disabled
        except ModuleNotFoundError:
            CUDA_ENABLED = False
            pass

        if not cfg['no_local_test']:
            LOCAL_TEST_ENABLED = Confirm.ask("Run Local Test?", default=False)

        asyncio.run(main(cfg))
