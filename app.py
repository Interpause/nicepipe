# NOTE: IF RUNNING ON WINDOWS, DISABLE GAME MODE!!! ELSE LAG WHEN SERVER IS NOT FOREGROUND
import logging
from rich import inspect

import cv2
import asyncio
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose

from mediapiping import Worker, rlloop
from mediapiping.rich import enable_fancy_console, rate_bar, layout


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# https://google.github.io/mediapipe/solutions/pose.html#cross-platform-configuration-options
mp_cfg = dict(
    static_image_mode=False,
    model_complexity=0,  # 0, 1 or 2 (0 or 1 is okay)
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# enable/disable the local preview/test window
LOCAL_TEST = True


async def main():
    log.info(':smiley: hewwo world! :eggplant:', extra={"markup": True})
    async with Worker(source=1, mp_pose_cfg=mp_cfg, max_fps=30) as worker:
        if not LOCAL_TEST:
            await asyncio.Future()
        else:
            main_loop = rate_bar.add_task("main loop", total=float('inf'))

            async for results, img in rlloop(60, iter=worker.next(), update_func=lambda: rate_bar.update(main_loop, advance=1)):
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


if __name__ == '__main__':
    # uvloop only available on unix platform
    try:
        import uvloop  # type: ignore
        uvloop.install()
    # means we on windows
    except ModuleNotFoundError:
        from multiprocessing import freeze_support
        freeze_support()  # needed on windows for multiprocessing
    with enable_fancy_console():
        asyncio.run(main())
