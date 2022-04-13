import cv2
import asyncio
from tqdm import tqdm
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose
#import mediapipe.framework.formats.landmark_pb2

from mediapiping import Worker

# TODO: is the websocket part of or external of Worker?


async def main():
    async with Worker() as worker:
        pbar = tqdm()
        for results, img in worker.next():
            await asyncio.sleep(0.016)
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
                        pbar.set_description(
                            f'arm up, left_elbow: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y}, left_wrist: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y}')
                    else:
                        pbar.set_description(
                            f'arm down, left_elbow: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y}, left_wrist: {landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y}')
                else:
                    pbar.set_description('arm out of bounds')
            else:
                pbar.set_description('no human')

            # Draw the pose annotation on the image.
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(img, 1))
            if cv2.waitKey(1) & 0xFF == 27:
                break
            pbar.update()


if __name__ == '__main__':
    asyncio.run(main())
