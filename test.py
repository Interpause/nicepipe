import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# @profile
def main():
    # For webcam input:
    cap = cv2.VideoCapture(1)
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0, 1 or 2 (0 or 1 is okay)
            smooth_landmarks=True,
            enable_segmentation=True,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        pbar = tqdm()
        print(f"w={cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, h={cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, fps={cap.get(cv2.CAP_PROP_FPS)}")
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # TODO extract some type hinting out...
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
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(1) & 0xFF == 27:
                break
            pbar.update()
    cap.release()


if __name__ == '__main__':
    main()
