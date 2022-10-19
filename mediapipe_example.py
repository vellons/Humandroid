# Mediapipe example from official documentation
import time
import cv2
import mediapipe as mp


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    start_time = time.time()
    frame_counter = 0
    fps = ""

    # For webcam input:
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while camera.isOpened():
            success, image = camera.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Plot pose world landmarks.
            # mp_drawing.plot_landmarks(
            #    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)

            # Calc FPS average over multiple frame
            frame_counter += 1
            if (time.time() - start_time) > 0.33:
                fps = "FPS: {:.2f}".format(frame_counter / (time.time() - start_time))
                frame_counter = 0
                start_time = time.time()
            cv2.putText(image, fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show image
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    camera.release()
