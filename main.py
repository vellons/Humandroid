import cv2
import time
from humandroid import Humandroid

if __name__ == '__main__':

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    start_time = time.time()
    frame_counter = 0
    fps = ""

    humandroid = Humandroid()  # Initialize object

    while camera.isOpened():
        success, image = camera.read()  # Get image
        if not success:
            continue

        image = cv2.flip(image, 1)  # Flip image for selfie-view
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR image to RGB
        image.flags.writeable = False  # To improve performance

        # Humandroid
        humandroid.process_pose_landmark(image)
        humandroid.draw_landmarks(image)
        humandroid.process_angles()
        humandroid.draw_angles(image)
        print(humandroid.computed_pose)
        # print(humandroid.computed_pose["pose_landmarks"][13].angle)  # Print left elbow angle

        # Calc FPS average over multiple frame
        frame_counter += 1
        if (time.time() - start_time) > 0.33:
            fps = "FPS: {:.2f}".format(frame_counter / (time.time() - start_time))
            frame_counter = 0
            start_time = time.time()

        # Build output
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB image to BGR
        cv2.putText(image, fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Humandroid body pose V0.2', image)

        # Show 3d environment - Comment this to go faster
        three_d_env = humandroid.draw_3d_environment()
        three_d_env = cv2.cvtColor(three_d_env, cv2.COLOR_RGB2BGR)  # RGB image to BGR
        three_d_env = cv2.resize(three_d_env, (0, 0), fx=1.5, fy=1.5)  # Resize image
        cv2.imshow('3D environment', three_d_env)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    camera.release()
