import cv2
import time
from humandroid import Humandroid
import math

if __name__ == '__main__':

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    start_time = time.time()
    frame_counter = 0
    fps = ""

    humandroid = Humandroid()  # Initialize object
    show_text = False

    while camera.isOpened():
        success, image = camera.read()  # Get image
        if not success:
            continue

        # image = cv2.flip(image, 1)  # Flip image for selfie-view
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR image to RGB
        image.flags.writeable = False  # To improve performance

        # Humandroid
        humandroid.process_pose_landmark(image)
        humandroid.draw_landmarks(image)
        humandroid.process_angles()
        humandroid.draw_angles(image)
        print(humandroid.computed_pose)
        # print(humandroid.computed_pose["pose_landmarks"][13].angle)  # Print left elbow angle

        # Happy 2021 text if hands are close
        left_hand = humandroid.computed_pose["pose_landmarks"][21]
        right_hand = humandroid.computed_pose["pose_landmarks"][22]
        if left_hand.visibility >= 0.5 and right_hand.visibility >= 0.5:
            if math.fabs(left_hand.x - right_hand.x) < 0.07 and math.fabs(left_hand.y - right_hand.y) < 0.07:
                show_text = True

            if show_text:
                image_height, image_width, _ = image.shape
                pos = (int(right_hand.x * image_width) - 110, int(right_hand.y * image_height))
                cv2.putText(image, "HAPPY 2021!", pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 4)

        # Blur face
        nose = humandroid.computed_pose["pose_landmarks"][0]
        left_hear = humandroid.computed_pose["pose_landmarks"][7]
        right_hear = humandroid.computed_pose["pose_landmarks"][8]
        if nose.visibility >= 0.5 and left_hear.visibility >= 0.5 and right_hear.visibility >= 0.5:
            image_height, image_width, _ = image.shape
            image.flags.writeable = True
            x = int(nose.x * image_width)
            y = int(nose.y * image_height)
            m = int(math.fabs(right_hear.x * image_width - left_hear.x * image_width))  # Margin
            if m < 20:
                m = 20
            try:
                image[y - m:y + m, x - m:x + m] = cv2.blur(image[y - m:y + m, x - m:x + m], (m, m))
            except:
                print("Blur outside of image area")

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
