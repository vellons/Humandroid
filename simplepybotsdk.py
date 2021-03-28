# Sent data via websocket to https://github.com/vellons/SimplePYBotSDK
import cv2
import time
from PoseInterpreter.poseInterpreterSimplePYBotSDK import PoseInterpreterSimplePyBotSDK

WEBSOCKET_HOST = "ws://192.168.1.131:65432"

if __name__ == '__main__':

    camera = cv2.VideoCapture(0)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    start_time = time.time()
    frame_counter = 0
    fps = ""

    poseInterpreter = PoseInterpreterSimplePyBotSDK(
        config_path="configurations/simple_humandroid.json",
        host=WEBSOCKET_HOST,
        upper_body_only=False,
        calc_z=False
    )

    while camera.isOpened():
        success, image = camera.read()  # Get image
        if not success:
            continue

        image = cv2.flip(image, 1)  # Flip image for selfie-view
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR image to RGB
        image.flags.writeable = False  # To improve performance

        # PoseInterpreter
        poseInterpreter.process_pose_landmark(image)
        poseInterpreter.draw_landmarks(image)
        poseInterpreter.process_angles()
        poseInterpreter.draw_angles(image)
        poseInterpreter.send_ptp_with_websocket()
        print(poseInterpreter.computed_ptp)
        # print(poseInterpreter.computed_pose)
        # print(poseInterpreter.computed_pose["pose_landmarks"][13].angle)  # Print left elbow angle

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
        cv2.imshow("Humandroid body pose V1.0", image)

        # Show 3d environment - Comment this to go faster
        # three_d_env = humandroid.draw_3d_environment()
        # three_d_env = cv2.cvtColor(three_d_env, cv2.COLOR_RGB2BGR)  # RGB image to BGR
        # three_d_env = cv2.resize(three_d_env, (0, 0), fx=1.5, fy=1.5)  # Resize image
        # cv2.imshow('3D environment', three_d_env)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    camera.release()
