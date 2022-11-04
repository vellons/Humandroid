# Sent data via websocket to https://github.com/vellons/SimplePYBotSDK
import cv2
import time
from PoseInterpreter.poseInterpreterSimplePYBotSDK import PoseInterpreterSimplePyBotSDK

WEBSOCKET_HOST = "ws://192.168.1.131:65432"

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

    start_time = time.time()
    frame_counter = 0
    fps = ""

    poseInterpreter = PoseInterpreterSimplePyBotSDK(
        config_path="configurations/simple_humandroid.json",
        host=WEBSOCKET_HOST,
        display_face_connections=True,
        calc_z=False
    )
    poseInterpreter.PLOT_ANIMATED_AZIMUTH = True

    while camera.isOpened():
        success, image = camera.read()  # Get image
        if not success:
            continue

        image = cv2.flip(image, 1)  # Flip image for selfie-view
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR image to RGB
        image.flags.writeable = False  # To improve performance, pass by reference

        # PoseInterpreter
        poseInterpreter.process_pose_landmark(image)
        poseInterpreter.draw_landmarks(image)
        poseInterpreter.process_angles()
        poseInterpreter.process_matching_pose()
        poseInterpreter.draw_angles(image)
        poseInterpreter.send_ptp_with_websocket()
        # print("{} matching_pose={}".format(poseInterpreter.computed_ptp, poseInterpreter.matching_pose))
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
        poseInterpreter.draw_plot(image, x_offset=0, y_offset=668, scale=1.5)  # This will downgrade fps

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB image to BGR
        cv2.putText(image, fps, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
        cv2.namedWindow("Elettra Robotics Lab", cv2.WINDOW_NORMAL)
        cv2.imshow("Elettra Robotics Lab", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    camera.release()
