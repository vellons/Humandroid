# Sent data via websocket to https://github.com/vellons/SimplePYBotSDK
import cv2
import time
import threading
from flask import Flask, Response

from PoseInterpreter.poseInterpreterSimplePYBotSDK import PoseInterpreterSimplePyBotSDK
from camera_opencv2 import Camera

WEBSOCKET_HOST = "ws://192.168.1.131:65432"

app = Flask(__name__)
mediapipe_image = None


def gen():
    """Video streaming generator function."""
    while True:
        frame = mediapipe_image

        if frame is None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n \r\n')
            continue
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video_feed_a():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    app.run(host='0.0.0.0', threaded=True)


if __name__ == "__main__":
    start_time = time.time()
    frame_counter = 0
    fps = ""

    flask_thread = threading.Thread(target=run_flask, args=())
    flask_thread.name = "flask"
    flask_thread.daemon = True
    flask_thread.start()

    poseInterpreter = PoseInterpreterSimplePyBotSDK(
        config_path="configurations/simple_humandroid.json",
        host=WEBSOCKET_HOST,
        upper_body_only=False,
        face_connections=True,
        calc_z=False
    )

    camera = Camera()

    while True:
        image = camera.get_frame()
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
        mediapipe_image = image
        cv2.namedWindow("Humandroid body pose V1.0", cv2.WINDOW_NORMAL)
        cv2.imshow("Humandroid body pose V1.0", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
