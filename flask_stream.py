import cv2
from flask import Flask, Response  # Not in requirements
from waitress import serve  # Not in requirements
from PoseInterpreter import poseInterpreterSimplePYBotSDK

flask_app = Flask(__name__)
flask_image = None

ADMIN_PAGE = """
<html>
<head>
<title>Elettra Robotics Lab</title>
</head>
<body>
<div>
<button onclick="location.href='/__on'" type="button">START SEND</button>
<button onclick="location.href='/__off'" type="button">STOP SEND</button>
</div>
<div>
<img src="/" width="100%">
</div>
</body>
</html>
"""


def gen():
    """Video streaming generator function."""
    while True:
        frame = flask_image

        if frame is None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n \r\n')
            continue
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@flask_app.route('/')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/__admin')
def admin():
    return ADMIN_PAGE


@flask_app.route('/__on')
def start_websocket():
    poseInterpreterSimplePYBotSDK.enable_websocket_send = True
    return ADMIN_PAGE


@flask_app.route('/__off')
def stop_websocket():
    poseInterpreterSimplePYBotSDK.enable_websocket_send = False
    return ADMIN_PAGE


def run_flask():
    serve(flask_app, host='0.0.0.0', port=80)
