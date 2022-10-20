import cv2
from flask import Flask, Response  # Not in requirements
from waitress import serve  # Not in requirements

flask_app = Flask(__name__)
flask_image = None


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


def run_flask():
    serve(flask_app, host='0.0.0.0', port=80)
