import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0
    mediapipe = None

    def __init__(self):
        Camera.set_video_source(0)
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        print('Camera OK')

        while True:
            # read current frame
            success, image = camera.read()
            yield image
