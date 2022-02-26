import cv2

from camera_opencv2 import Camera

if __name__ == "__main__":

    camera = Camera()

    while True:
        image = camera.get_frame()

        image = cv2.flip(image, 1)  # Flip image for selfie-view
        cv2.namedWindow("Humandroid body pose V1.0", cv2.WINDOW_NORMAL)
        cv2.imshow("Humandroid body pose V1.0", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
