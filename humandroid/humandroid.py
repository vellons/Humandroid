import math

import cv2
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.drawing_utils import PRESENCE_THRESHOLD
from mediapipe.python.solutions.drawing_utils import VISIBILITY_THRESHOLD
import numpy as np
import matplotlib.pyplot as plt

ANGLES = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),  # gomito sx
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),  # gomito dx
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),  # spalla sx
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW)  # spalla dx
]


class Humandroid:
    """
    Human pose detection to control humandroid robots.
    Works with MediaPipe technology

    Please refer to https://mediapipe.dev/ for more info.
    """

    _mp_solution_pose = mp.solutions.pose
    _mp_solution_drawing = mp.solutions.drawing_utils
    _mp_pose = None

    detected_pose = None

    # Use inside draw_3d_environment()
    _plt_fig = None

    def __init__(self, static_image_mode=False, upper_body_only=False):
        """
        Initializes Humandroid object and MediaPipe.

        Args:
          static_image_mode: Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream. See details in
            https://solutions.mediapipe.dev/pose#static_image_mode.
          upper_body_only: Whether to track the full set of 33 pose landmarks or
            only the 25 upper-body pose landmarks. See details in
            https://solutions.mediapipe.dev/pose#upper_body_only.
        """

        print("Humandroid initialization")
        self._mp_pose = self._mp_solution_pose.Pose(
            static_image_mode=static_image_mode,
            upper_body_only=upper_body_only,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_pose_landmark(self, image: np.ndarray):
        """
        Processes an RGB image and returns the pose landmarks on the most prominent person detected.

        Args:
          image: An RGB image represented as a numpy ndarray.

        Returns:
          A NamedTuple object with a "pose_landmarks" field that contains the pose
          landmarks on the most prominent person detected.
        """

        try:
            self.detected_pose = self._mp_pose.process(image)
        finally:
            return self.detected_pose

    def draw_landmarks(self, image: np.ndarray):
        """
        Draws the landmarks and the connections on the image.

        Args:
          image: A three channel RGB image represented as numpy ndarray.
        """

        if not self.detected_pose.pose_landmarks:
            return

        self._mp_solution_drawing.draw_landmarks(
            image,
            self.detected_pose.pose_landmarks,
            self._mp_solution_pose.POSE_CONNECTIONS,
            DrawingSpec(color=(255, 0, 0)),
            DrawingSpec(color=(0, 255, 0))
        )

    def _calc_angle_if_safe(self, points):
        """
        Calc angle between three points.
        Note that angle can be calculated also if the center is not visible

        Args:
          points: A tuple with 3 point that will be used to calc the angle.
        """

        pose = self.detected_pose.pose_landmarks.landmark
        for p in points:
            # Check if all 3 landmarks is enough present and visible
            if pose[p].HasField('visibility') and \
                    pose[p].visibility < VISIBILITY_THRESHOLD or \
                    pose[p].HasField('presence') and \
                    pose[p].presence < PRESENCE_THRESHOLD:
                return -1

        # Calc angle
        a = math.atan2(pose[points[0]].y - pose[points[1]].y, pose[points[0]].x - pose[points[1]].x)
        b = math.atan2(pose[points[2]].y - pose[points[1]].y, pose[points[2]].x - pose[points[1]].x)
        result = math.fabs(math.degrees(a - b))  # Make angle always positive
        if result > 180:
            result = (360.0 - result)
        return int(result)

    def draw_angles(self, image: np.ndarray):
        """
        Draws angles to focus on the image.

        Args:
          image: A three channel RGB image represented as numpy ndarray.
        """

        if not self.detected_pose.pose_landmarks:
            return
        image_height, image_width, _ = image.shape

        for angle in ANGLES:
            a = self._calc_angle_if_safe(angle)
            if a >= 0:
                angle_x = self.detected_pose.pose_landmarks.landmark[angle[1]].x * image_width
                angle_y = self.detected_pose.pose_landmarks.landmark[angle[1]].y * image_height
                margin = 15
                if margin <= angle_x <= image_width - margin and margin <= angle_y <= image_height - margin:
                    text_position = (int(angle_x) - 15, int(angle_y) + 20)
                    cv2.putText(image, str(a), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def draw_3d_environment(self):
        if self._plt_fig is None:
            # Initialize at first execution
            self._plt_fig = plt.figure()

        self._plt_fig.clf()  # Clear figure and configure plot
        ax = plt.axes(projection="3d")
        ax.set_xlim([2, -1])
        ax.set_ylim([2.5, -0.5])
        ax.set_zlim([-3, 3])
        ax.view_init(270, 90)
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.get_zaxis().set_ticklabels([])

        if self.detected_pose.pose_landmarks:
            pose = self.detected_pose.pose_landmarks.landmark
            num_landmarks = len(pose)
            # Draws the connections
            for connection in self._mp_solution_pose.POSE_CONNECTIONS:
                start = connection[0]
                end = connection[1]

                if not (0 <= start < num_landmarks and 0 <= end < num_landmarks):
                    continue

                ax.plot([pose[start].x, pose[end].x], [pose[start].y, pose[end].y], [pose[start].z, pose[end].z])

        # Draw the renderer
        self._plt_fig.canvas.draw()

        # Plot canvas to a three channel RGB image represented as numpy ndarray
        image = np.fromstring(self._plt_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(self._plt_fig.canvas.get_width_height()[::-1] + (3,))
        return image
