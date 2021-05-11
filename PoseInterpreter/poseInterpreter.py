import cv2
import math
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.drawing_utils import VISIBILITY_THRESHOLD
import numpy as np
import json

from PoseInterpreter.body_landmark import PoseInterpreterBodyLandmark


class PoseInterpreter:
    """
    Human pose detection to control humandroid robots.
    Works with MediaPipe technology

    Please refer to https://mediapipe.dev/ for more info about the pose processing.
    """

    _calc_z = False
    _upper_body_only = False

    _mp_solution_pose = mp.solutions.pose
    _mp_solution_drawing = mp.solutions.drawing_utils
    _mp_pose = None  # MediaPipe object

    detected_pose = None  # Data directly form MediaPipe
    computed_pose = None  # Computed data for humandroid robot with servo angles
    computed_ptp = {}  # Computed point to point pose (for humandroid robot motors)

    _plt_fig = None  # Used inside draw_3d_environment()

    def __init__(self, config_path: str, static_image_mode: bool = False, upper_body_only: bool = False,
                 face_connections: bool = True, calc_z: bool = False):
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

        print("PoseInterpreter initialization")
        self.config_path = config_path
        self._upper_body_only = upper_body_only
        self._face_connections = face_connections
        self._calc_z = calc_z

        try:
            with open(self.config_path) as f:
                self.configuration = json.load(f)
                for key, j in self.configuration["joints"].items():
                    angle = (PoseLandmark[j["points"][0]], PoseLandmark[j["points"][1]], PoseLandmark[j["points"][2]])
                    j["pose_landmarks"] = angle
        except Exception as e:
            print("Failed to load configuration. Impossible to calc angles. Error: {}".format(e))
            self.configuration = {
                "joints": {}
            }

        # print("Configuration", self.configuration)

        self._mp_connections = self._mp_solution_pose.POSE_CONNECTIONS
        if self._upper_body_only:
            self._mp_connections = self._mp_solution_pose.UPPER_BODY_POSE_CONNECTIONS

        if not self._face_connections:
            self._mp_connections = list(self._mp_connections)
            self._mp_connections.remove((PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER))
            self._mp_connections.remove((PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE))
            self._mp_connections.remove((PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER))
            self._mp_connections.remove((PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR))
            self._mp_connections.remove((PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER))
            self._mp_connections.remove((PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE))
            self._mp_connections.remove((PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER))
            self._mp_connections.remove((PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR))
            self._mp_connections.remove((PoseLandmark.MOUTH_RIGHT, PoseLandmark.MOUTH_LEFT))
            self._mp_connections = self._mp_connections

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
          A dict object with a "pose_landmarks" field that contains the pose
          landmarks on the most prominent person detected.
        """

        try:
            self.detected_pose = self._mp_pose.process(image)
        finally:
            # Create internal computed_pose object
            self.computed_pose = {
                "pose_landmarks": []
            }

            if not self.detected_pose.pose_landmarks:  # If not pose detect return a fake position nodes
                for idx, pose in enumerate(PoseLandmark):
                    landmark = PoseInterpreterBodyLandmark(
                        identifier=idx,
                        name=str(PoseLandmark.value2member_map_[idx])
                    )
                    self.computed_pose["pose_landmarks"].append(landmark)
                return self.computed_pose

            for idx, pose in enumerate(self.detected_pose.pose_landmarks.landmark):  # Copy pose info in landmark obj
                landmark = PoseInterpreterBodyLandmark(
                    identifier=idx,
                    name=str(PoseLandmark.value2member_map_[idx]),
                    x=pose.x,
                    y=pose.y,
                    z=pose.z,
                    visibility=pose.visibility,
                    angle=None,
                    z_angle=None
                )
                self.computed_pose["pose_landmarks"].append(landmark)

            return self.computed_pose

    def draw_landmarks(self, image: np.ndarray):
        """
        Draws the landmarks and the connections on the passed image.

        Args:
          image: A three channel RGB image represented as numpy ndarray.
        """

        if not self.detected_pose.pose_landmarks:
            return

        self._mp_solution_drawing.draw_landmarks(
            image,
            self.detected_pose.pose_landmarks,
            self._mp_connections,
            DrawingSpec(color=(255, 0, 0)),
            DrawingSpec(color=(0, 255, 0))
        )

    def _calc_angle_if_safe(self, points: tuple):
        """
        Calc angle between three points.
        Note that angle can be calculated also if the center is not visible.

        Args:
          points: A tuple with 3 point that will be used to calc the angle. points[1] is the center.

        Returns:
          The calculated angle or None if is not possible to calc.
        """

        landmarks = self.computed_pose["pose_landmarks"]
        for p in points:
            # Check if all 3 landmarks is enough present and visible
            # if landmarks[p].visibility < VISIBILITY_THRESHOLD or landmarks[p].presence < PRESENCE_THRESHOLD:
            if landmarks[p].visibility < VISIBILITY_THRESHOLD:
                return None

        # Calc angle
        a = math.atan2(landmarks[points[0]].y - landmarks[points[1]].y, landmarks[points[0]].x - landmarks[points[1]].x)
        b = math.atan2(landmarks[points[2]].y - landmarks[points[1]].y, landmarks[points[2]].x - landmarks[points[1]].x)
        result = math.degrees(a - b)
        if result < 0:
            result = (360.0 + result)
        return int(result)

    def _calc_z_angle_if_safe(self, points: tuple):
        """
        Calc Z angle between three points.
        MediaPipe Z is experimental and not currently stable.
        Note that angle can be calculated also if the center is not visible.

        Args:
          points: A tuple with 3 point that will be used to calc the angle. points[1] is the center.

        Returns:
          The calculated angle or None if is not possible to calc.
        """

        landmarks = self.computed_pose["pose_landmarks"]
        for p in points:
            # Check if all 3 landmarks is enough present and visible
            if landmarks[p].visibility < VISIBILITY_THRESHOLD:
                return None

        # Calc z angle
        a = math.atan2(landmarks[points[0]].y - landmarks[points[1]].y, landmarks[points[0]].z - landmarks[points[1]].z)
        b = math.atan2(landmarks[points[2]].y - landmarks[points[1]].y, landmarks[points[2]].z - landmarks[points[1]].z)
        result = math.degrees(a - b)
        if result < 0:
            result = (360.0 + result)
        return int(result)

    def process_angles(self):
        """
        Calculate focus angles from the current person detected.
        Also compute motors angle dict to move a humandroid robot.

        Returns:
          A dict object with a "pose_landmarks" field that contains the pose
          landmarks on the most prominent person detected and calculated angles.
        """
        ptp = {}
        for key, j in self.configuration["joints"].items():

            if j["type"] == "xy":
                a = self._calc_angle_if_safe(j["pose_landmarks"])
                if a is not None:
                    if j["orientation"] == "indirect":
                        a = 360 - a + j["offset"]
                    else:
                        a = a + j["offset"]
                    ptp[key] = a
                    self.computed_pose["pose_landmarks"][j["pose_landmarks"][1]].angle = a

            elif self._calc_z and j["type"] == "z":
                a = self._calc_z_angle_if_safe(j["pose_landmarks"])
                if a is not None:
                    if j["orientation"] == "indirect":
                        a = 360 - a + j["offset"]
                    else:
                        a = a + j["offset"]
                    ptp[key] = a
                    self.computed_pose["pose_landmarks"][j["pose_landmarks"][1]].z_angle = a

            elif j["type"] == "math" and "math_angle" in j:
                if j["math_angle"] == "head_z":
                    left = self.computed_pose["pose_landmarks"][j["pose_landmarks"][0]].x
                    nose = self.computed_pose["pose_landmarks"][j["pose_landmarks"][1]].x
                    right = self.computed_pose["pose_landmarks"][j["pose_landmarks"][2]].x
                    if left is not None and nose is not None and right is not None:
                        math_angle = (nose - left) / (right - left) * 100

                        if j["orientation"] == "indirect":
                            math_angle = 360 - math_angle + j["offset"]
                        else:
                            math_angle = math_angle + j["offset"]
                        ptp[key] = int(math_angle)
                        self.computed_pose["pose_landmarks"][j["pose_landmarks"][1]].math_angle = math_angle

        self.computed_ptp = ptp
        self._post_process_angles()
        return self.computed_pose

    def _post_process_angles(self):
        """
        Method that check if all joint are in the correct position and not break the servomotors.
        """

        # Crossing arms (left)
        if self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_ELBOW].angle is not None:
            left_elbow = self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_ELBOW]
            left_shoulder = self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_SHOULDER]
            left_shoulder_angle = 0
            if left_shoulder.angle is not None:
                left_shoulder_angle = left_shoulder.angle

            if left_elbow.angle < -7 and left_shoulder_angle < 60:
                angle = 60
                if "l_shoulder_x" in self.computed_ptp:
                    angle = self.computed_ptp["l_shoulder_x"] if self.computed_ptp["l_shoulder_x"] > angle else angle
                self.computed_ptp["l_shoulder_x"] = angle
            elif not self._calc_z:
                self.computed_ptp["l_shoulder_x"] = 0

        # Crossing arms (right)
        if self.computed_pose["pose_landmarks"][PoseLandmark.RIGHT_ELBOW].angle is not None:
            right_elbow = self.computed_pose["pose_landmarks"][PoseLandmark.RIGHT_ELBOW]
            right_shoulder = self.computed_pose["pose_landmarks"][PoseLandmark.RIGHT_SHOULDER]
            right_shoulder_angle = 0
            if right_shoulder.angle is not None:
                right_shoulder_angle = right_shoulder.angle

            if right_elbow.angle < -7 and right_shoulder_angle < 60:
                angle = 60
                if "r_shoulder_x" in self.computed_ptp:
                    angle = self.computed_ptp["r_shoulder_x"] if self.computed_ptp["r_shoulder_x"] > angle else angle
                self.computed_ptp["r_shoulder_x"] = angle
            elif not self._calc_z:
                self.computed_ptp["r_shoulder_x"] = 0

    def draw_angles(self, image: np.ndarray):
        """
        Draws angles processed angles on the passed image.

        Args:
          image: A three channel RGB image represented as numpy ndarray.
        """

        image_height, image_width, _ = image.shape
        font = 0.6
        margin = 15

        for landmark in self.computed_pose["pose_landmarks"]:
            if landmark.angle is not None:  # Draw calculated angles
                angle_x = landmark.x * image_width
                angle_y = landmark.y * image_height
                if margin <= angle_x <= image_width - margin and margin <= angle_y <= image_height - margin:
                    pos = (int(angle_x) - 15, int(angle_y) + 20)
                    cv2.putText(image, str(int(landmark.angle)), pos,
                                cv2.FONT_HERSHEY_SIMPLEX, font, (255, 0, 0), 2)

            if self._calc_z and landmark.z_angle is not None:  # Draw calculated z angles if enabled
                angle_x = landmark.x * image_width
                angle_y = landmark.y * image_height
                if margin <= angle_x <= image_width - margin and margin <= angle_y <= image_height - margin:
                    pos = (int(angle_x) + 20, int(angle_y) + 20)
                    cv2.putText(image, str(landmark.z_angle), pos,
                                cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 2)

            if landmark.math_angle is not None:  # Draw calculated math_angles
                angle_x = landmark.x * image_width
                angle_y = landmark.y * image_height
                if margin <= angle_x <= image_width - margin and margin <= angle_y <= image_height - margin:
                    pos = (int(angle_x) - 5, int(angle_y) + 20)
                    cv2.putText(image, str(int(landmark.math_angle)), pos,
                                cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 255), 2)

    def draw_3d_environment(self):
        """
        Draws landmarks and connections with matplotlib in a new image.

        Returns:
          A three channel RGB image represented as numpy ndarray with the body in a
          simulated environment made with matplotlib.
        """

        if self._plt_fig is None:
            # Initialize at first execution
            import matplotlib.pyplot as plt
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

        if self.detected_pose.pose_landmarks:  # If MediaPipe detect a body pose

            # Zoom in/out axis
            top = self.computed_pose["pose_landmarks"][PoseLandmark.RIGHT_EYE]
            bottom = self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_HIP]
            if not self._upper_body_only:
                bottom = self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_ANKLE]
            ax.set_ylim([bottom.y + 0.5, top.y - 0.5])

            landmarks = self.computed_pose["pose_landmarks"]
            num_landmarks = len(landmarks)
            # Draw the connections
            for connection in self._mp_connections:
                start = connection[0]
                end = connection[1]

                if not (0 <= start < num_landmarks and 0 <= end < num_landmarks):
                    continue

                ax.plot(
                    [landmarks[start].x, landmarks[end].x],
                    [landmarks[start].y, landmarks[end].y],
                    [landmarks[start].z, landmarks[end].z]
                )

            # Draw calculated angles
            for landmark in landmarks:
                if landmark.angle is not None:
                    ax.text(landmark.x, landmark.y, landmark.z, str(landmark.angle), fontsize=8)

        # Draw the render
        self._plt_fig.canvas.draw()

        # Plot canvas to a three channel RGB image represented as numpy ndarray
        image = np.fromstring(self._plt_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(self._plt_fig.canvas.get_width_height()[::-1] + (3,))
        image = image[70:400, 150:500]
        return image
