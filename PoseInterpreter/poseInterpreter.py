import cv2
import math
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec, _normalize_color
from mediapipe.python.solutions.drawing_utils import _VISIBILITY_THRESHOLD, _PRESENCE_THRESHOLD
from mediapipe.python.solutions.drawing_styles import _POSE_LANDMARKS_LEFT, _POSE_LANDMARKS_RIGHT
import numpy as np
import matplotlib.pyplot as plt
import json

from PoseInterpreter.body_landmark import PoseInterpreterBodyLandmark


class PoseInterpreter:
    """
    Human pose detection to control humandroid robots.
    Works with MediaPipe technology.
    This class works with RGB values.

    Please refer to https://mediapipe.dev/ for more info about the pose processing.
    """

    _calc_z = False

    _mp_solution_pose = mp.solutions.pose
    _mp_solution_drawing = mp.solutions.drawing_utils
    _mp_drawing_styles = mp.solutions.drawing_styles
    _mp_pose = None  # MediaPipe result object

    detected_pose = None  # Data directly form MediaPipe
    computed_pose = None  # Computed data for humandroid robot with servo angles
    computed_ptp = {}  # Computed point to point pose (for humandroid robot motors)

    _plt_fig = None  # Used inside draw_3d_environment()
    _plt_azim = 1

    _FACE_LANDMARKS = frozenset([
        PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE,
        PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR, PoseLandmark.MOUTH_RIGHT,
        PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE,
        PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR, PoseLandmark.MOUTH_LEFT,
    ])

    # Style configuration
    POSE_LANDMARKS_THICKNESS = 8
    POSE_LANDMARKS_CIRCLE_RADIUS = 2
    POSE_LANDMARKS_LEFT_COLOR = (0, 138, 255)
    POSE_LANDMARKS_RIGHT_COLOR = (255, 165, 0)
    POSE_LANDMARKS_NOSE_COLOR = (255, 48, 48)

    ANGLE_TEXT_FONT = 1.2
    ANGLE_TEXT_THICKNESS = 4
    ANGLE_TEXT_MARGIN = 50
    ANGLE_TEXT_COLOR = (255, 0, 0)
    ANGLE_Z_TEXT_COLOR = (255, 255, 255)
    ANGLE_MATH_TEXT_COLOR = (0, 0, 255)

    PLOT_COLOR = (24, 240, 24)
    PLOT_THICKNESS = 2

    def __init__(self, config_path: str, static_image_mode: bool = False,
                 display_face_connections: bool = True, calc_z: bool = False):
        """
        Initializes PoseInterpreter object and MediaPipe.

        Args:
          static_image_mode: Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream. See details in
            https://solutions.mediapipe.dev/pose#static_image_mode.
        """

        print("PoseInterpreter initialization")
        self.config_path = config_path
        self._display_face_connections = display_face_connections
        self._calc_z = calc_z

        try:
            with open(self.config_path) as f:
                self.configuration = json.load(f)
                for key, j in self.configuration["joints"].items():
                    angle = (PoseLandmark[j["points"][0]], PoseLandmark[j["points"][1]], PoseLandmark[j["points"][2]])
                    j["pose_landmarks"] = angle
        except Exception as e:
            raise Exception("Failed to load configuration. Impossible to calc angles. Error: {}".format(e))

        # print("Configuration", self.configuration)

        self._mp_connections = self._mp_solution_pose.POSE_CONNECTIONS

        if not self._display_face_connections:
            self._mp_connections = list(self._mp_connections)
            self._mp_connections.remove((PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE))
            self._mp_connections.remove((PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER))
            self._mp_connections.remove((PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR))
            self._mp_connections.remove((PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER))
            self._mp_connections.remove((PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER))
            self._mp_connections.remove((PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE))
            self._mp_connections.remove((PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER))
            self._mp_connections.remove((PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR))
            self._mp_connections.remove((PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT))
            self._mp_connections = frozenset(self._mp_connections)

        self._mp_pose = self._mp_solution_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
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
                        name=str(idx),
                    )
                    self.computed_pose["pose_landmarks"].append(landmark)
                return self.computed_pose

            for idx, pose in enumerate(self.detected_pose.pose_landmarks.landmark):  # Copy pose info in landmark obj
                landmark = PoseInterpreterBodyLandmark(
                    identifier=idx,
                    name=str(idx),
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
            landmark_drawing_spec=self.get_pose_landmarks_style()
            # DrawingSpec(color=(255, 0, 0)),
            # DrawingSpec(color=(0, 255, 0))
            # landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style()
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
            # if landmarks[p].visibility < _VISIBILITY_THRESHOLD or landmarks[p].presence < PRESENCE_THRESHOLD:
            if landmarks[p].visibility < _VISIBILITY_THRESHOLD:
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
            if landmarks[p].visibility < _VISIBILITY_THRESHOLD:
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
        self.post_process_angles()
        return self.computed_pose

    def post_process_angles(self):
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
        font = self.ANGLE_TEXT_FONT
        margin = self.ANGLE_TEXT_MARGIN

        for landmark in self.computed_pose["pose_landmarks"]:
            if landmark.angle is not None:  # Draw calculated angles
                angle_x = landmark.x * image_width
                angle_y = landmark.y * image_height
                if margin <= angle_x <= image_width - margin and margin <= angle_y <= image_height - margin:
                    pos = (int(angle_x) - margin, int(angle_y) + margin + 5)
                    cv2.putText(image, str(int(landmark.angle)), pos,
                                cv2.FONT_HERSHEY_SIMPLEX, font, self.ANGLE_TEXT_COLOR, self.ANGLE_TEXT_THICKNESS)

            if self._calc_z and landmark.z_angle is not None:  # Draw calculated z angles if enabled
                angle_x = landmark.x * image_width
                angle_y = landmark.y * image_height
                if margin <= angle_x <= image_width - margin and margin <= angle_y <= image_height - margin:
                    pos = (int(angle_x) + margin, int(angle_y) + margin)
                    cv2.putText(image, str(landmark.z_angle), pos,
                                cv2.FONT_HERSHEY_SIMPLEX, font, self.ANGLE_Z_TEXT_COLOR, self.ANGLE_TEXT_THICKNESS)

            if landmark.math_angle is not None:  # Draw calculated math_angles
                angle_x = landmark.x * image_width
                angle_y = landmark.y * image_height
                if margin <= angle_x <= image_width - margin and margin <= angle_y <= image_height - margin:
                    pos = (int(angle_x) - 5, int(angle_y) + margin + 5)
                    cv2.putText(image, str(int(landmark.math_angle)), pos,
                                cv2.FONT_HERSHEY_SIMPLEX, font, self.ANGLE_MATH_TEXT_COLOR, self.ANGLE_TEXT_THICKNESS)

    def get_pose_landmarks_style(self):
        """Returns a pose landmarks drawing style.

        Returns:
            A mapping from each pose landmark to its default drawing spec.
        """
        pose_landmark_style = {}
        left_spec = DrawingSpec(color=self.POSE_LANDMARKS_LEFT_COLOR, thickness=self.POSE_LANDMARKS_THICKNESS,
                                circle_radius=self.POSE_LANDMARKS_CIRCLE_RADIUS)
        right_spec = DrawingSpec(color=self.POSE_LANDMARKS_RIGHT_COLOR, thickness=self.POSE_LANDMARKS_THICKNESS,
                                 circle_radius=self.POSE_LANDMARKS_CIRCLE_RADIUS)
        for landmark in _POSE_LANDMARKS_LEFT:
            pose_landmark_style[landmark] = left_spec
        for landmark in _POSE_LANDMARKS_RIGHT:
            pose_landmark_style[landmark] = right_spec

        if not self._display_face_connections:
            empty_spec = DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)
            for landmark in self._FACE_LANDMARKS:
                pose_landmark_style[landmark] = empty_spec

        pose_landmark_style[PoseLandmark.NOSE] = \
            DrawingSpec(color=self.POSE_LANDMARKS_NOSE_COLOR, thickness=self.POSE_LANDMARKS_THICKNESS,
                        circle_radius=self.POSE_LANDMARKS_CIRCLE_RADIUS)
        return pose_landmark_style

    def get_graph_3d_environment(self, animated_azimuth: bool = True):
        """
        Draws landmarks and connections with matplotlib in a new image.

        Returns:
          A three channel RGB image represented as numpy ndarray with the body in a
          simulated environment made with matplotlib.
        """

        if self._plt_fig is None:
            # Initialize at first execution
            self._plt_fig = plt.figure()

        self._plt_fig.clf()  # Clear figure and configure plot
        ax = plt.axes(projection="3d")
        ax.set_xlim([-1, 2])
        ax.set_ylim([3, -2])
        ax.set_zlim([3, -2])
        ax.view_init(elev=5, azim=self._plt_azim, vertical_axis="y")
        if animated_azimuth:
            self._plt_azim += 3
            self._plt_azim = self._plt_azim % 360
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.get_zaxis().set_ticklabels([])

        if self.detected_pose.pose_landmarks:  # If MediaPipe detect a body pose

            # Zoom in/out axis
            top = self.computed_pose["pose_landmarks"][PoseLandmark.RIGHT_EYE]
            bottom = self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_HEEL]
            if self.computed_pose["pose_landmarks"][PoseLandmark.RIGHT_HEEL].y > bottom.y:
                bottom = self.computed_pose["pose_landmarks"][PoseLandmark.RIGHT_HEEL]
            right = self.computed_pose["pose_landmarks"][PoseLandmark.RIGHT_PINKY]
            left = self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_PINKY]
            ax.set_ylim([bottom.y, top.y - 0.5])
            ax.set_xlim([left.x + 0.5, right.x - 0.5])

            landmarks = self.computed_pose["pose_landmarks"]
            num_landmarks = len(landmarks)
            # Draw the connections
            for connection in self._mp_connections:
                start = connection[0]
                end = connection[1]

                if not (0 <= start < num_landmarks and 0 <= end < num_landmarks):
                    continue
                ax.plot3D(
                    xs=[landmarks[start].x, landmarks[end].x],
                    ys=[landmarks[start].y, landmarks[end].y],
                    zs=[landmarks[start].z, landmarks[end].z],
                    color=_normalize_color(self.PLOT_COLOR),
                    linewidth=self.PLOT_THICKNESS)

            # Draw calculated angles
            for landmark in landmarks:
                if landmark.angle is not None:
                    ax.text(landmark.x, landmark.y, landmark.z, str(int(landmark.angle)), fontsize=9)

        # Draw the render
        self._plt_fig.canvas.draw()

        # Plot canvas to a three channel RGB image represented as numpy ndarray
        image = np.fromstring(self._plt_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(self._plt_fig.canvas.get_width_height()[::-1] + (3,))
        image = image[70:400, 150:520]
        return image
