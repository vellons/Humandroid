import cv2
import math
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.drawing_utils import VISIBILITY_THRESHOLD
import numpy as np
import json

from humandroid.body_landmark import HumandroidBodyLandmark

# TODO: merge all these arrays in a JSON configuration
ANGLES = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),  # gomito sx
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),  # gomito dx
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),  # spalla sx
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),  # spalla dx
    # (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),  # anca sx
    # (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),  # anca dx
    # (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),  # ginocchio sx
    # (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),  # ginocchio dx
]

# WIP: (id, direct/indirect, offset)
ANGLES_WEBOTS_MOTOR_ID_OFFSET = [
    ("Left_Hand", -1, 180),
    ("Right_Hand", 1, 180),
    ("Left_Arm1", 1, 0),
    ("Right_Arm1", -1, 0),
    # ("Left_Hip", 1, 180),
    # ("Right_Hip", -1, 180),
]

Z_ANGLES = [
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),  # spalla sx
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),  # spalla dx
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),  # ginocchio sx
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE)  # ginocchio dx
]

# WIP: (id, direct/indirect, offset)
Z_ANGLES_WEBOTS_MOTOR_ID_OFFSET = [
    # ("Left_Shoulder", 1, 0),
    # ("Right_Shoulder", -1, 0),
]


class Humandroid:
    """
    Human pose detection to control humandroid robots.
    Works with MediaPipe technology

    Please refer to https://mediapipe.dev/ for more info about the pose processing.
    """

    _calc_z = False
    _upper_body_only = False  # TODO: at the moment only work for all body, wait for MediaPipe next release

    _mp_solution_pose = mp.solutions.pose
    _mp_solution_drawing = mp.solutions.drawing_utils
    _mp_pose = None  # MediaPipe object

    detected_pose = None  # Data directly form MediaPipe
    computed_pose = None  # Computed data for humandroid robot with servo angles

    _plt_fig = None  # Used inside draw_3d_environment()
    _webots_socket = None  # Used inside calc_and_send_angles_to_webots_socket()

    def __init__(self, static_image_mode=False, upper_body_only=False, calc_z=False):
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
        self._calc_z = calc_z
        self._upper_body_only = upper_body_only

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
                    landmark = HumandroidBodyLandmark(identifier=idx, name=str(PoseLandmark.value2member_map_[idx]))
                    self.computed_pose["pose_landmarks"].append(landmark)
                return self.computed_pose

            for idx, pose in enumerate(self.detected_pose.pose_landmarks.landmark):  # Copy pose info in landmark obj
                landmark = HumandroidBodyLandmark(
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
            self._mp_solution_pose.POSE_CONNECTIONS,  # UPPER_BODY_POSE_CONNECTIONS: available next release MediaPipe
            DrawingSpec(color=(255, 0, 0)),
            DrawingSpec(color=(0, 255, 0))
        )

    def _calc_angle_if_safe(self, points):
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
        result = math.fabs(math.degrees(a - b))  # Make angle always positive
        if result > 180:
            result = (360.0 - result)
        return int(result)

    def _calc_z_angle_if_safe(self, points):
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
        result = math.fabs(math.degrees(a - b))  # Make angle always positive
        if result > 180:
            result = (360.0 - result)
        return int(result)

    def process_angles(self):
        """
        Calculate focus angles from the current person detected.

        Returns:
          A dict object with a "pose_landmarks" field that contains the pose
          landmarks on the most prominent person detected and calculated angles.
        """

        for angle in ANGLES:
            a = self._calc_angle_if_safe(angle)
            self.computed_pose["pose_landmarks"][angle[1]].angle = a

        if self._calc_z:
            for angle in Z_ANGLES:
                a = self._calc_z_angle_if_safe(angle)
                self.computed_pose["pose_landmarks"][angle[1]].z_angle = a

        return self.computed_pose

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
                    cv2.putText(image, str(landmark.angle), pos, cv2.FONT_HERSHEY_SIMPLEX, font, (255, 0, 0), 2)

            if self._calc_z and landmark.z_angle is not None:  # Draw calculated z angles if enabled
                angle_x = landmark.x * image_width
                angle_y = landmark.y * image_height
                if margin <= angle_x <= image_width - margin and margin <= angle_y <= image_height - margin:
                    pos = (int(angle_x) + 20, int(angle_y) + 20)
                    cv2.putText(image, str(landmark.z_angle), pos, cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 2)

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
            bottom = self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_ANKLE]
            ax.set_ylim([bottom.y + 0.5, top.y - 0.5])

            landmarks = self.computed_pose["pose_landmarks"]
            num_landmarks = len(landmarks)
            # Draw the connections
            for connection in self._mp_solution_pose.POSE_CONNECTIONS:
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

    def calc_and_send_angles_to_webots_socket(self, host, port, block_on_error=True):
        """
        Work In Progress method.
        This is a TEST.
        This method create a socket communication to send the calculated motor's angle to Webots simulator.
        Download the simulator here: https://cyberbotics.com/
        This data can be used with Webots to control the "khr-2hv" robot: https://cyberbotics.com/doc/guide/khr-2hv
        You can import from: add > PROTO nodes (Webots Project) > robots > kondo > khr-2hv.
        See also webots/ folder.

        Args:
          host: IP for socket connection.
          port: port for socket connection.
          block_on_error: if true the program will raise an Exception if the socket communication fail.

        Returns:
          Return nothing but you can print the calculated angle inside self.computed_pose["webots_khw-2hv"].
        """

        if self._webots_socket is None:
            # Initialize at first execution
            print("Opening socket with: {}:{}".format(host, port))
            try:
                import socket
                self._webots_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._webots_socket.connect((host, port))
            except Exception as e:
                self._webots_socket = None
                if block_on_error:
                    raise Exception(e)
                else:
                    print("calc_and_send_angles_to_webots_socket(): {}".format(e))

        # TODO: merge ANGLES and Z_ANGLES array
        motors = []
        # x and y
        for idx, angle in enumerate(ANGLES):
            webots = ANGLES_WEBOTS_MOTOR_ID_OFFSET[idx]
            pose_angle = self.computed_pose["pose_landmarks"][angle[1]].angle
            # Calculate angle that works with the robot servomotor
            if pose_angle is not None:
                if webots[1] == 1:
                    motor = {"id": webots[0], "angle": pose_angle - webots[2]}
                else:
                    motor = {"id": webots[0], "angle": webots[2] - pose_angle}
                motors.append(motor)
            else:
                motor = {"id": webots[0], "angle": 0}  # Safe value
                motors.append(motor)

        # z
        if self._calc_z:
            # for idx, angle in enumerate(Z_ANGLES):
            #     webots = Z_ANGLES_WEBOTS_MOTOR_ID_OFFSET[idx]
            #     pose_angle = self.computed_pose["pose_landmarks"][angle[1]].z_angle
            #     # Calculate angle that works with the robot servomotor
            #     if pose_angle is not None:
            #         if webots[1] == 1:
            #             motor = {"id": webots[0], "angle": pose_angle - webots[2]}
            #         else:
            #             motor = {"id": webots[0], "angle": webots[2] - pose_angle}
            #         motors.append(motor)
            #     else:
            #         motor = {"id": webots[0], "angle": 0}  # Safe value
            #         motors.append(motor)

            if self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_KNEE].z_angle is not None:
                bottom_angle = 100 - self.computed_pose["pose_landmarks"][PoseLandmark.LEFT_KNEE].z_angle
                if bottom_angle < 0:
                    bottom_angle = 0
                # TODO: take this form JSON configuration
                motors.append({"id": "Left_Leg1", "angle": bottom_angle})
                motors.append({"id": "Left_Leg3", "angle": -bottom_angle * 2})
                motors.append({"id": "Left_Ankle", "angle": -bottom_angle})
                motors.append({"id": "Right_Leg1", "angle": -bottom_angle})
                motors.append({"id": "Right_leg3", "angle": bottom_angle * 2})
                motors.append({"id": "Right_Ankle", "angle": bottom_angle})

        self.computed_pose["webots_khw-2hv"] = json.dumps(motors)

        try:
            # Send information with socket
            self._webots_socket.sendall(str(self.computed_pose["webots_khw-2hv"]).encode())
        except Exception as e:
            if block_on_error:
                raise Exception(e)
            else:
                print("calc_and_send_angles_to_webots_socket(): {}".format(e))
                self._webots_socket = None  # Try to restart connection next time
